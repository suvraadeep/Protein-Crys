"""
Protein Crystallization Condition Predictor
Hybrid ESM-2 DL + XGBoost + LightGBM + CatBoost stacking ensemble

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import json
import re
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils.esm_embedder import ESMEmbedder
from utils.bio_features import BioFeatureExtractor, BIO_EXTRACTOR, _KD, _CHARGE_AT_7, _AA
from models.ph_model import PHModel
from models.salt_model import SaltModel
from models.peg_model import PEGModel
from models.temp_model import TempModel

# ── Directories ────────────────────────────────────────────────────────────────
PH_DIR   = ROOT / "ph"   / "models"
SALT_DIR = ROOT / "salt" / "models"
PEG_DIR  = ROOT / "peg"  / "models"
TEMP_DIR = ROOT / "temp" / "models"

_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

EXAMPLES = {
    "Hen Egg-White Lysozyme (14 kDa)":
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    "Human Ubiquitin (8.5 kDa)":
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "Bovine Pancreatic Trypsin Inhibitor":
        "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA",
    "Thaumatin (22 kDa)":
        "ATFECRLRNFSAPEQSCRFIQPEGCSGPGGLKCDKHFKTIPLWMDVNSLSCRKLNSGKAKRKNGTDCPKFVLAPNLALTSGKKLKLCNKQDCNFQLNRPKVPFKFEGSTSQIDCKPKDSTFKIDFAKLFRDKDQLKAVQNCKVKDYKNTYLNQLNFQIDQRHLRPLRLRNLDWGDLSCGKSKLVDVKEPNGEGHKLRSPGKIDVLNKLLNNTEKSGPYQMPLDVIGPAQISQKLVAGFLNVNPEFSGPVNNIYNLNNPQCHLKK",
}

# ── AA colour groups for sequence colouring ───────────────────────────────────
_AA_COLOR = {
    # hydrophobic → orange
    **{aa: "#F4A460" for aa in "AVILMFYW"},
    # positive → cornflower blue
    **{aa: "#6495ED" for aa in "KRH"},
    # negative → salmon
    **{aa: "#FA8072" for aa in "DE"},
    # polar → mediumseagreen
    **{aa: "#3CB371" for aa in "STNQ"},
    # special → orchid
    "G": "#DA70D6", "P": "#DA70D6", "C": "#FFD700",
}

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrystalNet — Protein Crystallization Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.metric-label { font-size: 0.8rem !important; }
.stMetric { background: #f8f9fa; border-radius: 8px; padding: 8px; }
div[data-testid="stExpander"] { border: 1px solid #ddd; border-radius: 8px; }
.seq-char { font-family: monospace; font-size: 13px; padding: 1px 0; }
</style>
""", unsafe_allow_html=True)


# ── Sequence fetching ──────────────────────────────────────────────────────────

def _parse_fasta(text: str) -> tuple[str, str]:
    """Returns (header, sequence) from FASTA text."""
    lines = text.strip().splitlines()
    header = lines[0].lstrip(">") if lines else ""
    seq = "".join(l.strip() for l in lines[1:] if not l.startswith(">"))
    return header, seq.upper()


def fetch_uniprot(uid: str) -> tuple[str, str, str]:
    """Fetches (name, organism, sequence) from UniProt REST API."""
    try:
        import urllib.request
        url = f"https://rest.uniprot.org/uniprotkb/{uid.strip().upper()}.fasta"
        with urllib.request.urlopen(url, timeout=10) as r:
            text = r.read().decode()
        header, seq = _parse_fasta(text)
        parts = header.split("|")
        name = parts[2].split(" OS=")[0].strip() if len(parts) > 2 else header
        org  = re.search(r"OS=(.*?)\s+OX=", header)
        organism = org.group(1) if org else "Unknown"
        return name, organism, seq
    except Exception as e:
        return "", "", f"ERROR:{e}"


def fetch_pdb(pdb_id: str) -> tuple[str, str]:
    """Fetches sequence from RCSB PDB FASTA endpoint."""
    try:
        import urllib.request
        url = f"https://www.rcsb.org/fasta/entry/{pdb_id.strip().upper()}/display"
        with urllib.request.urlopen(url, timeout=10) as r:
            text = r.read().decode()
        header, seq = _parse_fasta(text)
        return header, seq
    except Exception as e:
        return "", f"ERROR:{e}"


def _clean_seq(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in _VALID_AA)


def _validate(seq: str) -> tuple[bool, str]:
    seq = _clean_seq(seq)
    if not seq:
        return False, "Empty or no valid residues."
    if len(seq) < 20:
        return False, f"Too short ({len(seq)} aa). Minimum: 20."
    if len(seq) > 2000:
        return False, f"Too long ({len(seq)} aa). Maximum: 2000."
    return True, seq


# ── Model loading (cached) ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ESM-2 backbone and ensemble models…")
def load_models():
    embedder = ESMEmbedder(cache_dir=ROOT / "embeddings_cache")
    m = {}

    # pH
    if (PH_DIR / "ph_dl.pt").exists():
        dl = PHModel()
        dl.load_state_dict(torch.load(PH_DIR / "ph_dl.pt", map_location="cpu"))
        dl.eval()
        m["ph_dl"]     = dl
        m["ph_scaler"] = joblib.load(PH_DIR / "ph_scaler.joblib")
        m["ph_xgb"]    = joblib.load(PH_DIR / "ph_xgb.joblib")
        m["ph_lgb"]    = joblib.load(PH_DIR / "ph_lgb.joblib")
        m["ph_meta"]   = joblib.load(PH_DIR / "ph_meta.joblib")

    # Salt type + concentration
    if (SALT_DIR / "salt_type_dl.pt").exists():
        with open(SALT_DIR / "salt_classes.json") as f:
            salt_classes = json.load(f)
        n_sc = len(salt_classes)
        dl_clf = PEGModel(n_classes=n_sc)
        dl_clf.load_state_dict(torch.load(SALT_DIR / "salt_type_dl.pt", map_location="cpu"))
        dl_clf.eval()
        m["salt_type_dl"]  = dl_clf
        m["salt_type_xgb"] = joblib.load(SALT_DIR / "salt_type_xgb.joblib")
        m["salt_type_lgb"] = joblib.load(SALT_DIR / "salt_type_lgb.joblib")
        m["salt_scaler"]   = joblib.load(SALT_DIR / "salt_scaler.joblib")
        m["salt_classes"]  = salt_classes
        m["salt_le"]       = joblib.load(SALT_DIR / "salt_label_encoder.joblib")

        dl_reg = SaltModel()
        dl_reg.load_state_dict(torch.load(SALT_DIR / "salt_conc_dl.pt", map_location="cpu"))
        dl_reg.eval()
        m["salt_conc_dl"]     = dl_reg
        m["salt_conc_xgb"]    = joblib.load(SALT_DIR / "salt_conc_xgb.joblib")
        m["salt_conc_lgb"]    = joblib.load(SALT_DIR / "salt_conc_lgb.joblib")
        m["salt_conc_meta"]   = joblib.load(SALT_DIR / "salt_conc_meta.joblib")
        m["salt_conc_scaler"] = joblib.load(SALT_DIR / "salt_conc_scaler.joblib")

    # PEG
    if (PEG_DIR / "peg_dl.pt").exists():
        with open(PEG_DIR / "peg_label_map.json") as f:
            peg_label_map = json.load(f)
        n_peg = len(peg_label_map)
        peg_dl = PEGModel(n_classes=n_peg)
        peg_dl.load_state_dict(torch.load(PEG_DIR / "peg_dl.pt", map_location="cpu"))
        peg_dl.eval()
        m["peg_dl"]        = peg_dl
        m["peg_xgb"]       = joblib.load(PEG_DIR / "peg_xgb.joblib")
        m["peg_lgb"]       = joblib.load(PEG_DIR / "peg_lgb.joblib")
        m["peg_scaler"]    = joblib.load(PEG_DIR / "peg_scaler.joblib")
        m["peg_label_map"] = peg_label_map
        m["peg_inv_map"]   = {v: k for k, v in peg_label_map.items()}

    # Temperature
    if (TEMP_DIR / "temp_dl.pt").exists():
        temp_dl = TempModel()
        temp_dl.load_state_dict(torch.load(TEMP_DIR / "temp_dl.pt", map_location="cpu"))
        temp_dl.eval()
        m["temp_dl"]     = temp_dl
        m["temp_xgb"]    = joblib.load(TEMP_DIR / "temp_xgb.joblib")
        m["temp_lgb"]    = joblib.load(TEMP_DIR / "temp_lgb.joblib")
        m["temp_scaler"] = joblib.load(TEMP_DIR / "temp_scaler.joblib")
        m["temp_meta"]   = joblib.load(TEMP_DIR / "temp_meta.joblib")
        try:
            import catboost as cb
            cb_m = cb.CatBoostRegressor()
            cb_m.load_model(str(TEMP_DIR / "temp_cb.cbm"))
            m["temp_cb"] = cb_m
        except Exception:
            pass

    return embedder, m


# ── Feature extraction ─────────────────────────────────────────────────────────

def make_feature_vector(seq: str, pdb_id: str, embedder) -> np.ndarray:
    emb = embedder.embed_sequence(seq, pdb_id)   # [320]
    bio = BIO_EXTRACTOR.extract(seq)             # [48]
    return np.concatenate([emb, bio])[np.newaxis, :].astype(np.float32)  # [1, 368]


# ── Prediction pipeline ────────────────────────────────────────────────────────

def run_predictions(seq: str, embedder, models: dict) -> dict:
    pdb_id = f"_APP_{abs(hash(seq)) % 100000}"
    x   = make_feature_vector(seq, pdb_id, embedder)   # [1, 368]
    x_t = torch.tensor(x, dtype=torch.float32)
    res = {}

    if "ph_dl" in models:
        x_sc = models["ph_scaler"].transform(x)
        with torch.no_grad():
            dl_p = float(models["ph_dl"](x_t).item())
        xgb_p = float(models["ph_xgb"].predict(x_sc)[0])
        lgb_p = float(models["ph_lgb"].predict(x_sc)[0])
        ph    = float(models["ph_meta"].predict([[dl_p, xgb_p, lgb_p]])[0])
        res["pH"] = round(float(np.clip(ph, 2.0, 12.0)), 2)

    if "salt_type_dl" in models:
        x_sc = models["salt_scaler"].transform(x)
        with torch.no_grad():
            dl_probs = torch.softmax(models["salt_type_dl"](x_t), dim=-1).numpy()[0]
        xgb_probs = models["salt_type_xgb"].predict_proba(x_sc)[0]
        lgb_probs = models["salt_type_lgb"].predict_proba(x_sc)[0]
        ens_probs = (dl_probs + xgb_probs + lgb_probs) / 3
        classes   = models["salt_classes"]
        salt_type = classes[int(ens_probs.argmax())]
        res["salt_type"]       = salt_type
        res["salt_type_probs"] = {c: float(p) for c, p in zip(classes, ens_probs)}

        n_sc    = len(classes)
        le      = models["salt_le"]
        onehot  = np.zeros((1, n_sc), dtype=np.float32)
        onehot[0, le.transform([salt_type])[0]] = 1
        x_conc    = np.hstack([x, onehot])
        x_conc_sc = models["salt_conc_scaler"].transform(x_conc)
        with torch.no_grad():
            dl_log = float(models["salt_conc_dl"](x_t).item())
        xgb_log  = float(models["salt_conc_xgb"].predict(x_conc_sc)[0])
        lgb_log  = float(models["salt_conc_lgb"].predict(x_conc_sc)[0])
        conc_log = float(models["salt_conc_meta"].predict([[dl_log, xgb_log, lgb_log, xgb_log]])[0])
        res["salt_M"] = round(float(np.expm1(conc_log)), 4)

    if "peg_dl" in models:
        x_sc = models["peg_scaler"].transform(x)
        with torch.no_grad():
            dl_probs = torch.softmax(models["peg_dl"](x_t), dim=-1).numpy()[0]
        xgb_probs  = models["peg_xgb"].predict_proba(x_sc)[0]
        lgb_probs  = models["peg_lgb"].predict_proba(x_sc)[0]
        ens_probs  = (dl_probs + xgb_probs + lgb_probs) / 3
        inv_map    = models["peg_inv_map"]
        res["peg_type"]  = inv_map[int(ens_probs.argmax())]
        res["peg_probs"] = {inv_map[i]: float(p) for i, p in enumerate(ens_probs)}

    if "temp_dl" in models:
        x_sc = models["temp_scaler"].transform(x)
        with torch.no_grad():
            dl_p = float(models["temp_dl"](x_t).item())
        xgb_p = float(models["temp_xgb"].predict(x_sc)[0])
        lgb_p = float(models["temp_lgb"].predict(x_sc)[0])
        cols  = [dl_p, xgb_p, lgb_p]
        if "temp_cb" in models:
            cols.append(float(models["temp_cb"].predict(x_sc)[0]))
        temp_k = float(models["temp_meta"].predict([cols])[0])
        res["temp_k"] = round(temp_k, 1)
        res["temp_c"] = round(temp_k - 273.15, 1)

    return res


# ── Plotly visualization helpers ───────────────────────────────────────────────

def _ph_gauge(ph: float, pi_norm: float) -> go.Figure:
    pi = pi_norm * 14.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ph,
        delta={"reference": pi, "prefix": "pI=", "valueformat": ".1f"},
        title={"text": "Predicted pH", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 14], "tickwidth": 1},
            "bar":  {"color": "#4C72B0", "thickness": 0.3},
            "steps": [
                {"range": [0,  4],  "color": "#FF6B6B"},
                {"range": [4,  6],  "color": "#FFD93D"},
                {"range": [6,  8],  "color": "#6BCB77"},
                {"range": [8,  10], "color": "#4D96FF"},
                {"range": [10, 14], "color": "#845EC2"},
            ],
            "threshold": {
                "line": {"color": "#FF4800", "width": 3},
                "thickness": 0.75,
                "value": pi,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=20, l=20, r=20))
    return fig


def _salt_gauge(conc: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conc,
        number={"suffix": " M", "valueformat": ".3f"},
        title={"text": "Salt Concentration", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 4], "tickwidth": 1,
                     "tickvals": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]},
            "bar":  {"color": "#E8874A", "thickness": 0.3},
            "steps": [
                {"range": [0,    0.5],  "color": "#FFF3E0"},
                {"range": [0.5,  1.5],  "color": "#FFCC80"},
                {"range": [1.5,  2.5],  "color": "#FFA726"},
                {"range": [2.5,  4.0],  "color": "#EF6C00"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=20, l=20, r=20))
    return fig


def _temp_gauge(temp_c: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=temp_c,
        number={"suffix": " °C", "valueformat": ".1f"},
        title={"text": "Crystallization Temperature", "font": {"size": 18}},
        gauge={
            "axis": {"range": [-5, 45], "tickwidth": 1,
                     "tickvals": [0, 4, 10, 20, 25, 37, 45]},
            "bar":  {"color": "#7B5EA7", "thickness": 0.3},
            "steps": [
                {"range": [-5, 6],  "color": "#D6EAF8"},
                {"range": [6,  15], "color": "#AED6F1"},
                {"range": [15, 25], "color": "#85C1E9"},
                {"range": [25, 37], "color": "#5499C7"},
                {"range": [37, 45], "color": "#1B4F72"},
            ],
            "threshold": {
                "line": {"color": "#E74C3C", "width": 3},
                "thickness": 0.75,
                "value": 25,
            },
        },
    ))
    fig.add_annotation(text="◄ Cold room  |  RT ►  |  Warm ►",
                       xref="paper", yref="paper", x=0.5, y=-0.12,
                       showarrow=False, font=dict(size=10, color="gray"))
    fig.update_layout(height=280, margin=dict(t=40, b=40, l=20, r=20))
    return fig


def _peg_bar(peg_probs: dict, pred: str) -> go.Figure:
    items = sorted(peg_probs.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in items]
    probs = [v for _, v in items]
    colors = ["#2ECC71" if n == pred else "#AED6F1" for n in names]
    fig = go.Figure(go.Bar(
        x=probs, y=names, orientation="h",
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title={"text": "PEG Type Probability", "font": {"size": 18}},
        xaxis={"title": "Probability", "range": [0, 1.15]},
        yaxis={"autorange": "reversed"},
        height=max(200, 50 * len(names)),
        margin=dict(t=40, b=20, l=20, r=60),
        plot_bgcolor="white",
    )
    return fig


def _aa_composition_bar(seq: str) -> go.Figure:
    cnt = {aa: seq.count(aa) / len(seq) for aa in _AA}
    groups = {
        "Hydrophobic": list("AVILMFYW"),
        "Positive":    list("KRH"),
        "Negative":    list("DE"),
        "Polar":       list("STNQ"),
        "Special":     list("GPC"),
    }
    grp_colors = {
        "Hydrophobic": "#F4A460",
        "Positive":    "#6495ED",
        "Negative":    "#FA8072",
        "Polar":       "#3CB371",
        "Special":     "#DA70D6",
    }
    aas, vals, cols = [], [], []
    for grp, members in groups.items():
        for aa in members:
            aas.append(aa)
            vals.append(cnt.get(aa, 0))
            cols.append(grp_colors[grp])

    fig = go.Figure(go.Bar(
        x=aas, y=vals, marker_color=cols,
        text=[f"{v*100:.1f}%" for v in vals],
        textposition="outside",
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Amino Acid Composition",
        yaxis={"title": "Fraction", "range": [0, max(vals) * 1.3 if vals else 0.2]},
        xaxis={"title": "Amino acid"},
        height=320,
        plot_bgcolor="white",
        margin=dict(t=40, b=40, l=40, r=20),
    )
    # Legend annotations
    for i, (grp, col) in enumerate(grp_colors.items()):
        fig.add_annotation(
            text=grp, x=0.02 + i * 0.2, y=1.08, xref="paper", yref="paper",
            showarrow=False,
            font=dict(color=col, size=11, family="Arial Bold"),
        )
    return fig


def _hydrophobicity_plot(seq: str, window: int = 9) -> go.Figure:
    kd = [_KD.get(aa, 0) for aa in seq]
    half = window // 2
    scores, positions = [], []
    for i in range(half, len(seq) - half):
        scores.append(np.mean(kd[i - half: i + half + 1]))
        positions.append(i + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=positions, y=scores, mode="lines",
        line=dict(color="#E8874A", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(232,135,74,0.15)",
        name="Hydrophobicity",
        hovertemplate="Position %{x}: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=f"Kyte-Doolittle Hydrophobicity (window={window})",
        xaxis={"title": "Residue position"},
        yaxis={"title": "Score (KD)", "zeroline": False},
        height=280,
        plot_bgcolor="white",
        margin=dict(t=40, b=40, l=50, r=20),
    )
    return fig


def _charge_profile(seq: str) -> go.Figure:
    charges = [_CHARGE_AT_7.get(aa, 0) for aa in seq]
    cumulative = np.cumsum(charges)
    positions  = list(range(1, len(seq) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=positions, y=cumulative, mode="lines",
        line=dict(color="#4C72B0", width=2),
        fill="tozeroy",
        fillcolor="rgba(76,114,176,0.15)",
        name="Cumulative charge",
        hovertemplate="Position %{x}: net charge=%{y:.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Cumulative Charge Distribution (pH 7)",
        xaxis={"title": "Residue position"},
        yaxis={"title": "Net charge"},
        height=280,
        plot_bgcolor="white",
        margin=dict(t=40, b=40, l=50, r=20),
    )
    return fig


def _coloured_sequence_html(seq: str, wrap: int = 80) -> str:
    spans = []
    for i, aa in enumerate(seq):
        col = _AA_COLOR.get(aa, "#999")
        spans.append(
            f'<span title="{aa}{i+1}" '
            f'style="color:{col};font-family:monospace;font-size:13px;">{aa}</span>'
        )
        if (i + 1) % wrap == 0:
            spans.append("<br>")
    return "".join(spans)


# ── Main UI ────────────────────────────────────────────────────────────────────

st.title("🔬 CrystalNet — Protein Crystallization Predictor")
st.markdown(
    "Predict **pH · salt concentration · PEG type · temperature** from protein sequence "
    "using a hybrid **ESM-2 Transformer + XGBoost + LightGBM** stacking ensemble "
    "guided by protein crystallization biochemistry."
)

embedder, models = load_models()

if not models:
    st.warning(
        "No trained models found. Train them first:\n"
        "```\npython ph/train.py\npython salt/train.py\n"
        "python peg/train.py\npython temp/train.py\n```"
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Sequence Input")
    input_mode = st.radio(
        "How to provide sequence",
        ["Paste sequence", "UniProt ID", "PDB ID"],
        index=0,
    )

    if input_mode == "UniProt ID":
        uid = st.text_input("UniProt Accession (e.g. P00698)", max_chars=12)
        if st.button("Fetch from UniProt"):
            if uid.strip():
                with st.spinner("Fetching from UniProt…"):
                    name, org, seq_fetched = fetch_uniprot(uid)
                if seq_fetched.startswith("ERROR:"):
                    st.error(f"Failed: {seq_fetched[6:]}")
                else:
                    st.session_state["seq_input"]  = seq_fetched
                    st.session_state["seq_name"]   = name
                    st.session_state["seq_org"]    = org
                    st.success(f"Loaded: {name}")
            else:
                st.warning("Enter a UniProt ID first.")

    elif input_mode == "PDB ID":
        pdb_input = st.text_input("PDB ID (e.g. 1LYZ)", max_chars=8)
        if st.button("Fetch from PDB"):
            if pdb_input.strip():
                with st.spinner("Fetching from RCSB PDB…"):
                    header, seq_fetched = fetch_pdb(pdb_input)
                if seq_fetched.startswith("ERROR:"):
                    st.error(f"Failed: {seq_fetched[6:]}")
                else:
                    st.session_state["seq_input"] = seq_fetched
                    st.session_state["seq_name"]  = header.split("|")[0][:60]
                    st.success("Sequence loaded.")
            else:
                st.warning("Enter a PDB ID first.")

    st.divider()
    st.subheader("Examples")
    chosen = st.selectbox("Load example", ["— none —"] + list(EXAMPLES.keys()))
    if chosen != "— none —":
        st.session_state["seq_input"] = EXAMPLES[chosen]
        st.session_state["seq_name"]  = chosen

    st.divider()
    st.subheader("Model Status")
    status = {
        "pH": "ph_dl" in models,
        "Salt type": "salt_type_dl" in models,
        "Salt conc.": "salt_conc_dl" in models,
        "PEG type": "peg_dl" in models,
        "Temperature": "temp_dl" in models,
    }
    for lbl, ok in status.items():
        icon = "✅" if ok else "❌"
        st.markdown(f"{icon} {lbl}")

    st.divider()
    st.caption(
        "**Architecture:** ESM-2 (320-D) + 48-D bio features → "
        "Ridge meta-learner stacking over DL + XGB + LGB folds"
    )


# ── Sequence input area ────────────────────────────────────────────────────────
default_seq = st.session_state.get("seq_input", "")
seq_name    = st.session_state.get("seq_name", "")

seq_text = st.text_area(
    "Protein sequence",
    value=default_seq,
    height=140,
    placeholder="Paste sequence (single-letter code) or use the sidebar to fetch by UniProt / PDB ID …",
    help="20 standard amino acids only. Gaps and non-standard characters are stripped.",
)

col_btn, col_name = st.columns([1, 3])
with col_btn:
    run_btn = st.button("🚀  Predict", type="primary", use_container_width=True)
with col_name:
    if seq_name:
        st.markdown(f"**Source:** {seq_name}")


if run_btn:
    ok, seq_or_err = _validate(seq_text)
    if not ok:
        st.error(f"Input error: {seq_or_err}")
        st.stop()

    seq = seq_or_err

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION A — Protein Profile
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🧬 Section A — Protein Profile")

    bio = BIO_EXTRACTOR.extract(seq)
    pi_val = bio[22] * 14.0
    mw_val = bio[23] * 100_000
    gravy  = bio[21]
    net_ch = bio[29] * 50.0
    length = len(seq)

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Length", f"{length} aa")
    a2.metric("Mol. Weight", f"{mw_val/1000:.1f} kDa")
    a3.metric("pI (isoelectric point)", f"{pi_val:.2f}")
    a4.metric("GRAVY index", f"{gravy:.2f}", help="Kyte-Doolittle: positive=hydrophobic")
    a5.metric("Net charge (pH 7)", f"{net_ch:+.1f}")

    with st.expander("Coloured sequence (hydrophobic=🟠 positive=🔵 negative=🔴 polar=🟢 special=🟣)"):
        st.markdown(_coloured_sequence_html(seq), unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION B — Molecular Fingerprint
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Section B — Molecular Fingerprint")

    b1, b2, b3 = st.columns(3)
    with b1:
        st.plotly_chart(_aa_composition_bar(seq), use_container_width=True)
    with b2:
        if len(seq) >= 9:
            st.plotly_chart(_hydrophobicity_plot(seq), use_container_width=True)
        else:
            st.info("Sequence too short for hydrophobicity window plot (min 9 aa).")
    with b3:
        st.plotly_chart(_charge_profile(seq), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION C — Predictions
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔮 Section C — Predicted Crystallization Conditions")

    if not models:
        st.warning("No trained models available. Train the models first.")
    else:
        with st.spinner("Running ESM-2 + XGB + LGB ensemble inference…"):
            results = run_predictions(seq, embedder, models)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if "pH" in results:
                st.plotly_chart(_ph_gauge(results["pH"], bio[22]), use_container_width=True)
                ph_zone = (
                    "Strongly acidic" if results["pH"] < 4 else
                    "Acidic" if results["pH"] < 6 else
                    "Near-neutral" if results["pH"] < 8 else
                    "Basic" if results["pH"] < 10 else
                    "Strongly alkaline"
                )
                st.caption(f"Zone: **{ph_zone}** · pI = {pi_val:.1f}")
            else:
                st.info("pH model not loaded.")

        with c2:
            if "salt_M" in results:
                st.plotly_chart(_salt_gauge(results["salt_M"]), use_container_width=True)
                st.metric("Predicted salt", results.get("salt_type", "—"))
                st.caption(f"≈ {results['salt_M']*1000:.0f} mM")
            else:
                st.info("Salt model not loaded.")

        with c3:
            if "peg_type" in results:
                st.plotly_chart(
                    _peg_bar(results["peg_probs"], results["peg_type"]),
                    use_container_width=True,
                )
            else:
                st.info("PEG model not loaded.")

        with c4:
            if "temp_c" in results:
                st.plotly_chart(_temp_gauge(results["temp_c"]), use_container_width=True)
                room_note = (
                    "❄️ Cold room (4 °C)" if results["temp_c"] < 8 else
                    "🌡 Room temperature" if results["temp_c"] < 26 else
                    "🔥 Warm incubator"
                )
                st.caption(room_note)
            else:
                st.info("Temperature model not loaded.")

        # Salt type probability breakdown
        if "salt_type_probs" in results:
            with st.expander("Full salt type probability breakdown"):
                probs_df = (
                    pd.DataFrame(results["salt_type_probs"].items(), columns=["Salt", "Prob"])
                    .sort_values("Prob", ascending=False)
                    .reset_index(drop=True)
                )
                probs_df["Prob %"] = (probs_df["Prob"] * 100).round(1)
                fig_salt = px.bar(
                    probs_df, x="Prob", y="Salt", orientation="h",
                    color="Prob", color_continuous_scale="oranges",
                    text="Prob %",
                )
                fig_salt.update_traces(texttemplate="%{text}%", textposition="outside")
                fig_salt.update_layout(
                    height=max(250, 40 * len(probs_df)),
                    showlegend=False, coloraxis_showscale=False,
                    plot_bgcolor="white",
                    margin=dict(l=20, r=60, t=20, b=20),
                )
                st.plotly_chart(fig_salt, use_container_width=True)

        with st.expander("Raw JSON output"):
            st.json(results)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION D — Biology Explanations
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔬 Section D — The Biology Behind the Predictions")

    d1, d2 = st.columns(2)

    with d1:
        with st.expander("Why this pH? — pI relationship", expanded=False):
            st.markdown(f"""
**Isoelectric point (pI) of this protein: {pi_val:.2f}**

Proteins have minimum solubility and maximum likelihood of crystallization
when the solution pH is **near the protein's pI**.  At the pI, the net charge
is zero, electrostatic repulsion between molecules is minimised, and they pack
more readily into an ordered crystal lattice.

The predicted crystallization pH is **{results.get("pH", "N/A")}**.
The Henderson-Hasselbalch model predicts that your protein will carry
**net charge ≈ {net_ch:+.1f}** at pH 7.

*Rule of thumb:* aim for pH = pI ± 1 unit with a buffering salt.
Typical buffers: citrate (pH 3–6), acetate (pH 4–5.5), HEPES (pH 7–8), Tris (pH 8–9).
""")

        with st.expander("Why this PEG size? — excluded-volume theory", expanded=False):
            peg_name = results.get("peg_type", "PEG_3350")
            st.markdown(f"""
**Predicted PEG type: {peg_name}**

Polyethylene glycol (PEG) promotes crystallization via **excluded-volume (depletion)**
interactions.  PEG polymers are excluded from the volume immediately surrounding a
protein, creating an osmotic pressure that drives protein-protein contacts.

The correct PEG molecular weight depends on protein size:
| Protein MW | Recommended PEG |
|---|---|
| < 10 kDa | PEG 400 – PEG 1000 |
| 10 – 30 kDa | PEG 1500 – PEG 3350 |
| 30 – 60 kDa | PEG 3350 – PEG 4000 |
| > 60 kDa | PEG 6000 – PEG 8000 |

Your protein MW ≈ **{mw_val/1000:.1f} kDa** → model predicts **{peg_name}**.
""")

    with d2:
        with st.expander("Why this salt? — Hofmeister series", expanded=False):
            salt_name = results.get("salt_type", "Ammonium Sulfate")
            conc_val  = results.get("salt_M", "N/A")
            st.markdown(f"""
**Predicted salt: {salt_name} at {conc_val} M**

The **Hofmeister series** ranks ions by their ability to precipitate proteins
('salting-out').  The mechanism is dehydration of the protein surface:
kosmotropic (structure-making) ions strip water from protein surfaces,
reducing protein-solvent interactions and promoting crystal contacts.

Hofmeister order (most → least precipitating):
`SO₄²⁻ > HPO₄²⁻ > F⁻ > Cl⁻ > Br⁻ > I⁻ > SCN⁻`

Your protein has a **kosmotropic score of {bio[31]:.2f}**
(fraction of G, A, S, T, P, V, I, L residues).
Higher kosmotropic scores generally require lower salt concentrations.
""")

        with st.expander("Why this temperature? — thermostability indices", expanded=False):
            temp_c_val = results.get("temp_c", "N/A")
            thermo_idx = bio[37]
            aliph_idx  = bio[26] * 200
            st.markdown(f"""
**Predicted temperature: {temp_c_val} °C**

Two composite indices encode thermostability from the sequence:

1. **IVYWREL index** = {thermo_idx:.2f}
   Fraction of Ile, Val, Tyr, Trp, Arg, Glu, Leu — enriched in thermophilic proteins.

2. **Aliphatic index** = {aliph_idx:.0f}
   [Ala + 2.9×Val + 3.9×(Ile+Leu)] / len × 100.
   High values → thermostability through hydrophobic core packing.

**Interpretation:**
- IVYWREL > 0.40 and aliphatic > 90 → likely stable at room temperature (20–25 °C)
- Lower indices → recommend 4 °C (cold room) to prevent aggregation during setup
- Very high thermostability → can try 30–37 °C for faster nucleation

Your protein: IVYWREL = {thermo_idx:.2f}, Aliphatic = {aliph_idx:.0f}
""")
