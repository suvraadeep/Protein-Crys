"""
PEG type evaluation — reports per-class accuracy + macro-F1.

Usage:
    python peg/evaluate.py
"""

import sys
import json
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils.data_parser import load_and_merge_datasets, get_peg_label_mapping
from utils.esm_embedder import ESMEmbedder
from utils.dataset import PhysicochemicalExtractor
from models.peg_model import PEGModel
from peg.train import build_feature_matrix

SAVE_DIR = ROOT / "peg" / "models"
PHYS = PhysicochemicalExtractor()


def main():
    print("=" * 55)
    print("PEG TYPE HYBRID MODEL — EVALUATION")
    print("=" * 55)

    df = load_and_merge_datasets()
    df = df[df["peg_class"].notna()].reset_index(drop=True)

    with open(SAVE_DIR / "peg_label_map.json") as f:
        label_map = json.load(f)
    inv_map   = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)
    y = np.array([label_map[c] for c in df["peg_class"]], dtype=np.int64)

    embedder = ESMEmbedder()
    X = build_feature_matrix(df, embedder)
    scaler = joblib.load(SAVE_DIR / "peg_scaler.joblib")
    X_sc   = scaler.transform(X)

    _, te = train_test_split(np.arange(len(y)), test_size=0.15,
                             random_state=42, stratify=y)

    dl_m = PEGModel(n_classes=n_classes)
    dl_m.load_state_dict(torch.load(SAVE_DIR / "peg_dl.pt", map_location="cpu"))
    dl_m.eval()
    with torch.no_grad():
        dl_probs = torch.softmax(dl_m(torch.tensor(X[te])), dim=-1).numpy()

    xgb_m = joblib.load(SAVE_DIR / "peg_xgb.joblib")
    lgb_m = joblib.load(SAVE_DIR / "peg_lgb.joblib")
    xgb_probs = xgb_m.predict_proba(X_sc[te])
    lgb_probs = lgb_m.predict_proba(X_sc[te])
    ens_probs = (dl_probs + xgb_probs + lgb_probs) / 3

    print(f"\n{'Model':<12} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 36)
    for name, probs in [("DL", dl_probs), ("XGBoost", xgb_probs),
                         ("LightGBM", lgb_probs), ("Ensemble", ens_probs)]:
        acc = accuracy_score(y[te], probs.argmax(1))
        f1  = f1_score(y[te], probs.argmax(1), average="macro", zero_division=0)
        tag = " ←" if name == "Ensemble" else ""
        print(f"{name:<12} {acc:>10.4f} {f1:>10.4f}{tag}")

    print("\nClassification report (Ensemble):")
    class_names = [inv_map[i] for i in range(n_classes)]
    print(classification_report(y[te], ens_probs.argmax(1),
                                 target_names=class_names, zero_division=0))


if __name__ == "__main__":
    main()
