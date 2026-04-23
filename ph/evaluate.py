"""
pH model evaluation — loads all three base models + meta-learner
and reports test metrics on a held-out 15% split.

Usage:
    python ph/evaluate.py
"""

import sys
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.data_parser import load_and_merge_datasets
from utils.esm_embedder import ESMEmbedder
from utils.dataset import PhysicochemicalExtractor
from models.ph_model import PHModel

SAVE_DIR = ROOT / "ph" / "models"
PHYS = PhysicochemicalExtractor()


def build_feature_matrix(df, embedder):
    rows = []
    for _, row in df.iterrows():
        emb  = embedder.embed_sequence(row["sequence"], row["pdb_id"])
        phys = PHYS.extract(row["sequence"])
        rows.append(np.concatenate([emb, phys]))
    return np.stack(rows).astype(np.float32)


def main():
    print("=" * 55)
    print("pH HYBRID MODEL — EVALUATION")
    print("=" * 55)

    df = load_and_merge_datasets()
    df = df[df["pH"].notna() & df["pH"].between(2.0, 12.0)].reset_index(drop=True)
    y  = df["pH"].values.astype(np.float32)

    embedder = ESMEmbedder()
    X = build_feature_matrix(df, embedder)

    # Use same seed/split as training OOF final 15% test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    scaler = joblib.load(SAVE_DIR / "ph_scaler.joblib")
    meta   = joblib.load(SAVE_DIR / "ph_meta.joblib")
    X_test_sc = scaler.transform(X_test)

    preds_by_model = {}

    # DL model
    dl = PHModel(); dl.load_state_dict(torch.load(SAVE_DIR / "ph_dl.pt", map_location="cpu"))
    dl.eval()
    with torch.no_grad():
        preds_dl = dl(torch.tensor(X_test, dtype=torch.float32)).numpy()
    preds_by_model["DL"] = preds_dl

    # XGB
    xgb_m = joblib.load(SAVE_DIR / "ph_xgb.joblib")
    preds_by_model["XGBoost"] = xgb_m.predict(X_test_sc)

    # LGB
    lgb_m = joblib.load(SAVE_DIR / "ph_lgb.joblib")
    preds_by_model["LightGBM"] = lgb_m.predict(X_test_sc)

    # Ensemble
    S = np.column_stack(list(preds_by_model.values()))
    preds_by_model["Ensemble"] = meta.predict(S)

    print(f"\n{'Model':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 42)
    for name, preds in preds_by_model.items():
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        tag  = " ←" if name == "Ensemble" else ""
        print(f"{name:<12} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}{tag}")


if __name__ == "__main__":
    main()
