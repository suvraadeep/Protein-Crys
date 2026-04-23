"""
Salt model evaluation — reports test metrics for both
Model A (salt type classifier) and Model B (concentration regressor).

Usage:
    python salt/evaluate.py
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
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score,
)

from utils.data_parser import load_and_merge_datasets
from utils.esm_embedder import ESMEmbedder
from utils.dataset import PhysicochemicalExtractor
from models.salt_model import SaltModel
from models.peg_model import PEGModel
from salt.train import clean_salt_data, build_feature_matrix, SALT_NAME_MAP

SAVE_DIR = ROOT / "salt" / "models"
PHYS = PhysicochemicalExtractor()


def main():
    print("=" * 65)
    print("SALT HYBRID MODEL — EVALUATION")
    print("=" * 65)

    df_all = load_and_merge_datasets()
    df     = clean_salt_data(df_all)
    embedder = ESMEmbedder()

    le = joblib.load(SAVE_DIR / "salt_label_encoder.joblib")
    with open(SAVE_DIR / "salt_classes.json") as f:
        classes = json.load(f)
    n_classes = len(classes)

    y_type = le.transform(df["salt_display"].values)
    y_conc = df["salt_concentration_M"].values.astype(np.float32)
    y_log  = np.log1p(y_conc)

    X = build_feature_matrix(df, embedder)
    scaler = joblib.load(SAVE_DIR / "salt_scaler.joblib")
    X_sc   = scaler.transform(X)

    salt_onehot = np.zeros((len(y_type), n_classes), dtype=np.float32)
    salt_onehot[np.arange(len(y_type)), y_type] = 1
    X_conc    = np.hstack([X, salt_onehot])
    c_scaler  = joblib.load(SAVE_DIR / "salt_conc_scaler.joblib")
    X_conc_sc = c_scaler.transform(X_conc)

    # 15% test split
    idx = np.arange(len(df))
    _, te = train_test_split(idx, test_size=0.15, random_state=42)

    # ── Model A metrics ─────────────────────────────────────────────────────
    print("\n── MODEL A: Salt Type Classifier ──")
    print(f"{'Model':<12} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 36)

    dl_clf = PEGModel(n_classes=n_classes)
    dl_clf.load_state_dict(torch.load(SAVE_DIR / "salt_type_dl.pt", map_location="cpu"))
    dl_clf.eval()
    with torch.no_grad():
        dl_probs = torch.softmax(dl_clf(torch.tensor(X[te])), dim=-1).numpy()

    xgb_clf = joblib.load(SAVE_DIR / "salt_type_xgb.joblib")
    lgb_clf = joblib.load(SAVE_DIR / "salt_type_lgb.joblib")
    xgb_probs = xgb_clf.predict_proba(X_sc[te])
    lgb_probs = lgb_clf.predict_proba(X_sc[te])
    ens_probs = (dl_probs + xgb_probs + lgb_probs) / 3

    for name, probs in [("DL", dl_probs), ("XGBoost", xgb_probs),
                         ("LightGBM", lgb_probs), ("Ensemble", ens_probs)]:
        acc = accuracy_score(y_type[te], probs.argmax(1))
        f1  = f1_score(y_type[te], probs.argmax(1), average="macro", zero_division=0)
        tag = " ←" if name == "Ensemble" else ""
        print(f"{name:<12} {acc:>10.4f} {f1:>10.4f}{tag}")

    # ── Model B metrics ─────────────────────────────────────────────────────
    print("\n── MODEL B: Salt Concentration Regressor ──")
    print(f"{'Model':<12} {'MAE (M)':>10} {'RMSE (M)':>10} {'R²':>8}")
    print("-" * 44)

    dl_reg = SaltModel()
    dl_reg.load_state_dict(torch.load(SAVE_DIR / "salt_conc_dl.pt", map_location="cpu"))
    dl_reg.eval()
    with torch.no_grad():
        dl_log = dl_reg(torch.tensor(X[te])).numpy()

    xgb_reg  = joblib.load(SAVE_DIR / "salt_conc_xgb.joblib")
    lgb_reg  = joblib.load(SAVE_DIR / "salt_conc_lgb.joblib")
    meta_reg = joblib.load(SAVE_DIR / "salt_conc_meta.joblib")

    xgb_log = xgb_reg.predict(X_conc_sc[te])
    lgb_log = lgb_reg.predict(X_conc_sc[te])
    S = np.column_stack([dl_log, xgb_log, lgb_log, xgb_log])  # Ridge uses 4 cols; re-use xgb for ridge slot
    ens_log = meta_reg.predict(S)

    for name, preds_log in [("DL", dl_log), ("XGBoost", xgb_log),
                              ("LightGBM", lgb_log), ("Ensemble", ens_log)]:
        preds_m = np.expm1(preds_log)
        mae  = mean_absolute_error(y_conc[te], preds_m)
        rmse = np.sqrt(mean_squared_error(y_conc[te], preds_m))
        r2   = r2_score(y_conc[te], preds_m)
        tag  = " ←" if name == "Ensemble" else ""
        print(f"{name:<12} {mae:>10.4f} {rmse:>10.4f} {r2:>8.4f}{tag}")


if __name__ == "__main__":
    main()
