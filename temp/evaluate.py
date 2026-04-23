"""
Temperature model evaluation.

Usage:
    python temp/evaluate.py
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

from utils.data_parser import load_and_merge_datasets
from utils.esm_embedder import ESMEmbedder
from utils.dataset import PhysicochemicalExtractor
from models.temp_model import TempModel
from temp.train import build_feature_matrix, bin_temp

SAVE_DIR = ROOT / "temp" / "models"
PHYS = PhysicochemicalExtractor()


def main():
    print("=" * 55)
    print("TEMPERATURE HYBRID MODEL — EVALUATION")
    print("=" * 55)

    df = load_and_merge_datasets()
    df = df[df["temp_k"].notna() & df["temp_k"].between(250.0, 320.0)].reset_index(drop=True)
    y_k    = df["temp_k"].values.astype(np.float32)
    y_bins = np.array([bin_temp(k) for k in y_k], dtype=np.int64)

    embedder = ESMEmbedder()
    X = build_feature_matrix(df, embedder)
    scaler = joblib.load(SAVE_DIR / "temp_scaler.joblib")
    X_sc   = scaler.transform(X)

    with open(SAVE_DIR / "temp_config.json") as f:
        cfg = json.load(f)

    _, te = train_test_split(np.arange(len(y_k)), test_size=0.15, random_state=42)

    dl_m = TempModel()
    dl_m.load_state_dict(torch.load(SAVE_DIR / "temp_dl.pt", map_location="cpu"))
    dl_m.eval()
    with torch.no_grad():
        dl_preds = dl_m(torch.tensor(X[te])).numpy()

    xgb_m = joblib.load(SAVE_DIR / "temp_xgb.joblib")
    lgb_m = joblib.load(SAVE_DIR / "temp_lgb.joblib")
    meta  = joblib.load(SAVE_DIR / "temp_meta.joblib")

    xgb_preds = xgb_m.predict(X_sc[te])
    lgb_preds = lgb_m.predict(X_sc[te])

    cb_preds = None
    try:
        import catboost as cb_lib
        cb_m = cb_lib.CatBoostRegressor(); cb_m.load_model(str(SAVE_DIR / "temp_cb.cbm"))
        cb_preds = cb_m.predict(X_sc[te])
    except Exception:
        pass

    if cb_preds is not None:
        S = np.column_stack([dl_preds, xgb_preds, lgb_preds, cb_preds])
    else:
        S = np.column_stack([dl_preds, xgb_preds, lgb_preds])

    ens_preds = meta.predict(S)

    print(f"\n{'Model':<14} {'MAE (K)':>9} {'MAE (°C)':>9} {'RMSE':>8} {'R²':>8}")
    print("-" * 54)
    models_preds = {"DL": dl_preds, "XGBoost": xgb_preds, "LightGBM": lgb_preds}
    if cb_preds is not None:
        models_preds["CatBoost"] = cb_preds
    models_preds["Ensemble"] = ens_preds

    for name, preds in models_preds.items():
        mae_k  = mean_absolute_error(y_k[te], preds)
        rmse_k = np.sqrt(mean_squared_error(y_k[te], preds))
        r2     = r2_score(y_k[te], preds)
        tag    = " ←" if name == "Ensemble" else ""
        print(f"{name:<14} {mae_k:>9.3f} {mae_k:>9.3f} {rmse_k:>8.3f} {r2:>8.4f}{tag}")

    # Bin classification from regressor
    ens_bins = np.array([bin_temp(k) for k in ens_preds])
    f1 = f1_score(y_bins[te], ens_bins, average="weighted")
    print(f"\nTemp bin classification (from ensemble regression): weighted-F1={f1:.3f}")


if __name__ == "__main__":
    main()
