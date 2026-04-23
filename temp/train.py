"""
=============================================================================
Temperature Prediction — Hybrid ESM-DL + XGBoost + LightGBM + CatBoost
=============================================================================

Extends the original temp/Train_and_Evaluate.py (XGB+LGB+CB simple average)
with:
  1. ESM-based DL model (TempModel) as an additional base learner
  2. Ridge stacking meta-learner instead of simple average
  3. Shared embedding cache (no repeated ESM forward passes)
  4. K-fold OOF to ensure unbiased stacking

Original approach:  XGBoost + LightGBM + CatBoost  (simple mean)
New approach:       DL + XGBoost + LightGBM + CatBoost  → Ridge meta

Also mirrors the original dual output:
  - Regression: temperature in Kelvin (continuous)
  - Classification: 3 temperature bins (≤10°C, 11-20°C, >20°C)

Usage:
    python temp/train.py [--epochs N] [--batch-size B] [--folds K]
=============================================================================
"""

import sys
import argparse
import numpy as np
import joblib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    _CATBOOST = True
except ImportError:
    _CATBOOST = False
    print("CatBoost not installed — using XGB+LGB+DL only.")

from utils.data_parser import load_and_merge_datasets
from utils.esm_embedder import ESMEmbedder
from utils.bio_features import BioFeatureExtractor
from models.temp_model import TempModel
from training._train_utils import get_warmup_cosine_scheduler, EarlyStopping

SAVE_DIR = ROOT / "temp" / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PHYS = BioFeatureExtractor()


def bin_temp(k: float) -> int:
    """Original binning from temp/Train_and_Evaluate.py."""
    c = k - 273.15
    return 0 if c <= 10 else (1 if c <= 20 else 2)


def build_feature_matrix(df, embedder):
    rows = []
    for _, row in df.iterrows():
        emb  = embedder.embed_sequence(row["sequence"], row["pdb_id"])
        phys = PHYS.extract(row["sequence"])
        rows.append(np.concatenate([emb, phys]))
    return np.stack(rows).astype(np.float32)


def _train_dl_fold(X_tr, y_tr, X_val, y_val, epochs, batch_size, device="cpu"):
    model = TempModel().to(device)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    opt   = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    sched = get_warmup_cosine_scheduler(opt, int(0.1*len(loader)*epochs//4), len(loader)*epochs//4)
    crit  = nn.HuberLoss(delta=5.0)   # wider delta for temperature (K) scale
    es    = EarlyStopping(patience=10)

    Xv = torch.tensor(X_val, device=device)
    yv = torch.tensor(y_val, device=device)
    opt.zero_grad()

    for epoch in range(1, epochs+1):
        model.train()
        for i, (xb, yb) in enumerate(loader):
            loss = crit(model(xb.to(device)), yb.to(device)) / 4
            loss.backward()
            if (i+1) % 4 == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        if es.step(vl, model): break

    es.restore_best(model); model.eval()
    with torch.no_grad():
        return model, model(Xv).cpu().numpy()


def main(epochs=50, batch_size=8, n_folds=5):
    print("=" * 65)
    print("TEMPERATURE PREDICTION — HYBRID TRAINING PIPELINE")
    print("=" * 65)

    # ── Data ──
    df = load_and_merge_datasets()
    df = df[df["temp_k"].notna() & df["temp_k"].between(250.0, 320.0)].copy()
    df = df.reset_index(drop=True)
    print(f"Dataset: {len(df):,} rows  (temp range: "
          f"{df['temp_k'].min():.1f}–{df['temp_k'].max():.1f} K)")

    y_k    = df["temp_k"].values.astype(np.float32)
    y_c    = (y_k - 273.15).astype(np.float32)
    y_bins = np.array([bin_temp(k) for k in y_k], dtype=np.int64)
    print(f"Temp class distribution: { {i: int((y_bins==i).sum()) for i in range(3)} }")

    # ── Embedding cache ──
    embedder = ESMEmbedder()
    coverage = embedder.cache_coverage(df["pdb_id"].tolist())
    if coverage < 1.0:
        print(f"Cache {coverage*100:.1f}% — building embeddings …")
        embedder.cache_all(df["sequence"].tolist(), df["pdb_id"].tolist(), batch_size=4)
    else:
        print("Embedding cache complete.")

    print("Building feature matrix …")
    X = build_feature_matrix(df, embedder)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    joblib.dump(scaler, SAVE_DIR / "temp_scaler.joblib")

    # ── K-Fold OOF stacking ──
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    n_base = 4 if _CATBOOST else 3

    oof_dl  = np.zeros(len(y_k))
    oof_xgb = np.zeros(len(y_k))
    oof_lgb = np.zeros(len(y_k))
    oof_cb  = np.zeros(len(y_k)) if _CATBOOST else None
    dl_models = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_sc), 1):
        print(f"\n── Fold {fold}/{n_folds} ──────────────────────────────")
        X_tr, X_val = X[tr_idx], X[val_idx]
        X_tr_sc, X_val_sc = X_sc[tr_idx], X_sc[val_idx]
        y_tr, y_val = y_k[tr_idx], y_k[val_idx]

        # DL
        print(f"  [DL]  …")
        dl_m, dl_preds = _train_dl_fold(X_tr, y_tr, X_val, y_val, epochs, batch_size)
        oof_dl[val_idx] = dl_preds
        dl_models.append(dl_m)

        # XGBoost
        print(f"  [XGB] …")
        xgb_m = xgb.XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0, min_child_weight=5,
            random_state=42, n_jobs=-1,
        )
        xgb_m.fit(X_tr_sc, y_tr, eval_set=[(X_val_sc, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb_m.predict(X_val_sc)

        # LightGBM
        print(f"  [LGB] …")
        lgb_m = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=5, num_leaves=31,
            reg_alpha=0.3, reg_lambda=2.0, subsample=0.8, colsample_bytree=0.7,
            random_state=42, n_jobs=-1,
        )
        lgb_m.fit(X_tr_sc, y_tr,
                  eval_set=[(X_val_sc, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(period=-1)])
        oof_lgb[val_idx] = lgb_m.predict(X_val_sc)

        # CatBoost
        if _CATBOOST:
            print(f"  [CB]  …")
            cb_m = cb.CatBoostRegressor(iterations=500, learning_rate=0.03,
                                         depth=6, random_seed=42, verbose=0)
            cb_m.fit(X_tr_sc, y_tr, eval_set=(X_val_sc, y_val), use_best_model=True)
            oof_cb[val_idx] = cb_m.predict(X_val_sc)

        # Per-fold metrics (Kelvin)
        simple_avg = (oof_dl[val_idx] + oof_xgb[val_idx] + oof_lgb[val_idx]) / (
            3 if not _CATBOOST else 4)
        if _CATBOOST:
            simple_avg = (oof_dl[val_idx] + oof_xgb[val_idx] + oof_lgb[val_idx] + oof_cb[val_idx]) / 4
        for name, preds in [("DL", oof_dl[val_idx]), ("XGB", oof_xgb[val_idx]),
                             ("LGB", oof_lgb[val_idx]), ("Simple Avg", simple_avg)]:
            mae = mean_absolute_error(y_val, preds)
            r2  = r2_score(y_val, preds)
            print(f"    {name:<12} MAE={mae:.3f}K  R²={r2:.4f}")

    # ── Meta-learner (Ridge stacking) ──
    print("\n── Training Ridge meta-learner ─────────────────────")
    cols = [oof_dl, oof_xgb, oof_lgb]
    if _CATBOOST:
        cols.append(oof_cb)
    S_oof = np.column_stack(cols)
    meta  = Ridge(alpha=1.0)
    meta.fit(S_oof, y_k)
    meta_preds = meta.predict(S_oof)
    mae_m  = mean_absolute_error(y_k, meta_preds)
    rmse_m = np.sqrt(mean_squared_error(y_k, meta_preds))
    r2_m   = r2_score(y_k, meta_preds)

    # Also compare with original simple-average approach
    if _CATBOOST:
        orig_preds = (oof_dl + oof_xgb + oof_lgb + oof_cb) / 4
    else:
        orig_preds = (oof_dl + oof_xgb + oof_lgb) / 3
    orig_mae = mean_absolute_error(y_k, orig_preds)
    orig_r2  = r2_score(y_k, orig_preds)

    print(f"  Original-style average → MAE={orig_mae:.3f}K  R²={orig_r2:.4f}")
    print(f"  Ridge stacking         → MAE={mae_m:.3f}K   R²={r2_m:.4f} ← BEST")
    w_str = " ".join(f"{n}={c:.3f}" for n, c in
                     zip(["DL","XGB","LGB","CB"][:n_base], meta.coef_))
    print(f"  Meta weights: {w_str}")

    # ── Temperature classification OOF (from meta regression predictions) ──
    print("\n── Temperature bin classification from regressor ───")
    bins_pred = np.array([bin_temp(k) for k in meta_preds])
    clf_f1    = f1_score(y_bins, bins_pred, average="weighted")
    print(f"  Weighted F1 (bins from regression): {clf_f1:.3f}")

    # ── Retrain final models on full data ──
    print("\nRetraining final models on full data …")
    final_xgb = xgb.XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=2.0, min_child_weight=5,
        random_state=42, n_jobs=-1,
    )
    final_xgb.fit(X_sc, y_k)
    joblib.dump(final_xgb, SAVE_DIR / "temp_xgb.joblib")

    final_lgb = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=5, num_leaves=31,
        reg_alpha=0.3, reg_lambda=2.0, subsample=0.8, colsample_bytree=0.7,
        random_state=42, n_jobs=-1,
    )
    final_lgb.fit(X_sc, y_k, callbacks=[lgb.log_evaluation(period=-1)])
    joblib.dump(final_lgb, SAVE_DIR / "temp_lgb.joblib")

    if _CATBOOST:
        final_cb = cb.CatBoostRegressor(iterations=600, learning_rate=0.03,
                                         depth=6, random_seed=42, verbose=0)
        final_cb.fit(X_sc, y_k)
        final_cb.save_model(str(SAVE_DIR / "temp_cb.cbm"))

    torch.save(dl_models[-1].state_dict(), SAVE_DIR / "temp_dl.pt")
    joblib.dump(meta, SAVE_DIR / "temp_meta.joblib")

    with open(SAVE_DIR / "temp_config.json", "w") as f:
        json.dump({
            "oof_mae_k": float(mae_m), "oof_r2": float(r2_m),
            "oof_f1_bins": float(clf_f1),
            "catboost_available": _CATBOOST,
            "meta_coef": meta.coef_.tolist(),
        }, f, indent=2)

    print(f"\nAll temperature models saved to {SAVE_DIR}")
    print(f"Final stack → MAE={mae_m:.3f}K  R²={r2_m:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--folds",      type=int, default=5)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, n_folds=args.folds)
