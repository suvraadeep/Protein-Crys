"""
=============================================================================
pH Prediction — Hybrid ESM-DL + XGBoost + LightGBM Stacking Pipeline
=============================================================================

Architecture (3 base learners → Ridge meta-learner):
  Base 1: ESMBackbone + PHModel head (PyTorch)
  Base 2: XGBoost regressor on 368-D (ESM-320 + bio-48)
  Base 3: LightGBM regressor on same features

  Meta: Ridge regression trained on out-of-fold (OOF) predictions
        from the 3 base learners.

Embedding cache:  embeddings_cache/{pdb_id}.npy
                  Built once, reused across all target pipelines.

Usage:
    python ph/train.py [--epochs N] [--batch-size B] [--folds K]
=============================================================================
"""

import sys
import argparse
import numpy as np
import pandas as pd
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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from utils.data_parser import load_and_merge_datasets
from utils.esm_embedder import ESMEmbedder
from utils.bio_features import BioFeatureExtractor
from models.ph_model import PHModel
from training._train_utils import get_warmup_cosine_scheduler, EarlyStopping

SAVE_DIR = ROOT / "ph" / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PHYS = BioFeatureExtractor()


# ── Feature matrix builder ──────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, embedder: ESMEmbedder) -> np.ndarray:
    """Returns [N, 368] combined ESM + biological features."""
    rows = []
    for _, row in df.iterrows():
        emb  = embedder.embed_sequence(row["sequence"], row["pdb_id"])  # [320]
        bio  = PHYS.extract(row["sequence"])                            # [48]
        rows.append(np.concatenate([emb, bio]))
    return np.stack(rows, axis=0).astype(np.float32)                    # [N, 368]


# ── DL training on a single train/val split ─────────────────────────────────

def _train_dl_fold(X_tr, y_tr, X_val, y_val,
                   epochs=50, batch_size=8, lr=2e-4,
                   weight_decay=1e-2, accum=4, patience=10,
                   device="cpu"):
    model = PHModel().to(device)

    tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                          torch.tensor(y_tr, dtype=torch.float32))
    loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(loader) * epochs // accum
    scheduler = get_warmup_cosine_scheduler(optimizer, int(0.1*total_steps), total_steps)
    criterion = nn.HuberLoss(delta=1.0)
    stopper   = EarlyStopping(patience=patience)

    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    optimizer.zero_grad()
    for epoch in range(1, epochs+1):
        model.train()
        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb) / accum
            loss.backward()
            if (i+1) % accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if stopper.step(val_loss, model):
            break

    stopper.restore_best(model)
    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).cpu().numpy()
    return model, preds


# ── Main training ────────────────────────────────────────────────────────────

def main(epochs=50, batch_size=8, n_folds=5):
    print("=" * 60)
    print("pH PREDICTION — HYBRID TRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Load & filter data ──
    df = load_and_merge_datasets()
    df = df[df["pH"].notna() & df["pH"].between(2.0, 12.0)].copy()
    df = df.reset_index(drop=True)
    print(f"Dataset: {len(df):,} rows with pH labels")

    y = df["pH"].values.astype(np.float32)

    # ── 2. Build / verify embedding cache ──
    embedder = ESMEmbedder()
    coverage = embedder.cache_coverage(df["pdb_id"].tolist())
    if coverage < 1.0:
        print(f"Cache coverage {coverage*100:.1f}% — building embeddings …")
        embedder.cache_all(df["sequence"].tolist(), df["pdb_id"].tolist(), batch_size=4)
    else:
        print("Embedding cache complete.")

    print("Building feature matrix …")
    X = build_feature_matrix(df, embedder)   # [N, 368]
    print(f"Feature matrix: {X.shape}")

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # ── 3. K-Fold stacking: collect OOF predictions ──
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_dl   = np.zeros(len(y))
    oof_xgb  = np.zeros(len(y))
    oof_lgb  = np.zeros(len(y))

    fold_dl_models = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_sc), 1):
        print(f"\n── Fold {fold}/{n_folds} ──────────────────────────")
        X_tr, X_val = X[tr_idx], X[val_idx]
        X_tr_sc, X_val_sc = X_sc[tr_idx], X_sc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Base 1: DL
        print(f"  [DL]  training PHModel …")
        dl_model, dl_preds = _train_dl_fold(
            X_tr, y_tr, X_val, y_val,
            epochs=epochs, batch_size=batch_size
        )
        oof_dl[val_idx] = dl_preds
        fold_dl_models.append(dl_model)

        # Base 2: XGBoost
        print(f"  [XGB] training XGBoost …")
        xgb_model = xgb.XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=2.0, min_child_weight=5,
            random_state=42, n_jobs=-1,
        )
        xgb_model.fit(X_tr_sc, y_tr, eval_set=[(X_val_sc, y_val)],
                      verbose=False)
        oof_xgb[val_idx] = xgb_model.predict(X_val_sc)

        # Base 3: LightGBM
        print(f"  [LGB] training LightGBM …")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=5,
            num_leaves=31, reg_alpha=0.5, reg_lambda=2.0,
            subsample=0.8, colsample_bytree=0.7,
            random_state=42, n_jobs=-1,
        )
        lgb_model.fit(X_tr_sc, y_tr,
                      eval_set=[(X_val_sc, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=-1)])
        oof_lgb[val_idx] = lgb_model.predict(X_val_sc)

        # Per-fold OOF metrics
        for name, preds in [("DL", dl_preds), ("XGB", oof_xgb[val_idx]),
                             ("LGB", oof_lgb[val_idx])]:
            mae = mean_absolute_error(y_val, preds)
            r2  = r2_score(y_val, preds)
            print(f"    {name:4s} | MAE={mae:.4f} | R²={r2:.4f}")

    # ── 4. Train meta-learner on OOF stack ──
    print("\n── Training Ridge meta-learner ─────────────────────")
    S_oof = np.column_stack([oof_dl, oof_xgb, oof_lgb])    # [N, 3]
    meta  = Ridge(alpha=1.0)
    meta.fit(S_oof, y)

    meta_preds = meta.predict(S_oof)
    mae  = mean_absolute_error(y, meta_preds)
    rmse = np.sqrt(mean_squared_error(y, meta_preds))
    r2   = r2_score(y, meta_preds)
    print(f"  OOF meta predictions → MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
    print(f"  Meta weights: DL={meta.coef_[0]:.3f} "
          f"XGB={meta.coef_[1]:.3f} LGB={meta.coef_[2]:.3f}")

    # ── 5. Retrain final models on FULL data ──
    print("\n── Retraining final models on full data ────────────")

    # Best DL model = the one from the last fold (or retrain; use last fold's)
    final_dl = fold_dl_models[-1]
    torch.save(final_dl.state_dict(), SAVE_DIR / "ph_dl.pt")

    final_xgb = xgb.XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0, min_child_weight=5,
        random_state=42, n_jobs=-1,
    )
    final_xgb.fit(X_sc, y)
    joblib.dump(final_xgb, SAVE_DIR / "ph_xgb.joblib")

    final_lgb = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=5,
        num_leaves=31, reg_alpha=0.5, reg_lambda=2.0,
        subsample=0.8, colsample_bytree=0.7,
        random_state=42, n_jobs=-1,
    )
    final_lgb.fit(X_sc, y, callbacks=[lgb.log_evaluation(period=-1)])
    joblib.dump(final_lgb, SAVE_DIR / "ph_lgb.joblib")

    joblib.dump(scaler, SAVE_DIR / "ph_scaler.joblib")
    joblib.dump(meta,   SAVE_DIR / "ph_meta.joblib")

    # Save config for inference
    with open(SAVE_DIR / "ph_config.json", "w") as f:
        json.dump({"oof_mae": float(mae), "oof_r2": float(r2),
                   "meta_coef": meta.coef_.tolist(),
                   "meta_intercept": float(meta.intercept_)}, f, indent=2)

    print(f"\nSaved all pH models to {SAVE_DIR}")
    print(f"Final OOF ensemble → MAE={mae:.4f}  R²={r2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--folds",      type=int, default=5)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, n_folds=args.folds)
