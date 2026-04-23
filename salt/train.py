"""
=============================================================================
Salt Concentration + Salt Type — Hybrid ESM-DL + XGBoost + LightGBM Pipeline
=============================================================================

Builds TWO models:

Model A — Salt Type Classifier (improved from original 34% → stacking ensemble)
  Base learners: DL (SaltModel head adapted for classification), XGBoost, LightGBM
  Ensemble: soft-vote on class probabilities

Model B — Salt Concentration Regressor (improved from R²=0.45 → stacking)
  Base learners: DL (SaltModel), XGBoost, LightGBM, Ridge (baseline)
  Stacking: Ridge meta-learner on OOF predictions
  Input: ESM+physicochemical features + PREDICTED salt type one-hot (same design
         as original "Model B uses salt type as input feature")

Key improvements over salt-conc/03_train_and_evaluate.py:
  - Embedding cache → no repeated ESM forward passes
  - DL model in the ensemble (original had DL separate, not stacked)
  - LightGBM added
  - Stacking instead of pick-one
  - Unified unified dataset with pH/PEG/temp
  - IQR outlier removal per salt type preserved

Usage:
    python salt/train.py [--epochs N] [--batch-size B] [--folds K]
=============================================================================
"""

import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb

from utils.data_parser import load_and_merge_datasets
from utils.esm_embedder import ESMEmbedder
from utils.bio_features import BioFeatureExtractor
from models.salt_model import SaltModel
from models.peg_model import PEGModel   # reuse architecture for salt type clf
from training._train_utils import get_warmup_cosine_scheduler, EarlyStopping

SAVE_DIR = ROOT / "salt" / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PHYS = BioFeatureExtractor()

# ── Canonical salt name map (from original code) ────────────────────────────
SALT_NAME_MAP = {
    "SODIUM CHLORIDE": "Sodium Chloride", "NACL": "Sodium Chloride",
    "AMMONIUM SULFATE": "Ammonium Sulfate", "AMMONIUM SULPHATE": "Ammonium Sulfate",
    "(NH4)2SO4": "Ammonium Sulfate",
    "POTASSIUM CHLORIDE": "Potassium Chloride", "KCL": "Potassium Chloride",
    "MAGNESIUM CHLORIDE": "Magnesium Chloride", "MGCL2": "Magnesium Chloride",
    "MGCL": "Magnesium Chloride",
    "LITHIUM SULFATE": "Lithium Sulfate", "LITHIUM SULPHATE": "Lithium Sulfate",
    "LI2SO4": "Lithium Sulfate",
    "CALCIUM CHLORIDE": "Calcium Chloride", "CACL2": "Calcium Chloride",
    "SODIUM ACETATE": "Sodium Acetate", "NAAC": "Sodium Acetate",
    "NA ACETATE": "Sodium Acetate", "NAOAC": "Sodium Acetate",
    "SODIUM CITRATE": "Sodium Citrate", "SODIUM MALONATE": "Sodium Malonate",
    "SODIUM FORMATE": "Sodium Formate",
    "AMMONIUM CHLORIDE": "Ammonium Chloride", "NH4CL": "Ammonium Chloride",
    "SODIUM SULFATE": "Sodium Sulfate", "SODIUM SULPHATE": "Sodium Sulfate",
    "NA2SO4": "Sodium Sulfate",
    "ZINC ACETATE": "Zinc Acetate",
    "POTASSIUM PHOSPHATE": "Potassium Phosphate",
    "AMMONIUM ACETATE": "Ammonium Acetate",
}


def clean_salt_data(df: pd.DataFrame, min_count: int = 15) -> pd.DataFrame:
    """Replicate original data cleaning: IQR outlier removal per salt type."""
    df = df[df["salt_concentration_M"].notna()].copy()
    df = df[(df["salt_concentration_M"] > 0) & (df["salt_concentration_M"] <= 4.0)]
    df = df[df["salt_type"].notna()].copy()

    df["salt_display"] = df["salt_type"].map(SALT_NAME_MAP).fillna(df["salt_type"])

    # Keep only salt types with enough samples
    counts = df["salt_display"].value_counts()
    valid = counts[counts >= min_count].index.tolist()
    df = df[df["salt_display"].isin(valid)].copy()

    # IQR outlier removal per salt type (original logic)
    parts = []
    for salt in valid:
        sub = df[df["salt_display"] == salt].copy()
        Q1, Q3 = sub["salt_concentration_M"].quantile(0.25), sub["salt_concentration_M"].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = max(0, Q1 - 1.5*IQR), Q3 + 1.5*IQR
        parts.append(sub[(sub["salt_concentration_M"] >= lo) & (sub["salt_concentration_M"] <= hi)])

    df = pd.concat(parts, ignore_index=True)
    df["log_salt"] = np.log1p(df["salt_concentration_M"])
    return df


def build_feature_matrix(df: pd.DataFrame, embedder: ESMEmbedder) -> np.ndarray:
    rows = []
    for _, row in df.iterrows():
        emb  = embedder.embed_sequence(row["sequence"], row["pdb_id"])
        phys = PHYS.extract(row["sequence"])
        rows.append(np.concatenate([emb, phys]))
    return np.stack(rows).astype(np.float32)


# ── DL trainer helpers ───────────────────────────────────────────────────────

def _train_dl_regression_fold(X_tr, y_tr, X_val, y_val,
                               epochs, batch_size, device="cpu"):
    model = SaltModel().to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                        batch_size=batch_size, shuffle=True, num_workers=0)
    opt  = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    sched = get_warmup_cosine_scheduler(opt, int(0.1*len(loader)*epochs//4), len(loader)*epochs//4)
    crit  = nn.HuberLoss(delta=1.0)
    es    = EarlyStopping(patience=10)

    Xv = torch.tensor(X_val, device=device); yv = torch.tensor(y_val, device=device)
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


def _train_dl_classifier_fold(X_tr, y_tr, X_val, y_val,
                               n_classes, epochs, batch_size, device="cpu"):
    """Reuse PEGModel architecture for salt-type classification."""
    model = PEGModel(n_classes=n_classes).to(device)
    sw = compute_sample_weight("balanced", y_tr)
    sw_t = torch.tensor(sw, dtype=torch.float32, device=device)

    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long))
    loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    opt  = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()
    es   = EarlyStopping(patience=10)

    Xv = torch.tensor(X_val, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)
    opt.zero_grad()

    for epoch in range(1, epochs+1):
        model.train()
        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb) / 4
            loss.backward()
            if (i+1) % 4 == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        if es.step(vl, model): break

    es.restore_best(model); model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Xv), dim=-1).cpu().numpy()
    return model, probs


# ── Main ─────────────────────────────────────────────────────────────────────

def main(epochs=50, batch_size=8, n_folds=5):
    print("=" * 65)
    print("SALT CONCENTRATION + SALT TYPE — HYBRID TRAINING PIPELINE")
    print("=" * 65)

    # ── Data ──
    df_all = load_and_merge_datasets()
    df = clean_salt_data(df_all)
    print(f"Dataset: {len(df):,} rows | {df['salt_display'].nunique()} salt types")
    for s, c in df["salt_display"].value_counts().head(8).items():
        print(f"  {s:<28} {c:>5}")

    y_conc = df["salt_concentration_M"].values.astype(np.float32)
    y_log  = df["log_salt"].values.astype(np.float32)

    le = LabelEncoder()
    y_type = le.fit_transform(df["salt_display"].values)
    n_classes = len(le.classes_)
    joblib.dump(le, SAVE_DIR / "salt_label_encoder.joblib")
    with open(SAVE_DIR / "salt_classes.json", "w") as f:
        json.dump(le.classes_.tolist(), f)

    # ── Embedding cache ──
    embedder = ESMEmbedder()
    coverage = embedder.cache_coverage(df["pdb_id"].tolist())
    if coverage < 1.0:
        embedder.cache_all(df["sequence"].tolist(), df["pdb_id"].tolist(), batch_size=4)
    else:
        print("Embedding cache complete.")

    print("Building feature matrix …")
    X = build_feature_matrix(df, embedder)   # [N, 351]
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    joblib.dump(scaler, SAVE_DIR / "salt_scaler.joblib")

    # ── Salt type one-hot (for concentration regressor, like original design) ──
    salt_onehot = np.zeros((len(y_type), n_classes), dtype=np.float32)
    salt_onehot[np.arange(len(y_type)), y_type] = 1
    X_conc = np.hstack([X, salt_onehot])  # [N, 351+n_classes]
    X_conc_sc = StandardScaler().fit_transform(X_conc)

    # ════════════════════════════════════════════════════════════════════════
    # MODEL A: Salt Type Classifier  (K-fold stacking)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*50}")
    print("MODEL A — Salt Type Classifier")
    print(f"{'─'*50}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_dl_probs  = np.zeros((len(y_type), n_classes))
    oof_xgb_probs = np.zeros((len(y_type), n_classes))
    oof_lgb_probs = np.zeros((len(y_type), n_classes))
    clf_dl_models = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sc, y_type), 1):
        print(f"\n  Fold {fold}/{n_folds}")
        X_tr, X_val = X[tr_idx], X[val_idx]
        X_tr_sc, X_val_sc = X_sc[tr_idx], X_sc[val_idx]
        y_tr, y_val = y_type[tr_idx], y_type[val_idx]

        # DL classifier
        dl_clf, dl_probs = _train_dl_classifier_fold(
            X_tr, y_tr, X_val, y_val, n_classes, epochs, batch_size)
        oof_dl_probs[val_idx] = dl_probs
        clf_dl_models.append(dl_clf)

        # XGBoost classifier
        xgb_clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=3.0, min_child_weight=5,
            objective="multi:softprob", num_class=n_classes,
            random_state=42, n_jobs=-1,
        )
        sw = compute_sample_weight("balanced", y_tr)
        xgb_clf.fit(X_tr_sc, y_tr, sample_weight=sw,
                    eval_set=[(X_val_sc, y_val)], verbose=False)
        oof_xgb_probs[val_idx] = xgb_clf.predict_proba(X_val_sc)

        # LightGBM classifier
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=5,
            num_leaves=31, reg_alpha=0.5, reg_lambda=3.0,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        lgb_clf.fit(X_tr_sc, y_tr,
                    eval_set=[(X_val_sc, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                                lgb.log_evaluation(period=-1)])
        oof_lgb_probs[val_idx] = lgb_clf.predict_proba(X_val_sc)

        # OOF accuracy
        for name, probs in [("DL", dl_probs), ("XGB", oof_xgb_probs[val_idx]),
                             ("LGB", oof_lgb_probs[val_idx])]:
            acc = accuracy_score(y_val, probs.argmax(1))
            print(f"    {name}: Acc={acc:.3f}")

    # Soft-vote ensemble OOF accuracy
    oof_ensemble = (oof_dl_probs + oof_xgb_probs + oof_lgb_probs) / 3
    oof_acc = accuracy_score(y_type, oof_ensemble.argmax(1))
    oof_f1  = f1_score(y_type, oof_ensemble.argmax(1), average="macro", zero_division=0)
    print(f"\n  OOF Ensemble → Acc={oof_acc:.3f} | Macro-F1={oof_f1:.3f}")

    # Retrain full classifiers
    print("  Retraining final classifiers on full data …")
    final_xgb_clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=3.0,
        min_child_weight=5, objective="multi:softprob", num_class=n_classes,
        random_state=42, n_jobs=-1,
    )
    final_xgb_clf.fit(X_sc, y_type, sample_weight=compute_sample_weight("balanced", y_type))
    joblib.dump(final_xgb_clf, SAVE_DIR / "salt_type_xgb.joblib")

    final_lgb_clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.04, max_depth=5, num_leaves=31,
        reg_alpha=0.5, reg_lambda=3.0, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    final_lgb_clf.fit(X_sc, y_type, callbacks=[lgb.log_evaluation(period=-1)])
    joblib.dump(final_lgb_clf, SAVE_DIR / "salt_type_lgb.joblib")

    dl_clf_final = clf_dl_models[-1]
    torch.save(dl_clf_final.state_dict(), SAVE_DIR / "salt_type_dl.pt")

    # ════════════════════════════════════════════════════════════════════════
    # MODEL B: Salt Concentration Regressor  (K-fold stacking)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*50}")
    print("MODEL B — Salt Concentration Regressor")
    print(f"{'─'*50}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_dl_reg  = np.zeros(len(y_log))
    oof_xgb_reg = np.zeros(len(y_log))
    oof_lgb_reg = np.zeros(len(y_log))
    oof_ridge   = np.zeros(len(y_log))
    reg_dl_models = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_conc_sc), 1):
        print(f"\n  Fold {fold}/{n_folds}")
        X_tr, X_val = X_conc[tr_idx], X_conc[val_idx]
        X_tr_sc, X_val_sc = X_conc_sc[tr_idx], X_conc_sc[val_idx]
        y_tr, y_val = y_log[tr_idx], y_log[val_idx]
        y_raw_val   = y_conc[val_idx]

        # DL (uses base 351-D, not with onehot since SaltModel input=351)
        dl_reg, dl_preds_log = _train_dl_regression_fold(
            X[tr_idx], y_tr, X[val_idx], y_val, epochs, batch_size)
        oof_dl_reg[val_idx] = dl_preds_log
        reg_dl_models.append(dl_reg)

        # Ridge baseline (as in original)
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr_sc, y_tr)
        oof_ridge[val_idx] = ridge.predict(X_val_sc)

        # XGBoost
        xgb_reg = xgb.XGBRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=1.0, reg_lambda=5.0, min_child_weight=10,
            gamma=0.5, random_state=42, n_jobs=-1,
        )
        xgb_reg.fit(X_tr_sc, y_tr, eval_set=[(X_val_sc, y_val)], verbose=False)
        oof_xgb_reg[val_idx] = xgb_reg.predict(X_val_sc)

        # LightGBM
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4, num_leaves=31,
            reg_alpha=1.0, reg_lambda=5.0, subsample=0.7, colsample_bytree=0.6,
            random_state=42, n_jobs=-1,
        )
        lgb_reg.fit(X_tr_sc, y_tr,
                    eval_set=[(X_val_sc, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                                lgb.log_evaluation(period=-1)])
        oof_lgb_reg[val_idx] = lgb_reg.predict(X_val_sc)

        for name, preds_log in [("DL", dl_preds_log), ("Ridge", oof_ridge[val_idx]),
                                  ("XGB", oof_xgb_reg[val_idx]), ("LGB", oof_lgb_reg[val_idx])]:
            preds_m = np.expm1(preds_log)
            mae = mean_absolute_error(y_raw_val, preds_m)
            r2  = r2_score(y_raw_val, preds_m)
            print(f"    {name:6s} | MAE={mae:.4f}M | R²={r2:.4f}")

    # Meta-learner
    S_oof = np.column_stack([oof_dl_reg, oof_xgb_reg, oof_lgb_reg, oof_ridge])
    meta  = Ridge(alpha=1.0)
    meta.fit(S_oof, y_log)
    meta_log = meta.predict(S_oof)
    meta_m   = np.expm1(meta_log)
    mae_m  = mean_absolute_error(y_conc, meta_m)
    rmse_m = np.sqrt(mean_squared_error(y_conc, meta_m))
    r2_m   = r2_score(y_conc, meta_m)
    print(f"\n  OOF Ensemble → MAE={mae_m:.4f}M | RMSE={rmse_m:.4f}M | R²={r2_m:.4f}")
    print(f"  Meta weights: DL={meta.coef_[0]:.3f} XGB={meta.coef_[1]:.3f} "
          f"LGB={meta.coef_[2]:.3f} Ridge={meta.coef_[3]:.3f}")

    # Retrain full regressors
    print("  Retraining final regressors on full data …")
    final_xgb_reg = xgb.XGBRegressor(
        n_estimators=600, max_depth=3, learning_rate=0.03, subsample=0.7,
        colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=5.0, min_child_weight=10,
        gamma=0.5, random_state=42, n_jobs=-1,
    )
    final_xgb_reg.fit(X_conc_sc, y_log)
    joblib.dump(final_xgb_reg, SAVE_DIR / "salt_conc_xgb.joblib")

    final_lgb_reg = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=4, num_leaves=31,
        reg_alpha=1.0, reg_lambda=5.0, subsample=0.7, colsample_bytree=0.6,
        random_state=42, n_jobs=-1,
    )
    final_lgb_reg.fit(X_conc_sc, y_log, callbacks=[lgb.log_evaluation(period=-1)])
    joblib.dump(final_lgb_reg, SAVE_DIR / "salt_conc_lgb.joblib")

    torch.save(reg_dl_models[-1].state_dict(), SAVE_DIR / "salt_conc_dl.pt")
    joblib.dump(meta, SAVE_DIR / "salt_conc_meta.joblib")

    # Save conc scaler (for X+onehot)
    conc_scaler = StandardScaler().fit(X_conc)
    joblib.dump(conc_scaler, SAVE_DIR / "salt_conc_scaler.joblib")

    with open(SAVE_DIR / "salt_config.json", "w") as f:
        json.dump({
            "n_salt_classes": n_classes,
            "classes": le.classes_.tolist(),
            "clf_oof_acc": float(oof_acc), "clf_oof_f1": float(oof_f1),
            "reg_oof_mae": float(mae_m), "reg_oof_r2": float(r2_m),
        }, f, indent=2)

    print(f"\nAll salt models saved to {SAVE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--folds",      type=int, default=5)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, n_folds=args.folds)
