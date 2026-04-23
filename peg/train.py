"""
=============================================================================
PEG Type Classification — Hybrid ESM-DL + XGBoost + LightGBM Pipeline
=============================================================================

Soft-vote ensemble of three base classifiers:
  Base 1: DL (PEGModel — ESMBackbone + classification head)
  Base 2: XGBoost  (multi:softprob)
  Base 3: LightGBM

All use class-balanced weighting to handle PEG class imbalance.

Usage:
    python peg/train.py [--epochs N] [--batch-size B] [--folds K]
=============================================================================
"""

import sys
import argparse
import numpy as np
import joblib
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb

from utils.data_parser import load_and_merge_datasets, get_peg_label_mapping
from utils.esm_embedder import ESMEmbedder
from utils.bio_features import BioFeatureExtractor
from models.peg_model import PEGModel
from training._train_utils import get_warmup_cosine_scheduler, EarlyStopping

SAVE_DIR = ROOT / "peg" / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PHYS = BioFeatureExtractor()


def build_feature_matrix(df, embedder):
    rows = []
    for _, row in df.iterrows():
        emb  = embedder.embed_sequence(row["sequence"], row["pdb_id"])
        phys = PHYS.extract(row["sequence"])
        rows.append(np.concatenate([emb, phys]))
    return np.stack(rows).astype(np.float32)


def _train_dl_fold(X_tr, y_tr, X_val, y_val,
                   n_classes, epochs, batch_size, device="cpu"):
    model = PEGModel(n_classes=n_classes).to(device)

    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    class_weights = torch.tensor(
        (counts.sum() / (n_classes * np.maximum(counts, 1))), dtype=torch.float32, device=device
    )

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long)),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    opt   = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    sched = get_warmup_cosine_scheduler(opt, int(0.1*len(loader)*epochs//4), len(loader)*epochs//4)
    crit  = nn.CrossEntropyLoss(weight=class_weights)
    es    = EarlyStopping(patience=10)

    Xv = torch.tensor(X_val, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)
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
        probs = torch.softmax(model(Xv), dim=-1).cpu().numpy()
    return model, probs


def main(epochs=50, batch_size=8, n_folds=5):
    print("=" * 60)
    print("PEG TYPE CLASSIFICATION — HYBRID TRAINING PIPELINE")
    print("=" * 60)

    df = load_and_merge_datasets()
    df = df[df["peg_class"].notna()].copy().reset_index(drop=True)
    print(f"Dataset: {len(df):,} rows with PEG labels")

    label_map = get_peg_label_mapping(df)
    inv_map   = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)
    print(f"PEG classes ({n_classes}): {list(label_map.keys())}")

    for cls, cnt in df["peg_class"].value_counts().items():
        print(f"  {cls:<15}: {cnt:>5}")

    y = np.array([label_map[c] for c in df["peg_class"]], dtype=np.int64)

    # Save label map
    with open(SAVE_DIR / "peg_label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Embedding cache
    embedder = ESMEmbedder()
    coverage = embedder.cache_coverage(df["pdb_id"].tolist())
    if coverage < 1.0:
        embedder.cache_all(df["sequence"].tolist(), df["pdb_id"].tolist(), batch_size=4)
    else:
        print("Embedding cache complete.")

    print("Building feature matrix …")
    X = build_feature_matrix(df, embedder)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    joblib.dump(scaler, SAVE_DIR / "peg_scaler.joblib")

    # K-Fold stacking
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_dl_probs  = np.zeros((len(y), n_classes))
    oof_xgb_probs = np.zeros((len(y), n_classes))
    oof_lgb_probs = np.zeros((len(y), n_classes))
    dl_models = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sc, y), 1):
        print(f"\n── Fold {fold}/{n_folds} ──────────────────────")
        X_tr, X_val = X[tr_idx], X[val_idx]
        X_tr_sc, X_val_sc = X_sc[tr_idx], X_sc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        sw = compute_sample_weight("balanced", y_tr)

        # DL
        dl_m, dl_probs = _train_dl_fold(X_tr, y_tr, X_val, y_val,
                                         n_classes, epochs, batch_size)
        oof_dl_probs[val_idx] = dl_probs
        dl_models.append(dl_m)

        # XGBoost
        xgb_m = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=3.0, min_child_weight=3,
            objective="multi:softprob", num_class=n_classes,
            random_state=42, n_jobs=-1,
        )
        xgb_m.fit(X_tr_sc, y_tr, sample_weight=sw,
                   eval_set=[(X_val_sc, y_val)], verbose=False)
        oof_xgb_probs[val_idx] = xgb_m.predict_proba(X_val_sc)

        # LightGBM
        lgb_m = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=5, num_leaves=31,
            reg_alpha=0.5, reg_lambda=3.0, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        lgb_m.fit(X_tr_sc, y_tr,
                   eval_set=[(X_val_sc, y_val)],
                   callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(period=-1)])
        oof_lgb_probs[val_idx] = lgb_m.predict_proba(X_val_sc)

        for name, probs in [("DL", dl_probs), ("XGB", oof_xgb_probs[val_idx]),
                             ("LGB", oof_lgb_probs[val_idx])]:
            acc = accuracy_score(y_val, probs.argmax(1))
            print(f"  {name}: Acc={acc:.3f}")

    # Soft-vote OOF
    oof_ens = (oof_dl_probs + oof_xgb_probs + oof_lgb_probs) / 3
    oof_acc = accuracy_score(y, oof_ens.argmax(1))
    oof_f1  = f1_score(y, oof_ens.argmax(1), average="macro", zero_division=0)
    print(f"\nOOF Ensemble → Acc={oof_acc:.3f} | Macro-F1={oof_f1:.3f}")

    # Retrain final models on full data
    print("Retraining final models …")
    final_xgb = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.04, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=3.0, min_child_weight=3,
        objective="multi:softprob", num_class=n_classes, random_state=42, n_jobs=-1,
    )
    final_xgb.fit(X_sc, y, sample_weight=compute_sample_weight("balanced", y))
    joblib.dump(final_xgb, SAVE_DIR / "peg_xgb.joblib")

    final_lgb = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.04, max_depth=5, num_leaves=31,
        reg_alpha=0.5, reg_lambda=3.0, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    final_lgb.fit(X_sc, y, callbacks=[lgb.log_evaluation(period=-1)])
    joblib.dump(final_lgb, SAVE_DIR / "peg_lgb.joblib")

    torch.save(dl_models[-1].state_dict(), SAVE_DIR / "peg_dl.pt")

    with open(SAVE_DIR / "peg_config.json", "w") as f:
        json.dump({"n_classes": n_classes, "label_map": label_map,
                   "inv_map": inv_map, "oof_acc": float(oof_acc),
                   "oof_f1": float(oof_f1)}, f, indent=2)

    print(f"\nAll PEG models saved to {SAVE_DIR}")
    print(f"OOF Ensemble → Acc={oof_acc:.3f}  Macro-F1={oof_f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--folds",      type=int, default=5)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, n_folds=args.folds)
