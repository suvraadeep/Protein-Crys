"""
Unified evaluation for all four crystallization prediction models.

Usage:
    python training/evaluate.py          # evaluate all four
    python training/evaluate.py --task ph
    python training/evaluate.py --task salt
    python training/evaluate.py --task peg
    python training/evaluate.py --task temp
"""

import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_parser import load_and_merge_datasets, get_peg_label_mapping
from utils.esm_embedder import ESMEmbedder
from utils.dataset import (
    PHDataset, SaltDataset, PEGDataset, TempDataset,
    split_dataset, make_dataloader,
)
from models.ph_model import PHModel
from models.salt_model import SaltModel
from models.peg_model import PEGModel
from models.temp_model import TempModel

CHECKPOINTS = {
    "ph":   ROOT / "checkpoints" / "ph_best.pt",
    "salt": ROOT / "checkpoints" / "salt_best.pt",
    "peg":  ROOT / "checkpoints" / "peg_best.pt",
    "temp": ROOT / "checkpoints" / "temp_best.pt",
}


def _regression_metrics(y_true, y_pred, label: str):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{label}]")
    print(f"    MAE  = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    print(f"    R²   = {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_ph(df, embedder):
    ckpt = CHECKPOINTS["ph"]
    if not ckpt.exists():
        print("  [pH] checkpoint not found — run train_ph.py first.")
        return
    model = PHModel()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    dataset = PHDataset(df, embedder)
    _, _, test_ds = split_dataset(dataset)
    loader = make_dataloader(test_ds, batch_size=32, shuffle=False)

    preds, truths = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x).tolist())
            truths.extend(y.tolist())

    _regression_metrics(truths, preds, "pH")


def evaluate_salt(df, embedder):
    ckpt = CHECKPOINTS["salt"]
    if not ckpt.exists():
        print("  [Salt] checkpoint not found — run train_salt.py first.")
        return
    model = SaltModel()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    dataset = SaltDataset(df, embedder)
    _, _, test_ds = split_dataset(dataset)
    loader = make_dataloader(test_ds, batch_size=32, shuffle=False)

    preds_log, truths_log, preds_m, truths_m = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x).tolist()
            preds_log.extend(p)
            truths_log.extend(y.tolist())
            preds_m.extend(np.expm1(p).tolist())
            truths_m.extend(np.expm1(y.numpy()).tolist())

    print("\n  [Salt Conc] — log-space metrics:")
    _regression_metrics(truths_log, preds_log, "log1p(salt_M)")
    print("  [Salt Conc] — original-space metrics (Molar):")
    _regression_metrics(truths_m, preds_m, "salt_M")


def evaluate_peg(df, embedder):
    ckpt = CHECKPOINTS["peg"]
    label_map_file = ROOT / "checkpoints" / "peg_label_map.json"
    if not ckpt.exists():
        print("  [PEG] checkpoint not found — run train_peg.py first.")
        return
    if not label_map_file.exists():
        print("  [PEG] label map not found.")
        return

    with open(label_map_file) as f:
        label_map = json.load(f)
    inv_map = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)

    model = PEGModel(n_classes=n_classes)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    dataset = PEGDataset(df, embedder, label_map=label_map)
    _, _, test_ds = split_dataset(dataset)
    loader = make_dataloader(test_ds, batch_size=32, shuffle=False)

    preds, truths = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            preds.extend(logits.argmax(1).tolist())
            truths.extend(y.tolist())

    acc = accuracy_score(truths, preds)
    f1  = f1_score(truths, preds, average="macro", zero_division=0)
    cm  = confusion_matrix(truths, preds)
    print(f"\n  [PEG Type]")
    print(f"    Accuracy  = {acc:.4f}")
    print(f"    Macro-F1  = {f1:.4f}")
    print(f"    Classes   : {[inv_map[i] for i in range(n_classes)]}")
    print(f"    Confusion matrix:\n{cm}")
    return {"accuracy": acc, "macro_f1": f1}


def evaluate_temp(df, embedder):
    ckpt = CHECKPOINTS["temp"]
    if not ckpt.exists():
        print("  [Temp] checkpoint not found — run train_temp.py first.")
        return
    model = TempModel()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    dataset = TempDataset(df, embedder)
    _, _, test_ds = split_dataset(dataset)
    loader = make_dataloader(test_ds, batch_size=32, shuffle=False)

    preds, truths = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x).tolist())
            truths.extend(y.tolist())

    _regression_metrics(truths, preds, "Temperature (K)")
    # Also report in Celsius
    preds_c  = [p - 273.15 for p in preds]
    truths_c = [t - 273.15 for t in truths]
    _regression_metrics(truths_c, preds_c, "Temperature (°C)")


def main(task: str = "all"):
    print("Loading dataset ...")
    df = load_and_merge_datasets()
    embedder = ESMEmbedder()

    print("\n" + "=" * 55)
    print("   CRYSTALLIZATION CONDITION PREDICTION — EVALUATION")
    print("=" * 55)

    tasks = ["ph", "salt", "peg", "temp"] if task == "all" else [task]
    runners = {
        "ph":   evaluate_ph,
        "salt": evaluate_salt,
        "peg":  evaluate_peg,
        "temp": evaluate_temp,
    }
    for t in tasks:
        runners[t](df, embedder)

    print("\n" + "=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["all", "ph", "salt", "peg", "temp"], default="all")
    args = parser.parse_args()
    main(args.task)
