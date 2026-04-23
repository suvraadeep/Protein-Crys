"""
Train the PEG type classification model.

Usage:
    python training/train_peg.py [--epochs N] [--batch-size B]

Saves checkpoint and label mapping to checkpoints/peg_best.pt and
checkpoints/peg_label_map.json
"""

import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_parser import load_and_merge_datasets, get_peg_label_mapping
from utils.esm_embedder import ESMEmbedder
from utils.dataset import PEGDataset, split_dataset, make_dataloader
from models.peg_model import PEGModel
from training._train_utils import train_classification

CHECKPOINT = ROOT / "checkpoints" / "peg_best.pt"
LABEL_MAP_FILE = ROOT / "checkpoints" / "peg_label_map.json"


def compute_class_weights(labels: list[int], n_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(n_classes)
    for c in range(n_classes):
        weights[c] = total / (n_classes * max(counts.get(c, 1), 1))
    return weights


def main(epochs: int = 50, batch_size: int = 8):
    print("Loading dataset ...")
    df = load_and_merge_datasets()
    peg_rows = df[df["peg_class"].notna()]
    print(f"PEG dataset: {len(peg_rows):,} rows")

    if len(peg_rows) < 50:
        print("Warning: very few PEG samples — model may not converge well.")

    label_map = get_peg_label_mapping(df)
    n_classes = len(label_map)
    print(f"PEG classes ({n_classes}): {list(label_map.keys())}")

    # Save label map for inference
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with open(LABEL_MAP_FILE, "w") as f:
        json.dump(label_map, f, indent=2)

    embedder = ESMEmbedder()
    coverage = embedder.cache_coverage(peg_rows["pdb_id"].tolist())
    if coverage < 1.0:
        print(f"Cache coverage {coverage*100:.1f}% — building missing embeddings ...")
        embedder.cache_all(
            peg_rows["sequence"].tolist(),
            peg_rows["pdb_id"].tolist(),
            batch_size=4,
        )
    else:
        print("Embedding cache complete — skipping ESM forward pass.")

    dataset = PEGDataset(df, embedder, label_map=label_map)
    print(f"PEGDataset size: {len(dataset):,}")
    train_ds, val_ds, _ = split_dataset(dataset)
    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = make_dataloader(val_ds,   batch_size=batch_size, shuffle=False)

    # Class-balanced weights from training split
    train_labels = [int(dataset[i][1].item()) for i in train_ds.indices]
    class_weights = compute_class_weights(train_labels, n_classes)
    print(f"Class weights: {class_weights.tolist()}")

    model = PEGModel(n_classes=n_classes)
    print(f"PEGModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_classification(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=CHECKPOINT,
        class_weights=class_weights,
        max_epochs=epochs,
    )
    print("Done. Run  python training/evaluate.py  to see test metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
