"""
Train the pH regression model.

Usage:
    python training/train_ph.py [--epochs N] [--batch-size B]

Phase 1: Build/verify embedding cache (skipped if already complete).
Phase 2: Train PHModel with early stopping, save to checkpoints/ph_best.pt
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_parser import load_and_merge_datasets
from utils.esm_embedder import ESMEmbedder
from utils.dataset import PHDataset, split_dataset, make_dataloader
from models.ph_model import PHModel
from training._train_utils import train_regression

CHECKPOINT = ROOT / "checkpoints" / "ph_best.pt"


def main(epochs: int = 50, batch_size: int = 8):
    # ── Load data ──────────────────────────────────────────────────────
    print("Loading dataset ...")
    df = load_and_merge_datasets()
    ph_rows = df[df["pH"].notna()]
    print(f"pH dataset: {len(ph_rows):,} rows")

    # ── Phase 1: cache embeddings ──────────────────────────────────────
    embedder = ESMEmbedder()
    coverage = embedder.cache_coverage(ph_rows["pdb_id"].tolist())
    if coverage < 1.0:
        print(f"Cache coverage {coverage*100:.1f}% — building missing embeddings ...")
        embedder.cache_all(
            ph_rows["sequence"].tolist(),
            ph_rows["pdb_id"].tolist(),
            batch_size=4,
        )
    else:
        print("Embedding cache complete — skipping ESM forward pass.")

    # ── Phase 2: build Datasets / DataLoaders ─────────────────────────
    dataset = PHDataset(df, embedder)
    print(f"PHDataset size: {len(dataset):,}")
    train_ds, val_ds, _ = split_dataset(dataset)
    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = make_dataloader(val_ds,   batch_size=batch_size, shuffle=False)

    # ── Phase 3: train ─────────────────────────────────────────────────
    model = PHModel()
    print(f"PHModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_regression(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=CHECKPOINT,
        max_epochs=epochs,
        batch_size=batch_size if False else None,   # unused param placeholder
    )
    print("Done. Run  python training/evaluate.py  to see test metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
