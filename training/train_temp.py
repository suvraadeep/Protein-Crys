"""
Train the temperature regression model.

Usage:
    python training/train_temp.py [--epochs N] [--batch-size B]

Saves checkpoint to checkpoints/temp_best.pt
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_parser import load_and_merge_datasets
from utils.esm_embedder import ESMEmbedder
from utils.dataset import TempDataset, split_dataset, make_dataloader
from models.temp_model import TempModel
from training._train_utils import train_regression

CHECKPOINT = ROOT / "checkpoints" / "temp_best.pt"


def main(epochs: int = 50, batch_size: int = 8):
    print("Loading dataset ...")
    df = load_and_merge_datasets()
    temp_rows = df[df["temp_k"].notna() & df["temp_k"].between(250.0, 320.0)]
    print(f"Temp dataset: {len(temp_rows):,} rows")

    embedder = ESMEmbedder()
    coverage = embedder.cache_coverage(temp_rows["pdb_id"].tolist())
    if coverage < 1.0:
        print(f"Cache coverage {coverage*100:.1f}% — building missing embeddings ...")
        embedder.cache_all(
            temp_rows["sequence"].tolist(),
            temp_rows["pdb_id"].tolist(),
            batch_size=4,
        )
    else:
        print("Embedding cache complete — skipping ESM forward pass.")

    dataset = TempDataset(df, embedder)
    print(f"TempDataset size: {len(dataset):,}")
    train_ds, val_ds, _ = split_dataset(dataset)
    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = make_dataloader(val_ds,   batch_size=batch_size, shuffle=False)

    model = TempModel()
    print(f"TempModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_regression(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=CHECKPOINT,
        max_epochs=epochs,
    )
    print("Done. Run  python training/evaluate.py  to see test metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
