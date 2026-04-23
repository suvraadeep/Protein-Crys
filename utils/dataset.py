"""
PyTorch Dataset classes for each prediction target.
Uses 48-D BioFeatureExtractor (replaces 31-D PhysicochemicalExtractor).
Combined feature vector: [ESM-320 | bio-48] = 368-D.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional

from utils.esm_embedder import ESMEmbedder
from utils.bio_features import BioFeatureExtractor, COMBINED_DIM, BIO_EXTRACTOR

# Keep old name as alias so existing imports don't break
PhysicochemicalExtractor = BioFeatureExtractor


class _BaseProteinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, embedder: ESMEmbedder):
        self.df = df.reset_index(drop=True)
        self.embedder = embedder
        self.bio = BIO_EXTRACTOR

    def __len__(self):
        return len(self.df)

    def _get_features(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        emb  = self.embedder.embed_sequence(row["sequence"], row["pdb_id"])  # [320]
        bio  = self.bio.extract(row["sequence"])                             # [48]
        combined = np.concatenate([emb, bio], axis=0)                       # [368]
        return torch.tensor(combined, dtype=torch.float32)


class PHDataset(_BaseProteinDataset):
    def __init__(self, df: pd.DataFrame, embedder: ESMEmbedder):
        valid = df[df["pH"].notna() & df["pH"].between(2.0, 12.0)].copy()
        super().__init__(valid, embedder)

    def __getitem__(self, idx):
        x = self._get_features(idx)
        y = torch.tensor(float(self.df.iloc[idx]["pH"]), dtype=torch.float32)
        return x, y


class SaltDataset(_BaseProteinDataset):
    def __init__(self, df: pd.DataFrame, embedder: ESMEmbedder):
        valid = df[
            df["salt_concentration_M"].notna()
            & (df["salt_concentration_M"] > 0)
            & (df["salt_concentration_M"] <= 4.0)
        ].copy()
        super().__init__(valid, embedder)

    def __getitem__(self, idx):
        x = self._get_features(idx)
        raw = float(self.df.iloc[idx]["salt_concentration_M"])
        y = torch.tensor(np.log1p(raw), dtype=torch.float32)
        return x, y


class PEGDataset(_BaseProteinDataset):
    def __init__(self, df: pd.DataFrame, embedder: ESMEmbedder,
                 label_map: Optional[dict] = None):
        valid = df[df["peg_class"].notna()].copy()
        super().__init__(valid, embedder)
        if label_map is None:
            classes = sorted(valid["peg_class"].unique().tolist())
            self.label_map = {c: i for i, c in enumerate(classes)}
        else:
            self.label_map = label_map
        self.num_classes = len(self.label_map)

    def __getitem__(self, idx):
        x = self._get_features(idx)
        y = torch.tensor(self.label_map[self.df.iloc[idx]["peg_class"]], dtype=torch.long)
        return x, y


class TempDataset(_BaseProteinDataset):
    def __init__(self, df: pd.DataFrame, embedder: ESMEmbedder):
        valid = df[df["temp_k"].notna() & df["temp_k"].between(250.0, 320.0)].copy()
        super().__init__(valid, embedder)

    def __getitem__(self, idx):
        x = self._get_features(idx)
        y = torch.tensor(float(self.df.iloc[idx]["temp_k"]), dtype=torch.float32)
        return x, y


def split_dataset(dataset: Dataset, train=0.70, val=0.15, seed=42):
    from torch.utils.data import random_split
    n = len(dataset)
    n_train = int(n * train)
    n_val   = int(n * val)
    n_test  = n - n_train - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=gen)


def make_dataloader(dataset, batch_size=8, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)
