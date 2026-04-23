"""
Crystallization temperature regression model with thermostability injection.

Architecture:
  ESMBackbone(368) → trunk[128]
    → cat(trunk, thermostability_idx, aliphatic_idx) → head(130→64→1)

The injected features encode thermostability from two independent indices:
  thermostability_idx  — IVYWREL index (fraction of thermostability-linked AAs)
  aliphatic_idx        — [Ala + 2.9×Val + 3.9×(Ile+Leu)] / len; high → thermostable
"""

import torch
import torch.nn as nn
from models.esm_backbone import ESMBackbone
from utils.bio_features import BIO_IDX_THERMO, BIO_IDX_ALIPHATIC


class TempModel(nn.Module):
    """
    Predicts crystallization temperature (K) from 368-D combined features.

    Input:  [batch, 368]
    Output: [batch]  (temperature in Kelvin)
    """

    def __init__(self, input_dim: int = 368):
        super().__init__()
        self.backbone = ESMBackbone(input_dim=input_dim)
        # +2 for thermostability injection
        self.head = nn.Sequential(
            nn.Linear(128 + 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self._idxs = [BIO_IDX_THERMO, BIO_IDX_ALIPHATIC]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk = self.backbone(x)                                   # [batch, 128]
        bio   = x[:, self._idxs]                                   # [batch, 2]
        return self.head(torch.cat([trunk, bio], dim=1)).squeeze(-1)  # [batch]
