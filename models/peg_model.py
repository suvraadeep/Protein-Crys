"""
PEG type classification model with excluded-volume injection.

Architecture:
  ESMBackbone(368) → trunk[128]
    → cat(trunk, predicted_Rg, MW_norm) → head(130→64→n_classes)

The injected features reflect the polymer physics of PEG crystallization:
  predicted_Rg  — Flory N^0.6 radius; larger proteins need higher-MW PEG
  MW_norm       — molecular weight; correlates with PEG size by excluded-volume theory
"""

import torch
import torch.nn as nn
from models.esm_backbone import ESMBackbone
from utils.bio_features import BIO_IDX_RG, BIO_IDX_MW


class PEGModel(nn.Module):
    """
    Classifies PEG type from 368-D combined protein features.

    Input:  [batch, 368]
    Output: [batch, n_classes]  (raw logits; softmax for probabilities)
    """

    def __init__(self, n_classes: int, input_dim: int = 368):
        super().__init__()
        self.backbone = ESMBackbone(input_dim=input_dim)
        # +2 for Rg / MW injection
        self.head = nn.Sequential(
            nn.Linear(128 + 2, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )
        self._idxs = [BIO_IDX_RG, BIO_IDX_MW]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk = self.backbone(x)                        # [batch, 128]
        bio   = x[:, self._idxs]                        # [batch, 2]
        return self.head(torch.cat([trunk, bio], dim=1))  # [batch, n_classes]
