"""
Salt concentration regression model with Hofmeister-guided injection.

Architecture:
  ESMBackbone(368) → trunk[128]
    → cat(trunk, kosmotropic, hofmeister, surface_exposed) → head(131→64→1)

The three injected features encode the Hofmeister series directly:
  kosmotropic_score  — fraction of salting-out amino acids
  hofmeister_rank    — weighted Hofmeister position of surface residues
  surface_exposed    — fraction of solvent-exposed charged/polar AAs
"""

import torch
import torch.nn as nn
from models.esm_backbone import ESMBackbone
from utils.bio_features import BIO_IDX_KOSMOTROPIC, BIO_IDX_HOFMEISTER, BIO_IDX_SURFACE


class SaltModel(nn.Module):
    """
    Predicts log1p(salt_conc_M) from 368-D combined protein features.
    Apply expm1() at inference to recover Molar value.

    Input:  [batch, 368]
    Output: [batch]  (log-scale concentrations)
    """

    def __init__(self, input_dim: int = 368):
        super().__init__()
        self.backbone = ESMBackbone(input_dim=input_dim)
        # +3 for Hofmeister injection
        self.head = nn.Sequential(
            nn.Linear(128 + 3, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self._idxs = [BIO_IDX_KOSMOTROPIC, BIO_IDX_HOFMEISTER, BIO_IDX_SURFACE]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk = self.backbone(x)                                   # [batch, 128]
        bio   = x[:, self._idxs]                                   # [batch, 3]
        return self.head(torch.cat([trunk, bio], dim=1)).squeeze(-1)  # [batch]
