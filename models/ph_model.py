"""
pH regression model with biologically-guided pI injection.

Architecture:
  ESMBackbone(368) → trunk[128] → cat(trunk, pI_norm) → head(129→64→1)

The pI feature is the single strongest predictor of crystallization pH
(proteins tend to crystallize near their isoelectric point ± 1–2 units).
"""

import torch
import torch.nn as nn
from models.esm_backbone import ESMBackbone
from utils.bio_features import BIO_IDX_PI


class PHModel(nn.Module):
    """
    Predicts pH (0–14) from 368-D combined protein features.

    Input:  [batch, 368]
    Output: [batch]  (pH values in [0, 14])
    """

    def __init__(self, input_dim: int = 368):
        super().__init__()
        self.backbone = ESMBackbone(input_dim=input_dim)
        # +1 for pI injection
        self.head = nn.Sequential(
            nn.Linear(128 + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self._pi_idx = BIO_IDX_PI

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk = self.backbone(x)                          # [batch, 128]
        pi    = x[:, self._pi_idx].unsqueeze(1)           # [batch, 1]
        out   = self.head(torch.cat([trunk, pi], dim=1))  # [batch, 1]
        return torch.sigmoid(out).squeeze(-1) * 14.0      # [batch]  → [0, 14]
