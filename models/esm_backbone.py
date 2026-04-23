"""
Shared ESM-2 + Physicochemical MLP trunk.

Takes pre-computed (cached) 351-D input:
  [320-D ESM mean-pool embedding | 31-D physicochemical features]

Outputs a 128-D representation that is consumed by each task head.
"""

import torch
import torch.nn as nn


class ESMBackbone(nn.Module):
    """
    Shared MLP trunk that processes the concatenated ESM + bio feature vector
    and produces a 128-D task-agnostic representation.

    Input:  [batch, 368]   (320 ESM + 48 biological features)
    Output: [batch, 128]
    """

    def __init__(
        self,
        input_dim: int = 368,   # 320 (ESM) + 48 (biological features)
        hidden1: int = 256,
        hidden2: int = 128,
        dropout1: float = 0.3,
        dropout2: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # [batch, 128]
