"""
encoder.py — JSCC Encoder: maps CartPole state x ∈ R^4 → channel symbols z ∈ R^n.

Architecture:
    Linear(4, h1) → ReLU → Linear(h1, h2) → ReLU → Linear(h2, n) → Tanh

The Tanh final activation enforces the per-symbol power constraint |z_i| ≤ 1,
consistent with the paper (Kam et al., MILCOM 2024).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class JSCCEncoder(nn.Module):
    """
    Joint Source-Channel Coding encoder.

    Parameters
    ----------
    input_dim : int
        Dimension of the state vector (4 for CartPole).
    n : int
        Number of channel symbols (bottleneck dimension).
    hidden : List[int]
        Hidden layer widths.  Default: [64, 64].
    """

    def __init__(
        self,
        input_dim: int = 4,
        n: int = 8,
        hidden: List[int] = None,
    ) -> None:
        super().__init__()
        if hidden is None:
            hidden = [64, 64]

        layers: List[nn.Module] = []
        in_features = input_dim
        for h in hidden:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        layers.append(nn.Linear(in_features, n))
        layers.append(nn.Tanh())          # Power constraint: |z_i| ≤ 1

        self.net = nn.Sequential(*layers)
        self.n = n
        self.input_dim = input_dim

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, 4)
            CartPole state vector.

        Returns
        -------
        z : Tensor, shape (batch, n)
            Channel symbols before noise injection.  Values in [-1, 1].
        """
        return self.net(x)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, n={self.n}"
