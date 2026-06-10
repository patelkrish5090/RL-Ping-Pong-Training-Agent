"""
reconstruction_head.py — Auxiliary feature reconstruction head for Variant A.

Maps noisy channel output ẑ ∈ R^n → reconstructed state features x̂ ∈ R^4.

    g_phi: R^n → [32] → R^4   (no final activation — raw regression)

Used ONLY during training to compute the LLM-guided auxiliary loss L_llm.
Discarded at inference time (evaluation mode).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ReconstructionHead(nn.Module):
    """
    Lightweight state-feature reconstruction head.

    Parameters
    ----------
    n : int
        Input dimension (channel output / bottleneck size).
    hidden : List[int]
        Hidden layer widths.  Default: [32].
    output_dim : int
        Number of state features to reconstruct.  4 for CartPole.
    """

    def __init__(
        self,
        n: int = 8,
        hidden: List[int] = None,
        output_dim: int = 4,
    ) -> None:
        super().__init__()
        if hidden is None:
            hidden = [32]

        layers: List[nn.Module] = []
        in_features = n
        for h in hidden:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        layers.append(nn.Linear(in_features, output_dim))
        # No final activation — linear regression of raw state values

        self.net = nn.Sequential(*layers)
        self.n = n
        self.output_dim = output_dim

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_hat : Tensor, shape (batch, n)
            Noisy channel output (shared with Q-controller forward pass).

        Returns
        -------
        x_hat : Tensor, shape (batch, 4)
            Reconstructed state features.
        """
        return self.net(z_hat)

    def extra_repr(self) -> str:
        return f"n={self.n}, output_dim={self.output_dim}"
