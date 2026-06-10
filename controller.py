"""
controller.py — Q-network controller: maps noisy channel output ẑ ∈ R^n → Q-values ∈ R^2.

Architecture:
    Linear(n, h1) → ReLU → Linear(h1, h2) → ReLU → Linear(h2, 2)

No final activation — raw Q-values for action {0=push left, 1=push right}.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class QController(nn.Module):
    """
    Q-value controller (receiver-side network).

    Parameters
    ----------
    n : int
        Input dimension (bottleneck / channel output size).
    hidden : List[int]
        Hidden layer widths.  Default: [64, 64].
    n_actions : int
        Number of discrete actions.  CartPole has 2.
    """

    def __init__(
        self,
        n: int = 8,
        hidden: List[int] = None,
        n_actions: int = 2,
    ) -> None:
        super().__init__()
        if hidden is None:
            hidden = [64, 64]

        layers: List[nn.Module] = []
        in_features = n
        for h in hidden:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        layers.append(nn.Linear(in_features, n_actions))
        # No activation — raw Q-values

        self.net = nn.Sequential(*layers)
        self.n = n
        self.n_actions = n_actions

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
            Noisy channel output.

        Returns
        -------
        q_values : Tensor, shape (batch, n_actions)
            Action-value estimates.
        """
        return self.net(z_hat)

    def extra_repr(self) -> str:
        return f"n={self.n}, n_actions={self.n_actions}"
