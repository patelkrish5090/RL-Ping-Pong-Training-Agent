"""
channel.py — Non-trainable AWGN channel layer.

The channel adds Gaussian noise via reparameterization so gradients flow back
to the encoder (f_theta) without any learned parameters in the channel itself.

    z_hat = z + epsilon,  epsilon ~ N(0, sigma2 * I_n)

Power constraint is enforced by the encoder's Tanh final activation (|z_i| <= 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AWGNChannel(nn.Module):
    """
    Additive White Gaussian Noise channel.

    Parameters
    ----------
    sigma2 : float
        Noise variance.  Set to 0.0 for a noiseless channel.

    Notes
    -----
    - NO nn.Parameter instances — the channel has no trainable weights.
    - Differentiable via reparameterization: gradients of downstream losses
      flow back through (z + eps) to z (and therefore to the encoder).
    - At sigma2 == 0 the module returns z unchanged to avoid unnecessary
      allocations.
    """

    def __init__(self, sigma2: float = 0.0) -> None:
        super().__init__()
        self.sigma2 = float(sigma2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : Tensor, shape (..., n)
            Power-constrained encoder output (values in [-1, 1] due to Tanh).

        Returns
        -------
        z_hat : Tensor, same shape as z
            Noisy channel output.
        """
        if self.sigma2 <= 0.0 or not self.training:
            # At eval time or zero noise: no stochastic noise injection.
            # This ensures deterministic evaluation, exactly as in the paper.
            if self.sigma2 <= 0.0:
                return z
            # sigma2 > 0 but eval mode — still add noise so generalization
            # evaluation sees realistic channel conditions.
            std = self.sigma2 ** 0.5
            eps = torch.randn_like(z) * std
            return z + eps

        std = self.sigma2 ** 0.5
        eps = torch.randn_like(z) * std   # reparameterized: no grad through eps itself
        return z + eps

    def extra_repr(self) -> str:
        return f"sigma2={self.sigma2}"


class AWGNChannelEval(nn.Module):
    """
    Identical to AWGNChannel but ALWAYS adds noise regardless of train/eval mode.
    Used during generalization evaluation where we want realistic channel conditions.
    """

    def __init__(self, sigma2: float = 0.0) -> None:
        super().__init__()
        self.sigma2 = float(sigma2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.sigma2 <= 0.0:
            return z
        std = self.sigma2 ** 0.5
        eps = torch.randn_like(z) * std
        return z + eps
