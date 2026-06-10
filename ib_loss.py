"""
ib_loss.py — Nonlinear Information Bottleneck upper bound.

Implements Eq. 11 from Kolchinsky et al. (2019), as cited in Kam et al. (MILCOM 2024):

    I_theta(ẑ; x)  ≤  -(1/N) Σᵢ log[ (1/N) Σⱼ exp(-||f_θ(xᵢ) - f_θ(xⱼ)||² / (2σ²)) ]

where the outer sum is over samples i=1..N, the inner sum over j=1..N,
and σ² = sigma2_total (channel + trainable IB noise).

Implementation uses the log-sum-exp trick for numerical stability, and clamping
to prevent log(0) from NaN-corrupting training.
"""
from __future__ import annotations

import torch


_MIN_SIGMA2 = 1e-8   # Guard against division by zero when sigma2_total ≈ 0
_EPS_DIST   = 1e-10  # Clamp floor for pairwise distances before log


def nonlinear_ib_bound(z: torch.Tensor, sigma2_total: float) -> torch.Tensor:
    """
    Compute the nonlinear IB upper bound on I(ẑ; x) for a batch of latents.

    Parameters
    ----------
    z : Tensor, shape (N, n)
        Encoder outputs f_θ(xᵢ) for a batch of N samples.
        These are the PRE-CHANNEL latents (before noise is added).
    sigma2_total : float
        Total noise variance (channel + trainable).  Must be ≥ 0.
        If 0 (or very small), a safe epsilon is used so the bound stays finite.

    Returns
    -------
    ib_loss : Tensor, scalar
        Upper bound on mutual information I(ẑ; x), averaged over the batch.
        A lower value means better compression.

    Notes
    -----
    Efficient pairwise distance via:
        ||a - b||² = ||a||² + ||b||² - 2 aᵀb   (avoids explicit N×N×n tensor)
    Log-sum-exp stabilization: subtract row-max before exp, add back after.
    """
    sigma2 = max(float(sigma2_total), _MIN_SIGMA2)

    N = z.shape[0]

    # --- Pairwise squared Euclidean distances: shape (N, N) ---
    # ||z_i - z_j||² = ||z_i||² + ||z_j||² - 2 z_i·z_j
    z_sq = (z * z).sum(dim=1, keepdim=True)              # (N, 1)
    cross = torch.mm(z, z.t())                           # (N, N)
    dist2 = z_sq + z_sq.t() - 2.0 * cross               # (N, N)

    # Numerical guard: distances can become slightly negative due to FP error
    dist2 = torch.clamp(dist2, min=0.0)

    # --- Exponent: -||zᵢ - zⱼ||² / (2σ²) ---
    log_kernel = -dist2 / (2.0 * sigma2)                 # (N, N)

    # --- Log-sum-exp over j for each i (row-wise) ---
    # log[ (1/N) Σⱼ exp(log_kernel_ij) ]
    #   = log(Σⱼ exp(log_kernel_ij)) - log(N)
    # Stabilized:  max_j(log_kernel_ij) subtracted before exp, re-added after
    row_max = log_kernel.max(dim=1, keepdim=True).values          # (N, 1)
    stable_exp = torch.exp(log_kernel - row_max)                  # (N, N)
    log_sum = row_max.squeeze(1) + torch.log(                     # (N,)
        torch.clamp(stable_exp.sum(dim=1), min=_EPS_DIST)
    ) - torch.log(torch.tensor(float(N), device=z.device))

    # --- IB bound = -(1/N) Σᵢ log_sum_i ---
    ib_bound = -log_sum.mean()

    return ib_bound
