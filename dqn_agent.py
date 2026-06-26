"""
dqn_agent.py — End-to-end DQN agent with Variant A (LLM-guided auxiliary loss).

Combined training objective:
    L_total = L_DQN + β · L_IB + λ · L_llm

Architecture:
    Online:  Encoder → Channel → Controller  (Q-values)
             Encoder → Channel → ReconHead   (state reconstruction, training only)
    Target:  Encoder_tgt → [no channel noise for target Q]

Design notes:
- A single set of channel noise eps is sampled and shared across both the
  Q-value forward pass and the reconstruction forward pass (consistent channel realization).
- The reconstruction head is only active when variant_a.enabled=True.
- Target network uses the encoder + controller only; no channel noise is added
  for the target Q-values computation (common DQN practice for stability).
- LLM weights w are numpy arrays treated as constants (no autograd through them).
"""
from __future__ import annotations

import copy
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ExperimentConfig
from encoder import JSCCEncoder
from controller import QController
from reconstruction_head import ReconstructionHead
from ib_loss import nonlinear_ib_bound


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular experience replay buffer."""

    def __init__(self, capacity: int) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self._buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs),      dtype=torch.float32),
            torch.tensor(actions,            dtype=torch.long),
            torch.tensor(rewards,            dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(dones,              dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    End-to-end DQN with optional IB and Variant A (LLM-guided reconstruction) losses.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.cfg = config
        self.tcfg = config.training
        self.ccfg = config.channel
        self.ibcfg = config.ib
        self.vcfg = config.variant_a

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Online network components ---
        self.encoder = JSCCEncoder(
            input_dim=4,
            n=self.ccfg.n,
            hidden=config.network.encoder_hidden,
        ).to(self.device)

        self.controller = QController(
            n=self.ccfg.n,
            hidden=config.network.controller_hidden,
            n_actions=2,
        ).to(self.device)

        self.recon_head: Optional[ReconstructionHead] = None
        if self.vcfg.enabled:
            self.recon_head = ReconstructionHead(
                n=self.ccfg.n,
                hidden=config.network.recon_head_hidden,
                output_dim=4,
            ).to(self.device)

        # --- Target network (encoder + controller only, no ReconHead) ---
        self.target_encoder = copy.deepcopy(self.encoder).to(self.device)
        self.target_controller = copy.deepcopy(self.controller).to(self.device)
        self._freeze_target()

        # --- Optimizer over all ONLINE parameters ---
        online_params = (
            list(self.encoder.parameters())
            + list(self.controller.parameters())
            + (list(self.recon_head.parameters()) if self.recon_head else [])
        )
        self.optimizer = torch.optim.Adam(online_params, lr=self.tcfg.lr)

        # --- Replay buffer ---
        self.replay = ReplayBuffer(self.tcfg.replay_buffer_size)

        # --- Epsilon-greedy state ---
        self.epsilon = self.tcfg.eps_start

        # --- Current LLM weights (set externally by training loop) ---
        self._llm_weights: np.ndarray = np.array(
            [0.25, 0.25, 0.25, 0.25], dtype=np.float32
        )

    # ------------------------------------------------------------------
    # External setters
    # ------------------------------------------------------------------

    def set_llm_weights(self, weights: np.ndarray) -> None:
        """Update the LLM importance weights used in L_llm."""
        self._llm_weights = weights.astype(np.float32)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        obs : np.ndarray, shape (4,)
        greedy : bool
            If True, ignore epsilon (pure exploitation).

        Returns
        -------
        action : int  (0 or 1)
        """
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, 1)

        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            z = self.encoder(x)
            # At inference inside training loop: add channel noise deterministically
            if self.ccfg.sigma2_channel > 0:
                eps = torch.randn_like(z) * (self.ccfg.sigma2_channel ** 0.5)
                z_hat = z + eps
            else:
                z_hat = z
            q_values = self.controller(z_hat)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.push(obs, action, reward, next_obs, done)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def update(self) -> Optional[Dict[str, float]]:
        """
        Sample a minibatch and perform one gradient step.

        Returns
        -------
        losses : dict with keys 'loss_dqn', 'loss_ib', 'loss_llm', 'loss_total'
                 or None if replay is too small.
        """
        if len(self.replay) < self.tcfg.min_replay_size:
            return None

        obs_b, act_b, rew_b, next_obs_b, done_b = self.replay.sample(
            self.tcfg.batch_size
        )
        obs_b       = obs_b.to(self.device)
        act_b       = act_b.to(self.device)
        rew_b       = rew_b.to(self.device)
        next_obs_b  = next_obs_b.to(self.device)
        done_b      = done_b.to(self.device)

        # ---- Forward pass: online encoder ----
        z_online = self.encoder(obs_b)                   # (B, n)

        # Sample ONE shared noise draw for both DQN and recon heads
        if self.ccfg.sigma2_channel > 0:
            std = self.ccfg.sigma2_channel ** 0.5
            eps = torch.randn_like(z_online) * std       # reparameterized
            z_hat = z_online + eps
        else:
            z_hat = z_online

        # ---- L_DQN: Huber (smooth L1) loss ----
        q_online = self.controller(z_hat)                          # (B, 2)
        q_taken  = q_online.gather(1, act_b.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            # Target Q: encoder_tgt + controller_tgt, NO channel noise (stability)
            z_next_tgt  = self.target_encoder(next_obs_b)          # (B, n)
            q_next_tgt  = self.target_controller(z_next_tgt)       # (B, 2)
            q_next_max  = q_next_tgt.max(dim=1).values             # (B,)
            y_target    = rew_b + self.tcfg.gamma * q_next_max * (1.0 - done_b)

        loss_dqn = F.smooth_l1_loss(q_taken, y_target)

        # ---- L_IB: nonlinear IB bound ----
        loss_ib = torch.tensor(0.0, device=self.device)
        if self.ibcfg.use_ib and self.ibcfg.beta > 0.0:
            loss_ib = nonlinear_ib_bound(z_online, self.ccfg.sigma2_total)

        # ---- L_llm: weighted feature reconstruction ----
        loss_llm = torch.tensor(0.0, device=self.device)
        if self.vcfg.enabled and self.recon_head is not None and self.vcfg.lambda_llm > 0.0:
            x_hat = self.recon_head(z_hat)                         # (B, 4)
            # w is a numpy array of constants — convert to tensor on the right device
            w = torch.tensor(self._llm_weights, dtype=torch.float32, device=self.device)
            # Per-feature MSE, weighted by LLM importance
            per_feature_mse = ((x_hat - obs_b) ** 2).mean(dim=0)  # (4,)
            loss_llm = (w * per_feature_mse).sum()

        # ---- Combined loss ----
        loss_total = (
            loss_dqn
            + self.ibcfg.beta * loss_ib
            + self.vcfg.lambda_llm * loss_llm
        )

        # NaN guard
        if torch.isnan(loss_total):
            print(f"[DQNAgent] WARNING: NaN detected in loss! dqn={loss_dqn.item():.4f}, "
                  f"ib={loss_ib.item():.4f}, llm={loss_llm.item():.4f}")
            return None

        self.optimizer.zero_grad()
        loss_total.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.controller.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()

        # (Epsilon decay is now handled manually per-episode in train.py)
        
        return {
            "loss_dqn":   loss_dqn.item(),
            "loss_ib":    loss_ib.item(),
            "loss_llm":   loss_llm.item(),
            "loss_total": loss_total.item(),
        }

    # ------------------------------------------------------------------
    # Target network management
    # ------------------------------------------------------------------

    def update_target(self) -> None:
        """Hard copy: online → target (called every target_update_freq episodes)."""
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_controller.load_state_dict(self.controller.state_dict())

    def _freeze_target(self) -> None:
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        for p in self.target_controller.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save full online network state dict."""
        torch.save(
            {
                "encoder":       self.encoder.state_dict(),
                "controller":    self.controller.state_dict(),
                "recon_head":    self.recon_head.state_dict() if self.recon_head else None,
                "epsilon":       self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load online network state dict from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.controller.load_state_dict(ckpt["controller"])
        if self.recon_head and ckpt.get("recon_head") is not None:
            self.recon_head.load_state_dict(ckpt["recon_head"])
        self.epsilon = ckpt.get("epsilon", self.tcfg.eps_end)

    # ------------------------------------------------------------------
    # Eval helpers
    # ------------------------------------------------------------------

    def eval_mode(self) -> None:
        """Switch to eval mode (disables dropout/batchnorm, no grad)."""
        self.encoder.eval()
        self.controller.eval()
        if self.recon_head:
            self.recon_head.eval()

    def train_mode(self) -> None:
        self.encoder.train()
        self.controller.train()
        if self.recon_head:
            self.recon_head.train()
