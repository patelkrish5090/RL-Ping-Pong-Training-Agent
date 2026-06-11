"""
config.py — All hyperparameters as dataclasses for the Variant A RL-over-Noisy-Channels project.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """CartPole environment configuration."""
    pole_length: float = 0.5
    force_magnitude: float = 10.0
    max_episode_steps: int = 200


@dataclass
class ChannelConfig:
    """AWGN channel configuration."""
    n: int = 8                       # Bottleneck dimension (number of channel uses)
    sigma2_channel: float = 0.0      # Channel noise variance (non-trainable)
    sigma2_trainable: float = 0.0    # Trainable IB noise variance (receiver side)

    @property
    def sigma2_total(self) -> float:
        """Total noise variance = channel + trainable IB noise."""
        return self.sigma2_channel + self.sigma2_trainable


@dataclass
class NetworkConfig:
    """Neural network architecture sizes."""
    encoder_hidden: List[int] = field(default_factory=lambda: [64, 64])
    controller_hidden: List[int] = field(default_factory=lambda: [64, 64])
    recon_head_hidden: List[int] = field(default_factory=lambda: [32])


@dataclass
class TrainingConfig:
    """DQN training hyperparameters."""
    n_episodes: int = 3000
    batch_size: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995
    replay_buffer_size: int = 10000
    min_replay_size: int = 1000
    target_update_freq: int = 10     # Update target network every N episodes


@dataclass
class IBConfig:
    """Information Bottleneck configuration."""
    beta: float = 0.0                # IB regularization weight (Lagrange multiplier)
    use_ib: bool = True              # Whether to include the IB loss term


@dataclass
class VariantAConfig:
    """Variant A: LLM-guided semantic encoder reward configuration."""
    enabled: bool = True
    lambda_llm: float = 0.1             # Weight of auxiliary LLM reconstruction loss
    llm_query_interval: int = 50        # Query LLM every K episodes
    plateau_threshold: float = 10.0     # Reward drop (vs. prior query avg) to trigger early query
    plateau_window: int = 20            # Window size for moving-average plateau detection
    # Ollama LLM settings
    llm_model: str = "deepseek-r1:32b"
    ollama_url: str = "http://localhost:11434"   # Override via CLI --ollama-url
    api_timeout: int = 120                       # Seconds; DeepSeek-R1 32b reasoning takes time
    # Physics-based fallback weights: [x_cart, x_dot, theta, theta_dot]
    fallback_weights_low_noise: List[float] = field(
        default_factory=lambda: [0.15, 0.20, 0.35, 0.30]
    )
    fallback_weights_high_noise: List[float] = field(
        default_factory=lambda: [0.08, 0.20, 0.42, 0.30]
    )
    noise_threshold: float = 0.5     # σ² above which "high noise" fallback applies


@dataclass
class EvalConfig:
    """Generalization evaluation grid (matching paper exactly)."""
    pole_lengths: List[float] = field(default_factory=lambda: [
        0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6
    ])
    force_magnitudes: List[float] = field(default_factory=lambda: [
        2, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40
    ])
    n_eval_episodes: int = 100           # Paper uses 100 episodes per grid cell


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Master experiment configuration."""
    env: EnvConfig = field(default_factory=EnvConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ib: IBConfig = field(default_factory=IBConfig)
    variant_a: VariantAConfig = field(default_factory=VariantAConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    experiment_name: str = "variant_a"
    results_dir: str = "results"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _to_dict(obj) -> dict:
    """Recursively convert dataclass to plain dict, handling nested dataclasses."""
    if dataclasses.is_dataclass(obj):
        d = {}
        for f in dataclasses.fields(obj):
            d[f.name] = _to_dict(getattr(obj, f.name))
        return d
    elif isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    else:
        return obj


def save_config(config: ExperimentConfig, path: str) -> None:
    """Save config to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_dict(config), f, indent=2)


def get_results_dir(config: ExperimentConfig) -> str:
    """Return the per-experiment results directory path."""
    return os.path.join(config.results_dir, config.experiment_name)
