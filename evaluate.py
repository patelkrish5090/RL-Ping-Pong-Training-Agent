"""
evaluate.py — Generalization evaluation on the CartPole pole_length × force_magnitude grid.

Matches the paper (Kam et al., MILCOM 2024) methodology exactly:
  - Same trained model used for all grid cells (no retraining)
  - Greedy policy (no epsilon)
  - Channel noise is active (sigma2_channel from config)
  - Reconstruction head is NOT used (model.eval() + torch.no_grad())
  - 100 episodes per grid cell

Outputs (saved to results/{name}/):
  - eval_grid.npy     — shape (n_forces, n_lengths), float32
  - eval_grid.json    — same as dict
  - overall_avg.txt   — scalar mean over all grid cells
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from config import ExperimentConfig, get_results_dir
from dqn_agent import DQNAgent


def evaluate(
    config: ExperimentConfig,
    checkpoint_path: Optional[str] = None,
    agent: Optional[DQNAgent] = None,
) -> Tuple[np.ndarray, float]:
    """
    Evaluate a trained agent on the full generalization grid.

    Parameters
    ----------
    config : ExperimentConfig
    checkpoint_path : str, optional
        Path to model.pt.  If None, uses results/{name}/model.pt.
    agent : DQNAgent, optional
        Pre-loaded agent (skip checkpoint loading).

    Returns
    -------
    grid : np.ndarray, shape (n_forces, n_lengths)
        Mean episode rewards.
    overall_avg : float
        Mean over all grid cells.
    """
    results_dir = get_results_dir(config)
    os.makedirs(results_dir, exist_ok=True)

    # --- Load agent ---
    if agent is None:
        agent = DQNAgent(config)
        if checkpoint_path is None:
            checkpoint_path = os.path.join(results_dir, "model.pt")
        agent.load(checkpoint_path)
        print(f"[evaluate] Loaded checkpoint: {checkpoint_path}")

    agent.eval_mode()

    ecfg = config.eval
    pole_lengths    = ecfg.pole_lengths
    force_mags      = ecfg.force_magnitudes
    n_eval          = ecfg.n_eval_episodes

    grid = np.zeros((len(force_mags), len(pole_lengths)), dtype=np.float32)

    # Create a single environment and monkey-patch it for each grid cell
    env = gym.make("CartPole-v1", max_episode_steps=config.env.max_episode_steps)

    pbar = tqdm(
        total=len(force_mags) * len(pole_lengths),
        desc=f"Eval {config.experiment_name}",
        unit="cell",
    )

    with torch.no_grad():
        for fi, force in enumerate(force_mags):
            for li, length in enumerate(pole_lengths):
                # Monkey-patch physics parameters
                env.unwrapped.length    = length
                env.unwrapped.force_mag = force

                rewards = []
                for ep in range(n_eval):
                    obs, _ = env.reset(seed=config.seed + ep)
                    ep_reward = 0.0
                    done = False
                    while not done:
                        action = agent.select_action(obs, greedy=True)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        ep_reward += float(reward)
                    rewards.append(ep_reward)

                grid[fi, li] = float(np.mean(rewards))
                pbar.update(1)

    pbar.close()
    env.close()

    overall_avg = float(grid.mean())

    # --- Save results ---
    np.save(os.path.join(results_dir, "eval_grid.npy"), grid)

    grid_dict = {
        "pole_lengths":    pole_lengths,
        "force_magnitudes": force_mags,
        "grid":            grid.tolist(),
        "overall_avg":     overall_avg,
    }
    with open(os.path.join(results_dir, "eval_grid.json"), "w") as f:
        json.dump(grid_dict, f, indent=2)

    with open(os.path.join(results_dir, "overall_avg.txt"), "w") as f:
        f.write(f"{overall_avg:.4f}\n")

    print(f"[evaluate] {config.experiment_name}: overall_avg = {overall_avg:.2f}")
    return grid, overall_avg


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from config import ExperimentConfig, ChannelConfig, IBConfig, VariantAConfig

    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent on the generalization grid.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--name",       type=str, required=True, help="Experiment name")
    parser.add_argument("--n",          type=int, default=8,     help="Bottleneck dimension")
    parser.add_argument("--sigma2",     type=float, default=0.0, help="Channel noise variance")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        channel=ChannelConfig(n=args.n, sigma2_channel=args.sigma2),
        experiment_name=args.name,
        seed=args.seed,
    )
    evaluate(cfg, checkpoint_path=args.checkpoint)
