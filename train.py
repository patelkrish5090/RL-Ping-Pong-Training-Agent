"""
train.py — Training loop for the Variant A RL-over-Noisy-Channels system.

Runs a full DQN training session with:
  - JSCC encoder + AWGN channel + Q-controller
  - Optional IB regularization
  - Optional Variant A: LLM-guided feature reconstruction loss

Outputs (saved to results/{experiment_name}/):
  - model.pt          — final checkpoint
  - config.json       — full experiment config
  - training_log.jsonl — per-episode metrics
  - llm_log.jsonl     — LLM query log (if Variant A enabled)
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from config import ExperimentConfig, save_config, get_results_dir
from dqn_agent import DQNAgent
from llm_advisor import LLMAdvisor


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_env(config: ExperimentConfig) -> gym.Env:
    """Create and configure the CartPole environment."""
    env = gym.make("CartPole-v1", max_episode_steps=config.env.max_episode_steps)
    # Apply custom physics parameters
    env.unwrapped.length    = config.env.pole_length
    env.unwrapped.force_mag = config.env.force_magnitude
    env.reset(seed=config.seed)
    return env


def train(config: ExperimentConfig, ollama_url: Optional[str] = None) -> Dict:
    """
    Run one full training session.

    Parameters
    ----------
    config : ExperimentConfig
    ollama_url : str, optional
        Override the Ollama server URL from CLI (e.g., 'http://192.168.1.5:11434').

    Returns
    -------
    summary : dict
        Keys: 'overall_avg_reward' (from eval grid if run, else last-100 mean),
              'experiment_name', 'n_episodes_trained'.
    """
    # Override Ollama URL if provided via CLI
    if ollama_url:
        config.variant_a.ollama_url = ollama_url

    results_dir = get_results_dir(config)
    os.makedirs(results_dir, exist_ok=True)

    # Save config immediately
    save_config(config, os.path.join(results_dir, "config.json"))

    # Reproducibility
    _seed_everything(config.seed)

    # Environment
    env = _make_env(config)

    # Agent
    agent = DQNAgent(config)
    print(f"\n[train] Starting: {config.experiment_name}")
    print(f"  Device  : {agent.device}")
    print(f"  n={config.channel.n}, sigma2={config.channel.sigma2_channel}, "
          f"beta={config.ib.beta:.2e}, lambda={config.variant_a.lambda_llm}")
    print(f"  Variant A: {config.variant_a.enabled}")
    print(f"  Episodes: {config.training.n_episodes}")

    # LLM Advisor (only if Variant A enabled)
    advisor: Optional[LLMAdvisor] = None
    if config.variant_a.enabled:
        advisor = LLMAdvisor(config.variant_a, config.channel, results_dir)
        print(f"  LLM: {config.variant_a.llm_model} @ {config.variant_a.ollama_url}")

    # Logging
    log_path = os.path.join(results_dir, "training_log.jsonl")
    episode_rewards: list = []

    start_time = time.monotonic()

    # --- Training loop ---
    pbar = tqdm(range(config.training.n_episodes), desc=config.experiment_name, unit="ep")
    for episode in pbar:

        # --- Optionally update LLM weights ---
        current_llm_weights = [0.25, 0.25, 0.25, 0.25]
        if advisor is not None:
            w = advisor.get_weights(episode, episode_rewards)
            agent.set_llm_weights(w)
            current_llm_weights = w.tolist()

        # --- Run one episode ---
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, float(reward), next_obs, done)
            obs = next_obs
            ep_reward += float(reward)

        episode_rewards.append(ep_reward)

        # --- Training update (if replay buffer has enough samples) ---
        losses = agent.update()

        # --- Target network update ---
        if (episode + 1) % config.training.target_update_freq == 0:
            agent.update_target()

        # --- Logging ---
        log_entry = {
            "episode":      episode,
            "reward":       ep_reward,
            "epsilon":      round(agent.epsilon, 4),
            "loss_dqn":     losses["loss_dqn"]   if losses else None,
            "loss_ib":      losses["loss_ib"]    if losses else None,
            "loss_llm":     losses["loss_llm"]   if losses else None,
            "loss_total":   losses["loss_total"] if losses else None,
            "llm_weights":  current_llm_weights,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # --- Progress bar update ---
        if len(episode_rewards) >= 100:
            avg100 = np.mean(episode_rewards[-100:])
            pbar.set_postfix({"avg100": f"{avg100:.1f}", "eps": f"{agent.epsilon:.3f}"})
        elif len(episode_rewards) >= 20:
            avg20 = np.mean(episode_rewards[-20:])
            pbar.set_postfix({"avg20": f"{avg20:.1f}", "eps": f"{agent.epsilon:.3f}"})

    # --- Save model ---
    ckpt_path = os.path.join(results_dir, "model.pt")
    agent.save(ckpt_path)
    env.close()

    elapsed = time.monotonic() - start_time
    last100_mean = float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 else float(np.mean(episode_rewards))

    print(f"\n[train] Done: {config.experiment_name} in {elapsed/60:.1f} min")
    print(f"  Last-100 mean reward: {last100_mean:.2f}")
    print(f"  Checkpoint: {ckpt_path}")

    return {
        "experiment_name":     config.experiment_name,
        "last100_mean_reward": last100_mean,
        "n_episodes_trained":  config.training.n_episodes,
        "elapsed_sec":         elapsed,
    }
