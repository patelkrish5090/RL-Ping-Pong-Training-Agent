"""
Iterative Reward Refinement Training — Anti-Jamming Channel Selection
======================================================================
MILCOM Research Prototype: LLM-Guided Reward Shaping for RL-Based
Anti-Jamming Channel Selection in Tactical Wireless Networks

Training loop:
  1. Train PPO (MlpPolicy) on WirelessAntiJammingEnv
  2. Analyze recent episode metrics (PDR, throughput, jammed rate, etc.)
  3. Send training summary + current reward code to local LLM (Ollama)
  4. Save LLM-generated reward to rewards/history/
  5. Hot-swap current_reward.py
  6. Continue training
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from config import (
    N_CHANNELS, N_ENVS, PPO_CONFIG,
    EPISODES_PER_UPDATE, MAX_ITERATIONS, TARGET_PDR,
    TIMESTEPS_PER_ITERATION, LOG_DIR, MODEL_DIR, REWARD_FILE,
    MAX_STEPS_PER_EPISODE, JAMMER_MODES, JAMMER_CHANGE_INTERVAL,
    N_JAMMED_CHANNELS, SWEEP_WIDTH, REACTIVE_JAM_PROB, REACTIVE_EXTRA_RANDOM,
    SNR_MEAN, SNR_STD, SNR_THRESHOLD, QUEUE_CAPACITY, ARRIVAL_RATE,
    MAX_PACKET_ARRIVALS, ENERGY_PER_TX, SWITCH_DISRUPTION_PROB,
    SWITCH_ENERGY_COST, SWITCH_REWARD_PENALTY,
)
from wireless_env import WirelessAntiJammingEnv
from state_extractor import extract_wireless_state, compute_wireless_features
from episode_analyzer import EpisodeAnalyzer
from llm_interface import LLMRewardGenerator, test_ollama_connection


def _tb_log_dir():
    """Return tensorboard log dir only if tensorboard is installed, else None."""
    try:
        import tensorboard  # noqa
        return str(LOG_DIR)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Callback: per-step reward shaping + episode tracking
# ---------------------------------------------------------------------------

class AntiJammingCallback(BaseCallback):
    """
    SB3 callback that:
      - Applies the hot-swappable shaped reward at each step
      - Tracks per-episode wireless metrics (PDR, throughput, jammed rate, etc.)
      - Passes completed episode stats to the EpisodeAnalyzer
    """

    def __init__(self, analyzer: EpisodeAnalyzer, reward_file: Path, verbose: int = 0):
        super().__init__(verbose)
        self.analyzer = analyzer
        self.reward_file = reward_file
        self.compute_reward = None  # Loaded from reward_file

        # Per-env state tracking
        self._prev_states = {}
        self._ep_stats = {}   # Running episode accumulators per env

        self.reload_reward()

    def reload_reward(self) -> bool:
        """Reload compute_reward function from file (hot-swap)."""
        try:
            import importlib.util
            if not self.reward_file.exists():
                print(f"Reward file not found: {self.reward_file}")
                return False

            spec = importlib.util.spec_from_file_location("reward_module", self.reward_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, 'compute_reward'):
                self.compute_reward = module.compute_reward
                print(f"Reloaded reward function from {self.reward_file}")
                return True
            return False
        except Exception as e:
            print(f"Error reloading reward: {e}")
            return False

    def _init_ep(self, env_id: int):
        """Initialize per-env episode tracking."""
        self._prev_states[env_id] = None
        self._ep_stats[env_id] = {
            "delivered": 0,
            "total_tx": 0,
            "jammed_tx": 0,
            "switches": 0,
            "energy": 0.0,
            "queue_sum": 0,
            "steps": 0,
        }

    def _on_step(self) -> bool:
        """Called at each training step — apply shaped reward + track metrics."""
        # Unwrap DummyVecEnv to get individual envs
        envs = self.training_env.envs if hasattr(self.training_env, 'envs') else None
        if envs is None:
            return True

        for i, env in enumerate(envs):
            if i not in self._prev_states:
                self._init_ep(i)

            try:
                # Unwrap Monitor/Gym wrappers to reach WirelessAntiJammingEnv
                inner_env = env
                while hasattr(inner_env, 'env'):
                    inner_env = inner_env.env

                # Get state and features from the env's last step info
                # info is accessible via self.locals after each step
                infos = self.locals.get('infos', [{}])
                info = infos[i] if i < len(infos) else {}

                raw_state = info.get('state', {})
                raw_features = info.get('features', {})

                if not raw_state:
                    continue

                state = extract_wireless_state(raw_state)
                features = compute_wireless_features(state, self._prev_states[i])

                # Merge env-provided features (has is_jammed, tx_success, etc.)
                for k, v in raw_features.items():
                    if k not in features:
                        features[k] = v

                # Track episode-level metrics
                ep = self._ep_stats[i]
                ep["steps"] += 1
                ep["energy"] += state.get("ep_energy", 0.0) - (
                    self._prev_states[i].get("ep_energy", 0.0) if self._prev_states[i] else 0.0
                )
                ep["queue_sum"] += state.get("queue_length", 0)

                if features.get("tx_success", False):
                    ep["delivered"] += 1
                if info.get("is_jammed", False):
                    ep["jammed_tx"] += 1
                if features.get("switched", False):
                    ep["switches"] += 1
                ep["total_tx"] += 1

                # Apply shaped reward
                if self.compute_reward is not None:
                    env_reward = info.get("env_reward",
                                          self.locals.get('rewards', [0.0])[i] if 'rewards' in self.locals else 0.0)
                    done = self.locals.get('dones', [False])[i] if 'dones' in self.locals else False

                    try:
                        shaped = self.compute_reward(state, features, float(env_reward), done)
                        if 'rewards' in self.locals:
                            self.locals['rewards'][i] = shaped
                    except Exception:
                        pass  # Keep original reward on error

                self._prev_states[i] = state

            except Exception:
                pass  # Never crash training

            # Episode end: record stats
            dones = self.locals.get('dones', None)
            if dones is not None and dones[i]:
                infos = self.locals.get('infos', [{}])
                info = infos[i] if i < len(infos) else {}

                # Prefer episode_summary from env if available
                ep_summary = info.get('episode_summary', None)

                if ep_summary:
                    self.analyzer.add_episode(ep_summary)
                else:
                    # Fallback: use our accumulated stats
                    ep = self._ep_stats.get(i, {})
                    n_tx = max(ep.get("total_tx", 1), 1)
                    n_steps = max(ep.get("steps", 1), 1)
                    self.analyzer.add_episode({
                        "pdr":         ep.get("delivered", 0) / n_tx,
                        "throughput":  ep.get("delivered", 0) / n_steps,
                        "jammed_rate": ep.get("jammed_tx", 0) / n_tx,
                        "avg_queue":   ep.get("queue_sum", 0) / n_steps,
                        "switches":    ep.get("switches", 0),
                        "switch_rate": ep.get("switches", 0) / n_steps,
                        "energy":      ep.get("energy", 0.0),
                        "total_steps": n_steps,
                        "total_delivered": ep.get("delivered", 0),
                        "total_tx":    ep.get("total_tx", 0),
                    })

                # Reset per-env tracking
                self._init_ep(i)

        return True


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(seed: int = None):
    """Create a single WirelessAntiJammingEnv wrapped in Monitor."""
    def _init():
        env = WirelessAntiJammingEnv(
            n_channels=N_CHANNELS,
            max_steps=MAX_STEPS_PER_EPISODE,
            jammer_modes=JAMMER_MODES,
            jammer_change_interval=JAMMER_CHANGE_INTERVAL,
            n_jammed_channels=N_JAMMED_CHANNELS,
            sweep_width=SWEEP_WIDTH,
            reactive_jam_prob=REACTIVE_JAM_PROB,
            reactive_extra_random=REACTIVE_EXTRA_RANDOM,
            snr_mean=SNR_MEAN,
            snr_std=SNR_STD,
            snr_threshold=SNR_THRESHOLD,
            queue_capacity=QUEUE_CAPACITY,
            arrival_rate=ARRIVAL_RATE,
            max_packet_arrivals=MAX_PACKET_ARRIVALS,
            energy_per_tx=ENERGY_PER_TX,
            switch_disruption_prob=SWITCH_DISRUPTION_PROB,
            switch_energy_cost=SWITCH_ENERGY_COST,
            switch_reward_penalty=SWITCH_REWARD_PENALTY,
            seed=seed,
        )
        env = Monitor(env)
        return env
    return _init


def make_vec_env(n_envs: int = N_ENVS):
    """Create vectorized WirelessAntiJammingEnv environments."""
    return DummyVecEnv([make_env(seed=i) for i in range(n_envs)])


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_iterative(
    timesteps_per_iter: int = TIMESTEPS_PER_ITERATION,
    max_iterations: int = MAX_ITERATIONS,
    target_pdr: float = TARGET_PDR,
    episodes_for_analysis: int = EPISODES_PER_UPDATE,
    use_llm: bool = True,
    verbose: int = 1,
    resume_model: str = None,
):
    """
    Main iterative LLM-guided reward refinement training loop.

    Args:
        timesteps_per_iter:   Training timesteps between LLM reward updates
        max_iterations:       Maximum number of LLM refinement iterations
        target_pdr:           Target PDR to achieve (stop condition)
        episodes_for_analysis: Number of episodes to analyze for LLM feedback
        use_llm:              Whether to use LLM for reward refinement
        verbose:              Verbosity level (0-2)
        resume_model:         Path to existing model to resume training from
    """
    print("=" * 65)
    print("LLM-Guided Reward Shaping — Anti-Jamming Channel Selection")
    print("MILCOM Research Prototype")
    print("=" * 65)

    # Check Ollama if using LLM
    if use_llm:
        if not test_ollama_connection():
            print("\nWARNING: Ollama not running! Start with 'ollama serve'")
            print("Continuing without LLM refinement...\n")
            use_llm = False
        else:
            print("✓ Ollama connection OK")

    # Initialize components
    analyzer = EpisodeAnalyzer()
    llm = LLMRewardGenerator() if use_llm else None

    # Create environments
    print(f"\nCreating {N_ENVS} parallel environments...")
    print(f"  Channels: {N_CHANNELS} | Jammer modes: {JAMMER_MODES}")
    print(f"  Episode length: {MAX_STEPS_PER_EPISODE} steps")
    env = make_vec_env(N_ENVS)

    # Create callback
    callback = AntiJammingCallback(analyzer, REWARD_FILE, verbose=verbose)

    # Initialize or resume PPO
    if resume_model:
        print(f"\nResuming training from {resume_model}...")
        model = PPO.load(resume_model, env=env)
    else:
        print("\nInitializing PPO agent (MlpPolicy)...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=verbose,
            tensorboard_log=_tb_log_dir(),
            device="cpu",   # MlpPolicy is faster on CPU; avoids spurious GPU warning
            **PPO_CONFIG
        )

    # Training state
    total_timesteps = 0
    best_pdr = 0.0

    print(f"\nStarting iterative training...")
    print(f"  Timesteps per iteration:  {timesteps_per_iter:,}")
    print(f"  Max iterations:           {max_iterations}")
    print(f"  Target PDR:               {target_pdr}")
    print(f"  LLM refinement:           {'Enabled' if use_llm else 'Disabled'}")
    print()

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*65}")
        print(f"ITERATION {iteration}/{max_iterations}")
        print(f"{'='*65}")

        # --- Train ---
        start_time = time.time()
        model.learn(
            total_timesteps=timesteps_per_iter,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        elapsed = time.time() - start_time
        total_timesteps += timesteps_per_iter

        print(f"\nTraining completed in {elapsed:.1f}s | Total steps: {total_timesteps:,}")

        # --- Analyze ---
        try:
            analysis = analyzer.analyze(episodes_for_analysis)
            avg_pdr    = analysis.get('avg_pdr', 0.0)
            avg_jammed = analysis.get('avg_jammed_rate', 0.0)
            avg_queue  = analysis.get('avg_queue', 0.0)
            avg_switch = analysis.get('avg_switch_rate', 0.0)

            print(f"\n--- Performance Analysis ({analysis['n_episodes']} episodes) ---")
            print(f"  Avg PDR:              {avg_pdr:.3f}  (best: {analysis.get('best_pdr', 0):.3f})")
            print(f"  Avg Throughput:       {analysis.get('avg_throughput', 0):.3f} pkts/step")
            print(f"  Avg Jammed TX Rate:   {avg_jammed:.3f}")
            print(f"  Avg Queue Length:     {avg_queue:.2f}")
            print(f"  Avg Switch Rate:      {avg_switch:.3f}/step")
            print(f"  Good Episode Rate:    {analysis.get('good_episode_rate', 0):.1f}%")
            print(f"\nBehaviors:\n{analysis.get('behaviors', 'N/A')}")

        except Exception as e:
            print(f"\nERROR in Performance Analysis: {e}")
            print("Skipping analysis for this iteration.")
            avg_pdr = 0.0
            analysis = {}

        # --- Save best model ---
        if avg_pdr > best_pdr:
            best_pdr = avg_pdr
            model_path = MODEL_DIR / f"antijam_best_iter{iteration}"
            model.save(str(model_path))
            print(f"\n>>> New best PDR! Model saved to {model_path}")

        # --- Check target ---
        if avg_pdr >= target_pdr:
            print(f"\n*** TARGET PDR REACHED: {avg_pdr:.3f} >= {target_pdr} ***")
            break

        # --- LLM refinement ---
        if use_llm and iteration < max_iterations:
            print(f"\n--- LLM Reward Refinement (Iteration {iteration}) ---")

            current_code = callback.reward_file.read_text() if callback.reward_file.exists() else None

            try:
                summary = analyzer.get_summary_for_llm(episodes_for_analysis)
            except Exception as e:
                print(f"Error generating summary for LLM: {e}")
                summary = "Error generating summary."

            try:
                metrics_for_llm = {
                    "avg_pdr":          analysis.get("avg_pdr", 0.0),
                    "avg_jammed_rate":  analysis.get("avg_jammed_rate", 0.0),
                    "avg_switch_rate":  analysis.get("avg_switch_rate", 0.0),
                    "avg_queue":        analysis.get("avg_queue", 0.0),
                }
                new_code, reasoning = llm.generate_reward(
                    summary, current_code, metrics=metrics_for_llm
                )

                # Truncate reasoning for display
                reasoning_display = reasoning[:300] + "..." if len(reasoning) > 300 else reasoning
                print(f"LLM Reasoning (excerpt): {reasoning_display}")

                llm.save_current_reward(new_code)
                callback.reload_reward()
                print("✓ Reward function updated and hot-swapped!")

            except Exception as e:
                print(f"LLM refinement failed: {e}")
                print("Continuing with current reward function...")

        # --- Clear episode buffer ---
        analyzer.clear()

    # --- Training complete ---
    print(f"\n{'='*65}")
    print("TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"Total timesteps:  {total_timesteps:,}")
    print(f"Best avg PDR:     {best_pdr:.3f}")
    print(f"Iterations:       {iteration}")

    final_path = MODEL_DIR / "antijam_final"
    model.save(str(final_path))
    print(f"Final model saved to {final_path}")

    env.close()
    return model, best_pdr


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str,
    n_episodes: int = 20,
    render: bool = False,
    verbose: bool = True,
):
    """
    Evaluate a trained model on WirelessAntiJammingEnv.

    Args:
        model_path: Path to saved PPO model (.zip)
        n_episodes: Number of evaluation episodes
        render:     Print per-step channel state
        verbose:    Print per-episode stats
    """
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    env = WirelessAntiJammingEnv(
        n_channels=N_CHANNELS,
        max_steps=MAX_STEPS_PER_EPISODE,
        jammer_modes=JAMMER_MODES,
        jammer_change_interval=JAMMER_CHANGE_INTERVAL,
        n_jammed_channels=N_JAMMED_CHANNELS,
        sweep_width=SWEEP_WIDTH,
        reactive_jam_prob=REACTIVE_JAM_PROB,
        reactive_extra_random=REACTIVE_EXTRA_RANDOM,
        snr_mean=SNR_MEAN,
        snr_std=SNR_STD,
        snr_threshold=SNR_THRESHOLD,
        queue_capacity=QUEUE_CAPACITY,
        arrival_rate=ARRIVAL_RATE,
        max_packet_arrivals=MAX_PACKET_ARRIVALS,
        energy_per_tx=ENERGY_PER_TX,
        switch_disruption_prob=SWITCH_DISRUPTION_PROB,
        switch_energy_cost=SWITCH_ENERGY_COST,
        switch_reward_penalty=SWITCH_REWARD_PENALTY,
        seed=0,
    )

    ep_pdrs, ep_jammed, ep_switches, ep_throughputs = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            if render:
                env.render()

        summary = env.get_episode_summary()
        ep_pdrs.append(summary["pdr"])
        ep_jammed.append(summary["jammed_rate"])
        ep_switches.append(summary["switches"])
        ep_throughputs.append(summary["throughput"])

        if verbose:
            print(f"Episode {ep+1:3d}: PDR={summary['pdr']:.3f}  "
                  f"Jammed={summary['jammed_rate']:.3f}  "
                  f"Switches={summary['switches']:3d}  "
                  f"Throughput={summary['throughput']:.3f}")

    print(f"\n{'='*55}")
    print("EVALUATION SUMMARY")
    print(f"{'='*55}")
    print(f"Episodes:          {n_episodes}")
    print(f"Avg PDR:           {np.mean(ep_pdrs):.3f} ± {np.std(ep_pdrs):.3f}")
    print(f"Avg Jammed Rate:   {np.mean(ep_jammed):.3f} ± {np.std(ep_jammed):.3f}")
    print(f"Avg Throughput:    {np.mean(ep_throughputs):.3f} ± {np.std(ep_throughputs):.3f}")
    print(f"Avg Switches/ep:   {np.mean(ep_switches):.1f}")
    print(f"PDR >= 0.75 rate:  {sum(1 for p in ep_pdrs if p >= 0.75) / n_episodes * 100:.1f}%")

    env.close()
    return {
        "avg_pdr": float(np.mean(ep_pdrs)),
        "avg_jammed_rate": float(np.mean(ep_jammed)),
        "avg_throughput": float(np.mean(ep_throughputs)),
        "avg_switches": float(np.mean(ep_switches)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-Guided Reward Shaping — Anti-Jamming Channel Selection (MILCOM)"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the agent with iterative LLM reward refinement')
    train_parser.add_argument('--timesteps', type=int, default=TIMESTEPS_PER_ITERATION,
                              help=f'Timesteps per LLM iteration (default: {TIMESTEPS_PER_ITERATION})')
    train_parser.add_argument('--iterations', type=int, default=MAX_ITERATIONS,
                              help=f'Max LLM refinement iterations (default: {MAX_ITERATIONS})')
    train_parser.add_argument('--target-pdr', type=float, default=TARGET_PDR,
                              help=f'Target PDR to stop training (default: {TARGET_PDR})')
    train_parser.add_argument('--no-llm', action='store_true',
                              help='Disable LLM refinement (PPO with fixed reward only)')
    train_parser.add_argument('--resume', type=str, default=None,
                              help='Path to existing model to resume training from')
    train_parser.add_argument('--verbose', type=int, default=1,
                              help='Verbosity level 0-2 (default: 1)')

    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained model')
    eval_parser.add_argument('model', type=str, help='Path to model file (.zip)')
    eval_parser.add_argument('--episodes', type=int, default=20,
                             help='Number of evaluation episodes (default: 20)')
    eval_parser.add_argument('--render', action='store_true',
                             help='Print channel state at each step')
    eval_parser.add_argument('--quiet', action='store_true',
                             help='Only print summary, not per-episode stats')

    args = parser.parse_args()

    if args.command == 'train':
        train_iterative(
            timesteps_per_iter=args.timesteps,
            max_iterations=args.iterations,
            target_pdr=args.target_pdr,
            use_llm=not args.no_llm,
            verbose=args.verbose,
            resume_model=args.resume,
        )
    elif args.command == 'eval':
        evaluate(
            model_path=args.model,
            n_episodes=args.episodes,
            render=args.render,
            verbose=not args.quiet,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
