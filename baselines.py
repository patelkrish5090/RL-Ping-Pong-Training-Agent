"""
Baseline Evaluations — Anti-Jamming Channel Selection
======================================================
Compares four strategies on WirelessAntiJammingEnv:

  1. random     — Uniform random channel selection
  2. heuristic  — Avoid last jammed channel (greedy)
  3. ppo-sparse — PPO trained with sparse env_reward only (no shaping)
  4. ppo-shaped — PPO trained with manually shaped reward (no LLM)

Run examples:
    python baselines.py --mode random --episodes 50
    python baselines.py --mode heuristic --episodes 50
    python baselines.py --mode ppo-sparse --train-steps 100000 --episodes 20
    python baselines.py --mode ppo-shaped --train-steps 100000 --episodes 20
    python baselines.py --mode all --episodes 30 --train-steps 50000
"""

import argparse
import time
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from wireless_env import WirelessAntiJammingEnv
from config import (
    N_CHANNELS, MAX_STEPS_PER_EPISODE, JAMMER_MODES, JAMMER_CHANGE_INTERVAL,
    N_JAMMED_CHANNELS, SWEEP_WIDTH, REACTIVE_JAM_PROB, REACTIVE_EXTRA_RANDOM,
    SNR_MEAN, SNR_STD, SNR_THRESHOLD, QUEUE_CAPACITY, ARRIVAL_RATE,
    MAX_PACKET_ARRIVALS, ENERGY_PER_TX, SWITCH_DISRUPTION_PROB,
    SWITCH_ENERGY_COST, SWITCH_REWARD_PENALTY,
    PPO_CONFIG, MODEL_DIR, LOG_DIR,
)


# ---------------------------------------------------------------------------
# Shaped reward for the manual-shaping baseline
# ---------------------------------------------------------------------------

def _manual_shaped_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    """
    Hand-crafted reward for the 'ppo-shaped' baseline.
    Stronger shaping than the LLM-initial reward, but still within safety bounds.
    """
    reward = env_reward

    # Strong penalty for jammed channel
    jammed_decay = np.asarray(state.get("jammed_decay", np.zeros(N_CHANNELS)))
    prev_ch = state.get("prev_channel", -1)
    if 0 <= prev_ch < N_CHANNELS:
        if jammed_decay[prev_ch] > 0.5:
            reward -= 0.08  # Avoid revisiting hot channels

    # Reward selecting a high success-rate channel
    success_rates = np.asarray(state.get("channel_success_rates", np.full(N_CHANNELS, 0.5)))
    if 0 <= prev_ch < N_CHANNELS and success_rates[prev_ch] > 0.7:
        reward += 0.05

    # Switching penalty (discourage thrashing)
    if features.get("switched", False):
        reward -= 0.02

    # Queue buildup penalty
    queue_pressure = state.get("queue_length", 0) / max(QUEUE_CAPACITY, 1)
    if queue_pressure > 0.75:
        reward -= 0.05
    elif queue_pressure > 0.5:
        reward -= 0.02

    return reward


# ---------------------------------------------------------------------------
# Env factory helpers
# ---------------------------------------------------------------------------

def make_eval_env(seed: int = 0):
    return WirelessAntiJammingEnv(
        n_channels=N_CHANNELS,
        max_steps=MAX_STEPS_PER_EPISODE,
        jammer_modes=JAMMER_MODES,
        jammer_change_interval=JAMMER_CHANGE_INTERVAL,
        n_jammed_channels=N_JAMMED_CHANNELS,
        sweep_width=SWEEP_WIDTH,
        reactive_jam_prob=REACTIVE_JAM_PROB,
        reactive_extra_random=REACTIVE_EXTRA_RANDOM,
        snr_mean=SNR_MEAN, snr_std=SNR_STD, snr_threshold=SNR_THRESHOLD,
        queue_capacity=QUEUE_CAPACITY,
        arrival_rate=ARRIVAL_RATE, max_packet_arrivals=MAX_PACKET_ARRIVALS,
        energy_per_tx=ENERGY_PER_TX,
        switch_disruption_prob=SWITCH_DISRUPTION_PROB,
        switch_energy_cost=SWITCH_ENERGY_COST,
        switch_reward_penalty=SWITCH_REWARD_PENALTY,
        seed=seed,
    )


def make_train_vec_env(n_envs: int = 4, seed: int = 0):
    def _make(i):
        def _init():
            env = make_eval_env(seed=seed + i)
            return Monitor(env)
        return _init
    return DummyVecEnv([_make(i) for i in range(n_envs)])


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def run_episodes(policy_fn, n_episodes: int, seed_offset: int = 100) -> dict:
    """
    Run `n_episodes` using `policy_fn(obs, env) -> action` and return aggregate metrics.
    """
    pdrs, jammed_rates, switch_rates, throughputs = [], [], [], []

    for ep in range(n_episodes):
        env = make_eval_env(seed=seed_offset + ep)
        obs, _ = env.reset()
        done = False

        while not done:
            action = policy_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        summary = env.get_episode_summary()
        pdrs.append(summary["pdr"])
        jammed_rates.append(summary["jammed_rate"])
        switch_rates.append(summary["switch_rate"])
        throughputs.append(summary["throughput"])
        env.close()

    return {
        "avg_pdr":          float(np.mean(pdrs)),
        "std_pdr":          float(np.std(pdrs)),
        "avg_jammed_rate":  float(np.mean(jammed_rates)),
        "avg_switch_rate":  float(np.mean(switch_rates)),
        "avg_throughput":   float(np.mean(throughputs)),
        "good_ep_rate":     float(sum(1 for p in pdrs if p >= 0.75) / n_episodes * 100),
    }


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

def random_policy(obs: np.ndarray, env: WirelessAntiJammingEnv) -> int:
    """Uniform random channel selection."""
    return env.action_space.sample()


def heuristic_policy(obs: np.ndarray, env: WirelessAntiJammingEnv) -> int:
    """
    Greedy heuristic: pick the channel with the lowest jammed_decay score
    that also has the highest recent success rate.
    """
    N = env.n_channels
    # obs layout: [success_rates[N] | snr[N] | jammed[N] | prev_ch | queue | energy | mode | step_frac]
    success_rates = obs[:N]
    jammed_decay  = obs[2*N : 3*N]

    # Score: high success rate, low jammed decay
    scores = success_rates - jammed_decay
    return int(np.argmax(scores))


# ---------------------------------------------------------------------------
# PPO training helpers
# ---------------------------------------------------------------------------

def train_ppo(reward_fn=None, train_steps: int = 100000, label: str = "ppo") -> PPO:
    """
    Train a PPO agent with an optional reward-shaping wrapper applied
    inside the callback.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class ShapingCallback(BaseCallback):
        def __init__(self, reward_fn):
            super().__init__()
            self._reward_fn = reward_fn

        def _on_step(self) -> bool:
            if self._reward_fn is None:
                return True
            infos = self.locals.get('infos', [{}])
            for i, info in enumerate(infos):
                raw_state = info.get('state', {})
                raw_features = info.get('features', {})
                env_reward = info.get('env_reward', 0.0)
                done = self.locals.get('dones', [False])[i] if 'dones' in self.locals else False
                try:
                    shaped = self._reward_fn(raw_state, raw_features, float(env_reward), done)
                    if 'rewards' in self.locals:
                        self.locals['rewards'][i] = shaped
                except Exception:
                    pass
            return True

    vec_env = make_train_vec_env(n_envs=4, seed=42)
    try:
        import tensorboard  # noqa
        tb_log = str(LOG_DIR / f"baseline_{label}")
    except ImportError:
        tb_log = None
    model = PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=tb_log, **PPO_CONFIG)

    print(f"  Training {label} for {train_steps:,} steps...")
    t0 = time.time()
    cb = ShapingCallback(reward_fn)
    model.learn(total_timesteps=train_steps, callback=cb, progress_bar=True)
    print(f"  Completed in {time.time()-t0:.1f}s")

    save_path = MODEL_DIR / f"baseline_{label}"
    model.save(str(save_path))
    print(f"  Model saved to {save_path}")

    vec_env.close()
    return model


def ppo_policy(model: PPO):
    """Return a policy function that uses a trained PPO model."""
    def _policy(obs: np.ndarray, env: WirelessAntiJammingEnv) -> int:
        action, _ = model.predict(obs[np.newaxis], deterministic=True)
        return int(action[0])
    return _policy


# ---------------------------------------------------------------------------
# Result printer
# ---------------------------------------------------------------------------

def print_results(label: str, metrics: dict):
    print(f"\n  {label}:")
    print(f"    Avg PDR:          {metrics['avg_pdr']:.3f} ± {metrics['std_pdr']:.3f}")
    print(f"    Avg Throughput:   {metrics['avg_throughput']:.3f} pkts/step")
    print(f"    Avg Jammed Rate:  {metrics['avg_jammed_rate']:.3f}")
    print(f"    Avg Switch Rate:  {metrics['avg_switch_rate']:.3f}/step")
    print(f"    Good Episodes:    {metrics['good_ep_rate']:.1f}% (PDR >= 0.75)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baseline(mode: str, episodes: int, train_steps: int):
    print(f"\n{'='*60}")
    print(f"Baseline: {mode.upper()}")
    print(f"{'='*60}")

    if mode == "random":
        metrics = run_episodes(random_policy, n_episodes=episodes)
        print_results("Random Channel Selection", metrics)
        return metrics

    elif mode == "heuristic":
        metrics = run_episodes(heuristic_policy, n_episodes=episodes)
        print_results("Heuristic (Avoid Last Jammed)", metrics)
        return metrics

    elif mode == "ppo-sparse":
        model = train_ppo(reward_fn=None, train_steps=train_steps, label="ppo_sparse")
        metrics = run_episodes(ppo_policy(model), n_episodes=episodes)
        print_results("PPO — Sparse Reward Only", metrics)
        return metrics

    elif mode == "ppo-shaped":
        model = train_ppo(reward_fn=_manual_shaped_reward, train_steps=train_steps, label="ppo_shaped")
        metrics = run_episodes(ppo_policy(model), n_episodes=episodes)
        print_results("PPO — Manually Shaped Reward", metrics)
        return metrics

    elif mode == "all":
        results = {}

        print("\nRunning all baselines...")
        results["Random"] = run_episodes(random_policy, n_episodes=episodes)
        print_results("Random", results["Random"])

        results["Heuristic"] = run_episodes(heuristic_policy, n_episodes=episodes)
        print_results("Heuristic", results["Heuristic"])

        model_sparse = train_ppo(reward_fn=None, train_steps=train_steps, label="ppo_sparse")
        results["PPO-Sparse"] = run_episodes(ppo_policy(model_sparse), n_episodes=episodes)
        print_results("PPO-Sparse", results["PPO-Sparse"])

        model_shaped = train_ppo(reward_fn=_manual_shaped_reward, train_steps=train_steps, label="ppo_shaped")
        results["PPO-Shaped"] = run_episodes(ppo_policy(model_shaped), n_episodes=episodes)
        print_results("PPO-Shaped", results["PPO-Shaped"])

        # Summary table
        print(f"\n{'='*75}")
        print("BASELINE COMPARISON SUMMARY")
        print(f"{'='*75}")
        print(f"{'Method':<25} {'Avg PDR':>9} {'Jammed%':>9} {'Throughput':>11} {'Good Eps%':>10}")
        print("-" * 75)
        for name, m in results.items():
            print(f"{name:<25} {m['avg_pdr']:>9.3f} {m['avg_jammed_rate']:>9.3f} "
                  f"{m['avg_throughput']:>11.3f} {m['good_ep_rate']:>10.1f}%")
        print("(PPO+LLM results from train_iterative.py run separately)")

        return results

    else:
        print(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluations for anti-jamming channel selection"
    )
    parser.add_argument('--mode', type=str, default='all',
                        choices=['random', 'heuristic', 'ppo-sparse', 'ppo-shaped', 'all'],
                        help='Which baseline to run (default: all)')
    parser.add_argument('--episodes', type=int, default=30,
                        help='Number of evaluation episodes (default: 30)')
    parser.add_argument('--train-steps', type=int, default=100000,
                        help='Training steps for PPO baselines (default: 100000)')

    args = parser.parse_args()
    run_baseline(args.mode, args.episodes, args.train_steps)


if __name__ == "__main__":
    main()
