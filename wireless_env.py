"""
Tactical Wireless Anti-Jamming Channel Selection Environment
============================================================
A lightweight Gymnasium-compatible environment for MILCOM research on
LLM-guided reward shaping applied to anti-jamming channel selection in
tactical wireless networks.

Environment Overview:
- N discrete channels, each with time-varying SNR
- A jammer attacks one or more channels per timestep
- Agent selects a channel; packet succeeds if channel is not jammed AND SNR is sufficient
- Jammer switches strategies (modes) periodically to test adaptation
- Full episode metrics tracked: PDR, throughput, jammed_tx_rate, switches, energy, queue
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, List


# ---------------------------------------------------------------------------
# Jammer implementations
# ---------------------------------------------------------------------------

class RandomJammer:
    """Randomly jams K channels each step."""
    name = "random"

    def __init__(self, n_channels: int, n_jammed: int = 1, rng: np.random.Generator = None):
        self.n_channels = n_channels
        self.n_jammed = n_jammed
        self.rng = rng or np.random.default_rng()

    def get_jammed_channels(self, step: int, last_agent_channel: int) -> List[int]:
        return list(self.rng.choice(self.n_channels, size=self.n_jammed, replace=False))


class SweepJammer:
    """Sweeps through channels sequentially, jamming a contiguous band."""
    name = "sweep"

    def __init__(
        self,
        n_channels: int,
        sweep_speed: int = 1,
        sweep_width: int = 1,
        rng: np.random.Generator = None,
    ):
        self.n_channels = n_channels
        self.sweep_speed = sweep_speed
        self.sweep_width = sweep_width
        self.rng = rng or np.random.default_rng()

    def get_jammed_channels(self, step: int, last_agent_channel: int) -> List[int]:
        pos = (step // max(1, self.sweep_speed)) % self.n_channels
        return [(pos + offset) % self.n_channels for offset in range(self.sweep_width)]


class ReactiveJammer:
    """Reactively jams the agent's last used channel."""
    name = "reactive"

    def __init__(self, n_channels: int, reaction_prob: float = 0.85, n_random: int = 1,
                 rng: np.random.Generator = None):
        self.n_channels = n_channels
        self.reaction_prob = reaction_prob
        self.n_random = n_random
        self.rng = rng or np.random.default_rng()

    def get_jammed_channels(self, step: int, last_agent_channel: int) -> List[int]:
        jammed = []
        if last_agent_channel >= 0 and self.rng.random() < self.reaction_prob:
            jammed.append(last_agent_channel)

        remaining = [c for c in range(self.n_channels) if c not in jammed]
        if remaining and self.n_random > 0:
            sample_size = min(self.n_random, len(remaining))
            jammed.extend(list(self.rng.choice(remaining, size=sample_size, replace=False)))

        if not jammed:
            jammed = list(self.rng.choice(self.n_channels, size=1, replace=False))
        return sorted(set(int(c) for c in jammed))


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class WirelessAntiJammingEnv(gym.Env):
    """
    Tactical wireless anti-jamming channel selection environment.

    Observation space (flat Box, 3N + 5 floats):
        [0  : N]      channel_success_rate[N]  — rolling packet success rate per channel
        [N  : 2N]     channel_snr[N]           — estimated SNR per channel (normalized 0-1)
        [2N : 3N]     jammed_indicator[N]      — recent jamming detection (exponential decay)
        [3N]          prev_channel_norm        — previous channel (normalized 0-1)
        [3N+1]        queue_length_norm        — normalized queue length
        [3N+2]        energy_norm              — normalized cumulative energy used
        [3N+3]        jammer_mode_enc          — estimated jammer mode (0=random,0.5=sweep,1=reactive)
        [3N+4]        step_fraction            — step / max_steps

    Action space:
        Discrete(N_CHANNELS) — select a channel index

    Reward (sparse base, LLM-shapeable):
        +1.0  packet delivered (not jammed, SNR ok)
        -1.0  transmission failed (jammed or low SNR)
        Shaped additions injected via compute_reward()
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_channels: int = 8,
        max_steps: int = 500,
        jammer_modes: Optional[List[str]] = None,
        jammer_change_interval: int = 100,
        n_jammed_channels: int = 2,
        sweep_width: int = 2,
        reactive_jam_prob: float = 0.95,
        reactive_extra_random: int = 1,
        snr_mean: float = 8.0,         # dB mean SNR
        snr_std: float = 4.0,          # SNR fluctuation std dev
        snr_threshold: float = 7.0,    # Min SNR for successful TX (dB)
        queue_capacity: int = 20,       # Max queue length
        arrival_rate: float = 0.35,     # Packet arrival probability per slot
        max_packet_arrivals: int = 2,   # Arrival slots per step
        energy_per_tx: float = 1.0,     # Energy cost per transmission attempt
        switch_disruption_prob: float = 0.15,
        switch_energy_cost: float = 0.25,
        switch_reward_penalty: float = 0.05,
        snr_history_len: int = 10,      # Steps for rolling SNR/success window
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.max_steps = max_steps
        self.jammer_change_interval = jammer_change_interval
        self.n_jammed_channels = n_jammed_channels
        self.sweep_width = sweep_width
        self.reactive_jam_prob = reactive_jam_prob
        self.reactive_extra_random = reactive_extra_random
        self.snr_mean = snr_mean
        self.snr_std = snr_std
        self.snr_threshold = snr_threshold
        self.queue_capacity = queue_capacity
        self.arrival_rate = arrival_rate
        self.max_packet_arrivals = max_packet_arrivals
        self.energy_per_tx = energy_per_tx
        self.switch_disruption_prob = switch_disruption_prob
        self.switch_energy_cost = switch_energy_cost
        self.switch_reward_penalty = switch_reward_penalty
        self.snr_history_len = snr_history_len

        self.jammer_modes = jammer_modes or ["random", "sweep", "reactive"]

        # Spaces
        obs_dim = 3 * n_channels + 5  # snr + success + jammed + prev_ch + queue + energy + mode + step
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_channels)

        # RNG
        self._np_rng = np.random.default_rng(seed)

        # Will be initialized in reset()
        self._step = 0
        self._channel_snr = np.zeros(n_channels)
        self._jammer = None
        self._jammer_idx = 0
        self._prev_channel = -1
        self._queue_len = 0
        self._energy_used = 0.0

        # Rolling histories
        self._success_history = [[] for _ in range(n_channels)]   # per-channel success lists
        self._jammed_decay = np.zeros(n_channels)                  # decaying jammed indicator

        # Episode-level metric accumulators
        self._ep_delivered = 0
        self._ep_total_tx = 0
        self._ep_jammed_tx = 0
        self._ep_switches = 0
        self._ep_energy = 0.0
        self._ep_queue_sum = 0
        self._ep_arrivals = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_jammer(self, mode_name: str):
        if mode_name == "random":
            return RandomJammer(self.n_channels, n_jammed=self.n_jammed_channels, rng=self._np_rng)
        elif mode_name == "sweep":
            return SweepJammer(
                self.n_channels,
                sweep_speed=5,
                sweep_width=self.sweep_width,
                rng=self._np_rng,
            )
        elif mode_name == "reactive":
            return ReactiveJammer(
                self.n_channels,
                reaction_prob=self.reactive_jam_prob,
                n_random=self.reactive_extra_random,
                rng=self._np_rng,
            )
        else:
            return RandomJammer(self.n_channels, rng=self._np_rng)

    def _update_snr(self):
        """Update per-channel SNR using Rayleigh-like fading (Gaussian approximation)."""
        delta = self._np_rng.normal(0, self.snr_std * 0.3, size=self.n_channels)
        self._channel_snr = np.clip(
            self._channel_snr + delta,
            self.snr_mean - 3 * self.snr_std,
            self._channel_snr + 2 * self.snr_std  # Allow occasional good channels
        )
        # Clamp absolute range
        self._channel_snr = np.clip(self._channel_snr, 0.0, self.snr_mean + 4 * self.snr_std)

    def _channel_success_prob(self, ch: int) -> float:
        """Probability of a packet succeeding based on SNR (sigmoid model)."""
        snr = self._channel_snr[ch]
        # Sigmoid: p ~ 1 / (1 + exp(-(snr - threshold)))
        margin = snr - self.snr_threshold
        return float(1.0 / (1.0 + np.exp(-margin)))

    def _get_rolling_success(self, ch: int) -> float:
        """Rolling packet success rate for channel ch over recent history."""
        hist = self._success_history[ch]
        if not hist:
            return 0.5  # Prior: 50% unknown
        recent = hist[-self.snr_history_len:]
        return float(np.mean(recent))

    def _build_observation(self) -> np.ndarray:
        """Construct the flat observation vector."""
        N = self.n_channels

        # (1) Rolling success rate per channel [N]
        success_rates = np.array(
            [self._get_rolling_success(c) for c in range(N)], dtype=np.float32
        )

        # (2) Normalized SNR per channel [N] — map to [0,1]
        snr_max = self.snr_mean + 4 * self.snr_std
        snr_norm = np.clip(self._channel_snr / snr_max, 0.0, 1.0).astype(np.float32)

        # (3) Jammed decay indicator per channel [N]
        jammed_ind = self._jammed_decay.astype(np.float32)

        # (4) Previous channel (normalized)
        prev_ch_norm = np.float32(
            (self._prev_channel + 1) / (N + 1)  # -1 → 0, 0..N-1 → positive
        )

        # (5) Queue length normalized
        queue_norm = np.float32(self._queue_len / self.queue_capacity)

        # (6) Energy used normalized
        energy_norm = np.float32(
            min(self._energy_used / (self.max_steps * self.energy_per_tx), 1.0)
        )

        # (7) Jammer mode encoding (0=random, 0.5=sweep, 1.0=reactive)
        mode_name = getattr(self._jammer, "name", "random")
        mode_enc = {"random": 0.0, "sweep": 0.5, "reactive": 1.0}.get(mode_name, 0.0)
        mode_obs = np.float32(mode_enc)  # Full visibility for research; can be zeroed for harder task

        # (8) Step fraction
        step_frac = np.float32(self._step / self.max_steps)

        obs = np.concatenate([
            success_rates,     # N
            snr_norm,          # N
            jammed_ind,        # N
            [prev_ch_norm],    # 1
            [queue_norm],      # 1
            [energy_norm],     # 1
            [mode_obs],        # 1
            [step_frac],       # 1
        ])
        return obs.astype(np.float32)

    def _get_state_dict(self) -> Dict:
        """Return full internal state as a dict (used by reward/LLM interface)."""
        return {
            "channel_snr": self._channel_snr.copy(),
            "channel_success_rates": np.array(
                [self._get_rolling_success(c) for c in range(self.n_channels)]
            ),
            "jammed_decay": self._jammed_decay.copy(),
            "prev_channel": self._prev_channel,
            "queue_length": self._queue_len,
            "energy_used": self._energy_used,
            "jammer_mode": getattr(self._jammer, "name", "unknown"),
            "step": self._step,
            "max_steps": self.max_steps,
            "n_channels": self.n_channels,
            # Episode running totals
            "ep_delivered": self._ep_delivered,
            "ep_total_tx": self._ep_total_tx,
            "ep_jammed_tx": self._ep_jammed_tx,
            "ep_switches": self._ep_switches,
            "ep_energy": self._ep_energy,
        }

    def _get_episode_summary(self) -> Dict:
        """Compute episode-level metrics for the analyzer."""
        n_tx = max(self._ep_total_tx, 1)
        return {
            "pdr": self._ep_delivered / n_tx,
            "throughput": self._ep_delivered / max(self._step, 1),
            "jammed_rate": self._ep_jammed_tx / n_tx,
            "avg_queue": self._ep_queue_sum / max(self._step, 1),
            "switches": self._ep_switches,
            "switch_rate": self._ep_switches / max(self._step, 1),
            "energy": self._ep_energy,
            "total_delivered": self._ep_delivered,
            "total_tx": self._ep_total_tx,
            "total_steps": self._step,
            "arrivals": self._ep_arrivals,
        }

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        self._step = 0

        # Initialize SNR channels
        self._channel_snr = self._np_rng.normal(
            self.snr_mean, self.snr_std, size=self.n_channels
        ).clip(0.0, None)

        # Start with a random jammer mode
        self._jammer_idx = int(self._np_rng.integers(len(self.jammer_modes)))
        self._jammer = self._make_jammer(self.jammer_modes[self._jammer_idx])

        self._prev_channel = -1
        self._queue_len = 0
        self._energy_used = 0.0
        self._jammed_decay = np.zeros(self.n_channels)
        self._success_history = [[] for _ in range(self.n_channels)]

        # Reset episode metrics
        self._ep_delivered = 0
        self._ep_total_tx = 0
        self._ep_jammed_tx = 0
        self._ep_switches = 0
        self._ep_energy = 0.0
        self._ep_queue_sum = 0
        self._ep_arrivals = 0

        obs = self._build_observation()
        info = {"state": self._get_state_dict()}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep.

        Args:
            action: Channel index to select (0..N-1)

        Returns:
            obs, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        self._step += 1
        channel = int(action)

        # 1. Packet arrivals this step
        n_arrivals = int(self._np_rng.binomial(self.max_packet_arrivals, self.arrival_rate))
        self._queue_len = min(self._queue_len + n_arrivals, self.queue_capacity)
        self._ep_arrivals += n_arrivals

        # 2. Update channel SNR (fading)
        self._update_snr()

        # 3. Possibly change jammer mode
        if self._step % self.jammer_change_interval == 0:
            old_idx = self._jammer_idx
            available = list(range(len(self.jammer_modes)))
            if len(available) > 1:
                available.remove(old_idx)
            self._jammer_idx = int(self._np_rng.choice(available))
            self._jammer = self._make_jammer(self.jammer_modes[self._jammer_idx])

        # 4. Jammer acts
        jammed_channels = self._jammer.get_jammed_channels(self._step, self._prev_channel)

        # 5. Update jammed decay for each channel
        self._jammed_decay *= 0.75  # Exponential decay
        for jc in jammed_channels:
            self._jammed_decay[jc] = 1.0  # Fresh jamming signal

        # 6. Track channel switch
        switched = (self._prev_channel >= 0 and channel != self._prev_channel)
        if switched:
            self._ep_switches += 1

        # 7. Attempt packet transmission
        is_jammed = channel in jammed_channels
        snr_ok = self._np_rng.random() < self._channel_success_prob(channel)
        switch_disrupted = switched and (self._np_rng.random() < self.switch_disruption_prob)

        tx_success = (not is_jammed) and snr_ok and (not switch_disrupted) and (self._queue_len > 0)

        # Update success history for selected channel
        self._success_history[channel].append(1.0 if tx_success else 0.0)

        # 8. Queue drain
        if tx_success:
            packets_sent = min(self._queue_len, 1)
            self._queue_len -= packets_sent
            self._ep_delivered += packets_sent
        else:
            packets_sent = 0

        # 9. Energy cost (transmit attempt always costs energy)
        energy_cost = self.energy_per_tx + (self.switch_energy_cost if switched else 0.0)
        self._energy_used += energy_cost
        self._ep_energy += energy_cost

        # 10. Transmission accounting
        if self._queue_len > 0 or tx_success:
            self._ep_total_tx += 1
        if is_jammed:
            self._ep_jammed_tx += 1

        # 11. Queue accumulation tracking
        self._ep_queue_sum += self._queue_len

        # 12. Base sparse reward
        if tx_success:
            env_reward = 1.0
        elif is_jammed:
            env_reward = -1.0
        elif switch_disrupted:
            env_reward = -0.75
        else:
            # Failed due to SNR — softer penalty
            env_reward = -0.5

        if switched:
            env_reward -= self.switch_reward_penalty

        # Update prev channel
        self._prev_channel = channel

        # 13. Build observation and info
        obs = self._build_observation()
        state = self._get_state_dict()
        features = {
            "switched": switched,
            "is_jammed": is_jammed,
            "snr_ok": snr_ok,
            "switch_disrupted": switch_disrupted,
            "tx_success": tx_success,
            "jammed_channels": jammed_channels,
            "channel_snr": float(self._channel_snr[channel]),
            "queue_pressure": float(self._queue_len / self.queue_capacity),
            "best_channel": int(np.argmax([self._get_rolling_success(c) for c in range(self.n_channels)])),
            "avoided_jammed": channel not in jammed_channels,
            "switch_penalty_active": switched,
            "throughput_so_far": self._ep_delivered / max(self._step, 1),
        }

        terminated = False
        truncated = self._step >= self.max_steps

        info = {
            "state": state,
            "features": features,
            "env_reward": env_reward,
            "jammed_channels": jammed_channels,
            "jammer_mode": getattr(self._jammer, "name", "unknown"),
            "is_jammed": is_jammed,
            "switch_disrupted": switch_disrupted,
            "tx_success": tx_success,
        }
        if truncated:
            info["episode_summary"] = self._get_episode_summary()

        return obs, env_reward, terminated, truncated, info

    def get_episode_summary(self) -> Dict:
        """Call at end of episode to get metrics dict."""
        return self._get_episode_summary()

    def render(self):
        """Simple text render of current state."""
        N = self.n_channels
        jammed_now = self._jammer.get_jammed_channels(self._step, self._prev_channel)
        print(f"\nStep {self._step}/{self.max_steps} | Jammer: {getattr(self._jammer, 'name', '?')}")
        print(f"Queue: {self._queue_len}/{self.queue_capacity} | Energy: {self._energy_used:.1f}")
        for c in range(N):
            jmark = "🔴" if c in jammed_now else "🟢"
            sel = "◄" if c == self._prev_channel else " "
            print(f"  Ch{c}: SNR={self._channel_snr[c]:.1f}dB  "
                  f"Succ={self._get_rolling_success(c):.2f}  "
                  f"Jam={self._jammed_decay[c]:.2f}  {jmark}{sel}")

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = WirelessAntiJammingEnv(n_channels=8, max_steps=200, seed=42)
    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Obs space: {env.observation_space}")
    print(f"Initial obs:\n{obs}")

    total_reward = 0.0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    summary = env.get_episode_summary()
    print(f"\nEpisode complete — {step+1} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"PDR: {summary['pdr']:.3f}")
    print(f"Jammed TX rate: {summary['jammed_rate']:.3f}")
    print(f"Switches: {summary['switches']}")
    print(f"Avg queue: {summary['avg_queue']:.2f}")
