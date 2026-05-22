"""
Wireless State Extraction and Feature Computation
==================================================
Replaces the Pong/ALE RAM extractor.

State comes from the WirelessAntiJammingEnv's internal state dict
(passed via info["state"] and info["features"] on each step).
This module provides:
  - extract_wireless_state()  — thin pass-through / validation
  - compute_wireless_features() — derived features for reward shaping
  - state_to_string()         — human-readable string for LLM prompts
"""

import numpy as np
from typing import Dict, Optional


def extract_wireless_state(env_state: Dict) -> Dict:
    """
    Validate and return the wireless environment state dict.

    The env already provides a rich state dict via info["state"].
    This function acts as a typed pass-through and fills in defaults
    for any missing keys so downstream code never KeyErrors.

    Args:
        env_state: dict from info["state"] in WirelessAntiJammingEnv

    Returns:
        Validated state dict with guaranteed keys
    """
    n = env_state.get("n_channels", 8)
    return {
        "channel_snr":           np.asarray(env_state.get("channel_snr", np.zeros(n))),
        "channel_success_rates": np.asarray(env_state.get("channel_success_rates", np.full(n, 0.5))),
        "jammed_decay":          np.asarray(env_state.get("jammed_decay", np.zeros(n))),
        "prev_channel":          int(env_state.get("prev_channel", -1)),
        "queue_length":          int(env_state.get("queue_length", 0)),
        "energy_used":           float(env_state.get("energy_used", 0.0)),
        "jammer_mode":           str(env_state.get("jammer_mode", "unknown")),
        "step":                  int(env_state.get("step", 0)),
        "max_steps":             int(env_state.get("max_steps", 500)),
        "n_channels":            n,
        "ep_delivered":          int(env_state.get("ep_delivered", 0)),
        "ep_total_tx":           int(env_state.get("ep_total_tx", 0)),
        "ep_jammed_tx":          int(env_state.get("ep_jammed_tx", 0)),
        "ep_switches":           int(env_state.get("ep_switches", 0)),
        "ep_energy":             float(env_state.get("ep_energy", 0.0)),
    }


def compute_wireless_features(state: Dict, prev_state: Optional[Dict] = None) -> Dict:
    """
    Compute derived features from current (and optionally previous) state.

    These features are passed to compute_reward() alongside the raw state dict,
    giving the LLM-generated reward function clean booleans and scalars to work with.

    Args:
        state:      Current wireless state dict (from extract_wireless_state)
        prev_state: Previous step's state dict (optional)

    Returns:
        Dict of derived features
    """
    features: Dict = {}

    if not state:
        return features

    n = state.get("n_channels", 8)
    success_rates = state.get("channel_success_rates", np.full(n, 0.5))
    jammed_decay  = state.get("jammed_decay", np.zeros(n))
    snr           = state.get("channel_snr", np.zeros(n))
    prev_ch       = state.get("prev_channel", -1)
    queue         = state.get("queue_length", 0)
    max_steps     = state.get("max_steps", 500)
    step          = state.get("step", 0)

    # Best channel by rolling success rate
    best_ch = int(np.argmax(success_rates))
    features["best_channel"] = best_ch
    features["best_channel_success_rate"] = float(success_rates[best_ch])

    # Worst (most jammed) channel
    features["worst_channel"] = int(np.argmax(jammed_decay))
    features["worst_jammed_decay"] = float(np.max(jammed_decay))

    # Current channel selected (same as prev_ch since step already advanced)
    features["current_channel"] = prev_ch

    # Did agent switch channel from last step?
    if prev_state is not None:
        features["switched"] = bool(prev_ch != prev_state.get("prev_channel", -1))
    else:
        features["switched"] = False

    # Is the current channel heavily jammed? (decay > 0.5 means recently jammed)
    if 0 <= prev_ch < n:
        features["current_ch_jammed"] = bool(jammed_decay[prev_ch] > 0.5)
        features["current_ch_snr"] = float(snr[prev_ch])
        features["current_ch_success_rate"] = float(success_rates[prev_ch])
    else:
        features["current_ch_jammed"] = False
        features["current_ch_snr"] = float(np.mean(snr))
        features["current_ch_success_rate"] = 0.5

    # Queue pressure: how full is the queue?
    queue_cap = 20
    features["queue_pressure"] = float(queue / queue_cap)
    features["queue_critical"] = bool(queue > queue_cap * 0.8)

    # Running delivery ratio
    ep_tx = max(state.get("ep_total_tx", 0), 1)
    features["running_pdr"] = float(state.get("ep_delivered", 0) / ep_tx)
    features["running_jammed_rate"] = float(state.get("ep_jammed_tx", 0) / ep_tx)

    # Adaptation signal: is agent on the best available channel?
    features["on_best_channel"] = bool(prev_ch == best_ch)

    # Energy budget progress
    ep_energy = state.get("ep_energy", 0.0)
    features["energy_budget_used"] = float(
        ep_energy / max(step * 1.0, 1.0)  # energy per step
    )

    # Step fraction (how far into the episode)
    features["step_fraction"] = float(step / max(max_steps, 1))

    # Throughput trend vs previous state
    if prev_state is not None:
        prev_pdr_num = prev_state.get("ep_delivered", 0)
        prev_pdr_den = max(prev_state.get("ep_total_tx", 1), 1)
        prev_pdr = prev_pdr_num / prev_pdr_den
        features["pdr_improving"] = bool(features["running_pdr"] > prev_pdr)
    else:
        features["pdr_improving"] = False

    return features


def state_to_string(state: Dict, features: Dict) -> str:
    """Convert wireless state and features to human-readable string for LLM."""
    if not state:
        return "No state available"

    n = state.get("n_channels", 8)
    snr = state.get("channel_snr", np.zeros(n))
    sr  = state.get("channel_success_rates", np.full(n, 0.5))
    jd  = state.get("jammed_decay", np.zeros(n))

    lines = [
        f"Step: {state.get('step', 0)} / {state.get('max_steps', 500)}",
        f"Jammer Mode: {state.get('jammer_mode', 'unknown')}",
        f"Queue Length: {state.get('queue_length', 0)}",
        f"Energy Used: {state.get('energy_used', 0.0):.1f}",
        f"Prev Channel: {state.get('prev_channel', -1)}",
        "",
        "Per-channel status (SNR dB | success% | jammed_decay):",
    ]
    for c in range(n):
        lines.append(
            f"  Ch{c}: SNR={snr[c]:.1f} | succ={sr[c]:.2f} | jammed={jd[c]:.2f}"
        )
    lines += [
        "",
        f"Best Channel: {features.get('best_channel', '?')} "
        f"(success={features.get('best_channel_success_rate', 0):.2f})",
        f"Current Channel Jammed: {features.get('current_ch_jammed', '?')}",
        f"Queue Pressure: {features.get('queue_pressure', 0):.2f}",
        f"Running PDR: {features.get('running_pdr', 0):.3f}",
        f"Running Jammed Rate: {features.get('running_jammed_rate', 0):.3f}",
    ]
    return "\n".join(lines)
