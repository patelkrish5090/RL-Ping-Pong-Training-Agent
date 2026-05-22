"""
Anti-Jamming Reward Function v0 — Aggressive Initial Baseline
=============================================================
This is the STARTING reward before any LLM refinement.

Key design decisions (learned from Colab run analysis):
  - Previous weak magnitudes (0.01–0.08) failed to stop excessive switching
  - Stay-bonus of +0.20 needed to compete with the agent's switching urge
  - Jammed-channel penalty of -0.25 on top of env's -1.0 for strong aversion
  - Switch penalty only applied when switching AWAY from a GOOD channel
    (so the agent still escapes bad channels freely)
"""


def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    """
    Anti-Jamming Reward v0 — Aggressive Initial Baseline
    Designed to push PDR from ~0.47 baseline above 0.70 in first 2-3 iterations.
    """
    reward = env_reward  # Base: +1.0 delivery / -1.0 jammed / -0.75 switch-disrupt / -0.5 SNR-fail

    prev_ch  = state.get("prev_channel", -1)
    jammed_d = state.get("jammed_decay", [0.0] * 8)
    succ_r   = state.get("channel_success_rates", [0.5] * 8)
    n_ch     = state.get("n_channels", 8)

    switched  = features.get("switched", False)
    ch_jammed = features.get("current_ch_jammed", False)
    ch_succ   = features.get("current_ch_success_rate", 0.5)

    # ---- [1] Strong jammed-channel penalty --------------------------------
    # Magnitude -0.25: on top of env_reward's -1.0, makes jammed TX cost -1.25
    # Previous runs used -0.05 — too weak, jammed rate stayed at 20-25%
    if ch_jammed:
        reward -= 0.25

    # ---- [2] Stay-on-good-channel bonus ------------------------------------
    # The root cause of excessive switching: switching away from a clean channel
    # had no cost. Now staying on a good channel earns +0.20.
    # This directly competes with the switch-impulse from the policy.
    if not switched and not ch_jammed and ch_succ > 0.60:
        reward += 0.20

    # ---- [3] Intelligent switching penalty --------------------------------
    # Only penalise switching away from a GOOD channel (not forced escape).
    # This preserves the agent's ability to escape jammed/bad channels.
    if switched and 0 <= prev_ch < n_ch:
        prev_jammed  = jammed_d[prev_ch] > 0.5
        prev_success = succ_r[prev_ch]
        if not prev_jammed and prev_success > 0.55:
            # Switched away from a working clean channel — unnecessary!
            reward -= 0.25
        else:
            # Escaped from a jammed or bad channel — small positive reinforcement
            reward += 0.05

    # ---- [4] Reactive jammer escape bonus --------------------------------
    # Reactive jammer targets the agent's last channel. Switching away from
    # a highly-jammed channel is exactly correct behaviour — reward it.
    jammer_mode = state.get("jammer_mode", "random")
    if switched and jammer_mode == "reactive":
        if 0 <= prev_ch < n_ch and jammed_d[prev_ch] > 0.7:
            reward += 0.12

    # ---- [5] Queue buildup penalty ----------------------------------------
    queue_pressure = features.get("queue_pressure", 0.0)
    if queue_pressure > 0.80:
        reward -= 0.15
    elif queue_pressure > 0.60:
        reward -= 0.07

    # ---- [6] Best-channel alignment bonus ---------------------------------
    # Small extra reward for being on the empirically best channel
    if features.get("on_best_channel", False) and not ch_jammed:
        reward += 0.05

    return reward
