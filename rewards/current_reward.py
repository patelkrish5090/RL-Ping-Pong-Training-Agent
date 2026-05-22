def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    """
    Initial shaped reward for anti-jamming channel selection.

    Base signal:
        env_reward = +1.0  packet delivered (not jammed, SNR OK)
        env_reward = -1.0  transmission jammed
        env_reward = -0.5  failed due to low SNR

    Shaping terms (all magnitudes in [0.001, 0.1] per MILCOM safety constraints):
        - Penalty for selecting a recently-jammed channel
        - Reward for selecting the empirically-best channel when it is clean
        - Small penalty for unnecessary channel switches
        - Penalty for critical queue buildup
    """
    reward = env_reward  # Preserve sparse base signal

    # (1) Penalty: agent picked a channel still hot from recent jamming
    if features.get("current_ch_jammed", False):
        reward -= 0.05

    # (2) Bonus: agent chose the channel with best recent success rate
    #     (only reward if that channel is actually clean)
    if features.get("on_best_channel", False) and not features.get("current_ch_jammed", False):
        reward += 0.03

    # (3) Switching penalty — discourage excessive hopping
    if features.get("switched", False):
        reward -= 0.01

    # (4) Queue pressure penalty — encourage timely delivery
    queue_pressure = features.get("queue_pressure", 0.0)
    if queue_pressure > 0.8:
        reward -= 0.04
    elif queue_pressure > 0.5:
        reward -= 0.01

    return reward