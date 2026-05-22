def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    """
    Initial shaped reward for the harder anti-jamming channel selection task.

    Base signal:
        env_reward = +1.0  packet delivered
        env_reward = -1.0  transmission jammed
        env_reward = -0.75 switch disruption
        env_reward = -0.5  failed due to low SNR

    Shaping terms stay small so the true communication objective remains dominant.
    """
    reward = env_reward

    # Penalize channels that still show strong recent jamming evidence.
    if features.get("current_ch_jammed", False):
        reward -= 0.05

    # Reward choosing the empirically best clean channel.
    if features.get("on_best_channel", False) and not features.get("current_ch_jammed", False):
        reward += 0.03

    # Switching already has base overhead in the environment; keep shaping modest.
    if features.get("switch_disrupted", False):
        reward -= 0.03
    elif features.get("switched", False):
        reward -= 0.01

    # Penalize critical queue buildup.
    queue_pressure = features.get("queue_pressure", 0.0)
    if queue_pressure > 0.8:
        reward -= 0.04
    elif queue_pressure > 0.5:
        reward -= 0.01

    return reward
