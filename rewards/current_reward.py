def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base game reward
    
    # Reward for staying aligned with the ball when it's approaching
    if features["ball_approaching"]:
        if abs(features["paddle_ball_distance_y"]) <= 10:
            reward += 0.25  # Slightly larger bonus for perfect alignment
        else:
            reward += max(0, -0.05 * (abs(features["paddle_ball_distance_y"]) - 10))  # Penalties for being far from the ball
    
    # Reward for moving toward the ball when it approaches
    if features["ball_approaching"] and features["moving_toward_ball"]:
        reward += 0.1  # Additional small reward for moving toward the ball
    
    # Small penalty for unnecessary movement when the ball is far
    if not features["ball_approaching"] and abs(features["paddle_ball_distance_y"]) > 40:
        reward -= 0.03  # Penalty for being too far from the ball when not needed
    
    return reward