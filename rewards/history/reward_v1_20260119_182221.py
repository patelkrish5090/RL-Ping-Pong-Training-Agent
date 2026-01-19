"""
Reward Function v1
Generated: 2026-01-19T18:22:21.128540

REASONING:
The agent is losing badly because it is not effectively tracking the ball and misses most incoming shots. I will add a more significant reward for staying close to the ball's Y position (alignment) and a small penalty when the paddle is far from the ball.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: -20.40
- Best Score: -20
- Worst Score: -21
- Win Rate: 0.0%
- Average Episode Length: 886 steps

### Ball Tracking
- Average Paddle-Ball Distance: 107.9 pixels
- Hit Rate: 2.9%
- Total Hits: 6
- Total Misses: 200

### Observed Behaviors
- ❌ Agent is losing badly (score < -15) - not tracking ball effectively
- 🔴 Paddle very far from ball (>60px) - needs fundamental tracking improvement
- Miss Rate: Very high (>80%) ...
"""

def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base game reward
    
    # === Reward Shaping ===
    
    # 1. Reward staying aligned with ball (Y-axis distance)
    distance = features.get("paddle_ball_distance_y", 50)
    # Scale: 0 distance = +0.1, 100 distance = 0
    alignment_reward = max(0, (100 - distance) / 100) * 0.1
    reward += alignment_reward
    
    # 2. Small penalty for being far from ball
    if distance > 80:
        reward -= 0.01  # Penalty for not tracking
    
    return reward