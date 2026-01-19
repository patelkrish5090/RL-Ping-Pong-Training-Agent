"""
Reward Function v8
Generated: 2026-01-19T18:29:43.379669

REASONING:
The agent is losing badly and missing many balls because it is not effectively tracking the ball's Y position. To improve this, I will add a strong positive reward for when the paddle is aligned with the ball.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: -15.20
- Best Score: -7
- Worst Score: -19
- Win Rate: 0.0%
- Average Episode Length: 1697 steps

### Ball Tracking
- Average Paddle-Ball Distance: 44.1 pixels
- Hit Rate: 22.5%
- Total Hits: 58
- Total Misses: 200

### Observed Behaviors
- ❌ Agent is losing badly (score < -15) - not tracking ball effectively
- 🟠 Paddle often far from ball (40-60px) - tracking too slow
- Miss Rate: High (60-80%) - agent misses many ...
"""

def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base game reward
    
    # Penalty for being far from the ball
    distance = features.get("paddle_ball_distance_y", 50)
    if distance > 40:
        reward -= 0.1  # Stronger penalty
    
    # Reward for staying close to the ball's Y position
    alignment_reward = max(0, 1 - (distance / 60))
    reward += alignment_reward * 0.25  # Increased alignment reward
    
    # Penalty for missing the ball when it approaches
    if features["ball_approaching"] and distance > 30:
        reward -= 0.5  # Strong penalty for missing a close approach
    
    return reward