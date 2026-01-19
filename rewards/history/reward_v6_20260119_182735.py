"""
Reward Function v6
Generated: 2026-01-19T18:27:35.905384

REASONING:
The agent is losing because it is not tracking the ball effectively and missing most incoming balls. I will add a stronger penalty for being far from the ball and a smaller reward for staying close to the ball's Y position to guide the agent towards better alignment.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: -16.70
- Best Score: -12
- Worst Score: -19
- Win Rate: 0.0%
- Average Episode Length: 1896 steps

### Ball Tracking
- Average Paddle-Ball Distance: 48.2 pixels
- Hit Rate: 17.7%
- Total Hits: 43
- Total Misses: 200

### Observed Behaviors
- ❌ Agent is losing badly (score < -15) - not tracking ball effectively
- 🟠 Paddle often far from ball (40-60px) - tracking too slow
- Miss Rate: Very high (>80%) - agent misses m...
"""

def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base game reward
    
    # Penalty for being far from the ball
    distance = features.get("paddle_ball_distance_y", 50)
    if distance > 40:
        reward -= 0.1  # Stronger penalty
    
    # Reward for staying close to the ball's Y position
    alignment_reward = max(0, 1 - (distance / 60))
    reward += alignment_reward * 0.02  # Smaller reward
    
    return reward