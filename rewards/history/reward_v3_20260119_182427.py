"""
Reward Function v3
Generated: 2026-01-19T18:24:27.634813

REASONING:
The agent is losing because it's not effectively tracking the ball. To improve this, I will add a small penalty for being far from the ball and a slight bonus for staying closer to the ball. This should help guide the agent towards better ball tracking behavior.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: -19.50
- Best Score: -16
- Worst Score: -21
- Win Rate: 0.0%
- Average Episode Length: 1202 steps

### Ball Tracking
- Average Paddle-Ball Distance: 57.9 pixels
- Hit Rate: 7.0%
- Total Hits: 15
- Total Misses: 200

### Observed Behaviors
- ❌ Agent is losing badly (score < -15) - not tracking ball effectively
- 🟠 Paddle often far from ball (40-60px) - tracking too slow
- Miss Rate: Very high (>80%) - agent misses mo...
"""

def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base game reward
    
    # Small penalty for being far from the ball
    distance = features.get("paddle_ball_distance_y", 50)
    if distance > 80:
        reward -= 0.1
    
    # Alignment bonus for staying close to the ball (Y-axis distance)
    alignment_bonus = max(0, (100 - distance) / 100) * 0.2
    reward += alignment_bonus
    
    return reward