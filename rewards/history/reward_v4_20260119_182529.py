"""
Reward Function v4
Generated: 2026-01-19T18:25:29.999408

REASONING:
The agent is losing badly (score < -15) and the paddle is often far from the ball, indicating that it's not tracking the ball effectively. To improve this, I will add a small penalty for being far from the ball to encourage the agent to stay closer to the ball.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: -19.70
- Best Score: -17
- Worst Score: -21
- Win Rate: 0.0%
- Average Episode Length: 1231 steps

### Ball Tracking
- Average Paddle-Ball Distance: 54.4 pixels
- Hit Rate: 6.1%
- Total Hits: 13
- Total Misses: 200

### Observed Behaviors
- ❌ Agent is losing badly (score < -15) - not tracking ball effectively
- 🟠 Paddle often far from ball (40-60px) - tracking too slow
- Miss Rate: Very high (>80%) - agent misses mo...
"""

def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base game reward
    
    # Penalty for being far from the ball
    distance = features.get("paddle_ball_distance_y", 50)
    if distance > 40:
        reward -= 0.01
    
    return reward