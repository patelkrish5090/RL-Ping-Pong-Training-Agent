"""
Reward Function v20
Generated: 2026-01-19T18:42:29.020793

REASONING:
The agent is demonstrating excellent performance but can still improve its anticipation of the ball's movement. A small penalty for being too far from the ball when it is approaching can help guide the agent to stay more aligned with the ball, which may lead to better returns.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: 18.60
- Best Score: 20
- Worst Score: 13
- Win Rate: 100.0%
- Average Episode Length: 2032 steps

### Ball Tracking
- Average Paddle-Ball Distance: 31.4 pixels
- Hit Rate: 89.3%
- Total Hits: 200
- Total Misses: 24

### Observed Behaviors
- ✅ Agent is dominating - excellent performance
- 🟡 Paddle moderately close (20-40px) - can improve anticipation
- Hit Rate: Good (>60%) - successfully returns most balls
...
"""

def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base game reward
    
    # Reward for staying aligned with the ball when it's approaching
    if features["ball_approaching"]:
        if abs(features["paddle_ball_distance_y"]) <= 10:
            reward += 0.15  # Slightly larger bonus for perfect alignment
        else:
            reward += max(0, -0.1 * (abs(features["paddle_ball_distance_y"]) - 10))  # Penalties for being far from the ball
    
    # Reward for moving toward the ball when it approaches
    if features["ball_approaching"] and features["moving_toward_ball"]:
        reward += 0.05  # Additional small reward for moving toward the ball
    
    # Small penalty for being too far from the ball when it's approaching
    if features["ball_approaching"] and abs(features["paddle_ball_distance_y"]) > 40:
        reward -= 0.03  # Penalty for being too far from the ball

    return reward