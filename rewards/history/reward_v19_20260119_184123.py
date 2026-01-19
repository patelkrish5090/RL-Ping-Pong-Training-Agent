"""
Reward Function v19
Generated: 2026-01-19T18:41:23.563213

REASONING:
The agent is performing well but could improve its anticipation of the ball's movement. Currently, the agent receives a bonus when it aligns perfectly with the ball and another small bonus when moving toward the ball as it approaches. To enhance this further, I will add a shaping reward for slightly deviating from perfect alignment and penalize unnecessary movement when the ball is far.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: 17.30
- Best Score: 21
- Worst Score: 10
- Win Rate: 100.0%
- Average Episode Length: 2157 steps

### Ball Tracking
- Average Paddle-Ball Distance: 30.8 pixels
- Hit Rate: 84.4%
- Total Hits: 200
- Total Misses: 37

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
    
    return reward