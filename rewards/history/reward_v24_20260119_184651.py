"""
Reward Function v24
Generated: 2026-01-19T18:46:51.660538

REASONING:
The agent is dominating with a high average score and win rate, but we can still improve the agent's behavior by adding more subtle penalties for unnecessary movements when the ball is far away. This will encourage the agent to move only when necessary, which should help maintain the high level of performance.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: 18.80
- Best Score: 21
- Worst Score: 13
- Win Rate: 100.0%
- Average Episode Length: 1912 steps

### Ball Tracking
- Average Paddle-Ball Distance: 30.5 pixels
- Hit Rate: 90.1%
- Total Hits: 200
- Total Misses: 22

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