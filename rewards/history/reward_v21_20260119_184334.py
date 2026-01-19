"""
Reward Function v21
Generated: 2026-01-19T18:43:34.527582

REASONING:
The agent is showing strong alignment and anticipation of the ball, but it could benefit from a more consistent reward for staying close to the ball while moving toward it. Additionally, a small penalty for unnecessary movement when the ball is far can help the agent be more selective in its movements.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: 18.30
- Best Score: 21
- Worst Score: 14
- Win Rate: 100.0%
- Average Episode Length: 2002 steps

### Ball Tracking
- Average Paddle-Ball Distance: 30.1 pixels
- Hit Rate: 88.1%
- Total Hits: 200
- Total Misses: 27

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