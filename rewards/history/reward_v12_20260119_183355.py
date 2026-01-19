"""
Reward Function v12
Generated: 2026-01-19T18:33:55.907862

REASONING:
The agent is performing well but can improve anticipation and positioning around the ball. I will add a small penalty for being far from the ball when it approaches, as well as a reward for staying aligned with the ball's Y position to better prepare for returns.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: 16.40
- Best Score: 21
- Worst Score: 9
- Win Rate: 100.0%
- Average Episode Length: 2205 steps

### Ball Tracking
- Average Paddle-Ball Distance: 34.1 pixels
- Hit Rate: 81.3%
- Total Hits: 200
- Total Misses: 46

### Observed Behaviors
- ✅ Agent is dominating - excellent performance
- 🟡 Paddle moderately close (20-40px) - can improve anticipation
- Hit Rate: Good (>60%) - successfully returns most balls
...
"""

def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base game reward
    
    # Penalty for missing the ball when it approaches
    if features["ball_approaching"] and abs(features["paddle_ball_distance_y"]) > 40:
        reward -= 0.05  # Small penalty for being far from a close approach
    
    # Reward for staying aligned with the ball's Y position
    if features["paddle_aligned"]:
        reward += 0.02  # Small reward for alignment
    
    return reward