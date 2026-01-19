"""
Reward Function v17
Generated: 2026-01-19T18:39:15.381354

REASONING:
The agent is performing well but can improve its anticipation by staying closer to the ball when it's approaching. This will help in reducing misses and increasing hits.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: 17.40
- Best Score: 20
- Worst Score: 13
- Win Rate: 100.0%
- Average Episode Length: 1947 steps

### Ball Tracking
- Average Paddle-Ball Distance: 31.0 pixels
- Hit Rate: 84.7%
- Total Hits: 200
- Total Misses: 36

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
    
    # Reward for moving toward the ball when it approaches
    if features["ball_approaching"] and features["moving_toward_ball"]:
        reward += 0.03  # Additional small reward for moving toward the ball
    
    # Reward for staying aligned with the ball when it's approaching
    if features["ball_approaching"] and abs(features["paddle_ball_distance_y"]) <= 10:
        reward += 0.02  # Small bonus for being aligned
    
    return reward