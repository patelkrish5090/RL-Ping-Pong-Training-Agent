"""
Reward Function v15
Generated: 2026-01-19T18:37:07.596464

REASONING:
The agent is dominating but occasionally missing the ball when it approaches. To improve this, I will add a small penalty for being far from the ball when it's approaching, as well as a small reward for moving toward the ball when it does approach.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: 16.70
- Best Score: 21
- Worst Score: 5
- Win Rate: 100.0%
- Average Episode Length: 2128 steps

### Ball Tracking
- Average Paddle-Ball Distance: 29.7 pixels
- Hit Rate: 82.3%
- Total Hits: 200
- Total Misses: 43

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
    
    return reward