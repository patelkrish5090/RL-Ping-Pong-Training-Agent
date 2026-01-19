"""
Reward Function v11
Generated: 2026-01-19T18:32:52.930152

REASONING:
The agent is scoring well but missing opportunities to increase score further. I will add a small penalty for being too far from the ball when it's approaching, and a reward for moving towards the ball when it's close.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: 10.90
- Best Score: 19
- Worst Score: -9
- Win Rate: 90.0%
- Average Episode Length: 2679 steps

### Ball Tracking
- Average Paddle-Ball Distance: 34.1 pixels
- Hit Rate: 67.8%
- Total Hits: 192
- Total Misses: 91

### Observed Behaviors
- ✓ Agent is winning - good ball tracking established
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
        reward += 0.03  # Small reward for moving toward a close approach
    
    return reward