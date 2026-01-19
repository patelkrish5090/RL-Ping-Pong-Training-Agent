"""
Reward Function v10
Generated: 2026-01-19T18:31:49.984564

REASONING:
The agent is losing because it is missing many balls when they are close by. This indicates that the agent needs better anticipation of ball movements. I will add a small penalty for being far from the ball when it approaches and an additional reward for moving toward the ball when it is approaching.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: -12.00
- Best Score: -7
- Worst Score: -18
- Win Rate: 0.0%
- Average Episode Length: 2986 steps

### Ball Tracking
- Average Paddle-Ball Distance: 36.2 pixels
- Hit Rate: 31.0%
- Total Hits: 90
- Total Misses: 200

### Observed Behaviors
- ⚠️ Agent loses more than wins - tracking is inconsistent
- 🟡 Paddle moderately close (20-40px) - can improve anticipation
- Miss Rate: High (60-80%) - agent misses many balls
...
"""

def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base game reward
    
    # Penalty for missing the ball when it approaches
    if features["ball_approaching"] and abs(features["paddle_ball_distance_y"]) > 40:
        reward -= 0.10  # Small penalty for being far from a close approach
    
    # Reward for moving toward the ball when it approaches
    if features["ball_approaching"] and features["moving_toward_ball"]:
        reward += 0.05  # Small reward for moving toward a close approach
    
    return reward