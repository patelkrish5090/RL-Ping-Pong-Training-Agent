"""
Reward Function v9
Generated: 2026-01-19T18:30:46.554647

REASONING:
The agent is losing because it frequently misses the ball when approaching, and its alignment with the ball is inconsistent. To improve performance, I will add a strong penalty for missing the ball when it approaches and a small reward for staying close to the ball's Y position to help it anticipate the ball's movement more effectively.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: -11.80
- Best Score: 14
- Worst Score: -21
- Win Rate: 10.0%
- Average Episode Length: 2277 steps

### Ball Tracking
- Average Paddle-Ball Distance: 36.4 pixels
- Hit Rate: 29.2%
- Total Hits: 77
- Total Misses: 187

### Observed Behaviors
- ⚠️ Agent loses more than wins - tracking is inconsistent
- 🟡 Paddle moderately close (20-40px) - can improve anticipation
- Miss Rate: High (60-80%) - agent misses many balls
...
"""

def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base game reward
    
    # Penalty for missing the ball when it approaches
    if features["ball_approaching"] and distance > 30:
        reward -= 1.0  # Strong penalty for missing a close approach
    
    # Reward for staying close to the ball's Y position
    alignment_reward = max(0, 1 - (distance / 60))
    reward += alignment_reward * 0.15  # Increased alignment reward
    
    return reward