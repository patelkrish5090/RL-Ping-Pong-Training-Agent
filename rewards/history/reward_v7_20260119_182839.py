"""
Reward Function v7
Generated: 2026-01-19T18:28:39.513700

REASONING:
The agent is losing because it misses many balls due to inconsistent tracking and poor anticipation. To improve this, I will add a strong penalty for missing the ball when it approaches.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: -12.60
- Best Score: -8
- Worst Score: -20
- Win Rate: 0.0%
- Average Episode Length: 2701 steps

### Ball Tracking
- Average Paddle-Ball Distance: 37.4 pixels
- Hit Rate: 29.6%
- Total Hits: 84
- Total Misses: 200

### Observed Behaviors
- ⚠️ Agent loses more than wins - tracking is inconsistent
- 🟡 Paddle moderately close (20-40px) - can improve anticipation
- Miss Rate: High (60-80%) - agent misses many balls
...
"""

def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base game reward
    
    # Penalty for being far from the ball
    distance = features.get("paddle_ball_distance_y", 50)
    if distance > 40:
        reward -= 0.1  # Stronger penalty
    
    # Reward for staying close to the ball's Y position
    alignment_reward = max(0, 1 - (distance / 60))
    reward += alignment_reward * 0.02  # Smaller reward
    
    # Penalty for missing the ball when it approaches
    if features["ball_approaching"] and distance > 30:
        reward -= 0.5  # Strong penalty for missing a close approach
    
    return reward