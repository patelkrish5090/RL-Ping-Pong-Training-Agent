"""
Reward Function v16
Generated: 2026-01-19T18:38:11.421712

REASONING:
The agent is performing well but could benefit from additional incentives to stay aligned with the ball when it's approaching. This will help improve anticipation and reaction times.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: 16.80
- Best Score: 20
- Worst Score: 8
- Win Rate: 100.0%
- Average Episode Length: 2133 steps

### Ball Tracking
- Average Paddle-Ball Distance: 30.5 pixels
- Hit Rate: 82.6%
- Total Hits: 200
- Total Misses: 42

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