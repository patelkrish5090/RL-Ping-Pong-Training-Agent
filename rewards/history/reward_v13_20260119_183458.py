"""
Reward Function v13
Generated: 2026-01-19T18:34:58.263248

REASONING:
The agent is performing well but could improve by maintaining better alignment with the ball when it approaches. This will help in making more accurate predictions about the ball's movement and improve hit rates.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: 15.00
- Best Score: 20
- Worst Score: 9
- Win Rate: 100.0%
- Average Episode Length: 2282 steps

### Ball Tracking
- Average Paddle-Ball Distance: 30.2 pixels
- Hit Rate: 76.9%
- Total Hits: 200
- Total Misses: 60

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