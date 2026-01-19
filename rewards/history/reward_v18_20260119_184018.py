"""
Reward Function v18
Generated: 2026-01-19T18:40:18.778321

REASONING:
The agent is performing well and showing consistent dominance. However, there are still opportunities to enhance its behavior by providing additional incentives for precise paddle control and timely movements. Specifically, we can add a small bonus when the paddle stays aligned with the ball and when it moves toward the ball just as the ball approaches.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: 18.30
- Best Score: 20
- Worst Score: 10
- Win Rate: 100.0%
- Average Episode Length: 1903 steps

### Ball Tracking
- Average Paddle-Ball Distance: 31.1 pixels
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
    if features["ball_approaching"] and abs(features["paddle_ball_distance_y"]) <= 10:
        reward += 0.05  # Small bonus for being perfectly aligned
    
    # Reward for moving toward the ball when it approaches
    if features["ball_approaching"] and features["moving_toward_ball"]:
        reward += 0.03  # Additional small reward for moving toward the ball
    
    return reward