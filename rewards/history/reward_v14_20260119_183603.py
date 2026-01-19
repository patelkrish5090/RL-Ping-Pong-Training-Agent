"""
Reward Function v14
Generated: 2026-01-19T18:36:03.995674

REASONING:
The agent is demonstrating excellent performance with a high average score and win rate. However, we can fine-tune the reward function to further enhance its ability to anticipate the ball's movement and improve overall control.

To achieve this, we will:
1. Reward the agent for keeping the paddle close to the ball when it approaches.
2. Add a small penalty for unnecessary movements when the ball is far from the paddle.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: 17.40
- Best Score: 20
- Worst Score: 12
- Win Rate: 100.0%
- Average Episode Length: 2146 steps

### Ball Tracking
- Average Paddle-Ball Distance: 30.5 pixels
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
    
    # Reward for staying aligned with the ball's Y position
    if features["paddle_aligned"]:
        reward += 0.02  # Small reward for alignment
    
    # Reward for moving toward the ball when it approaches
    if features["ball_approaching"] and features["moving_toward_ball"]:
        reward += 0.03  # Additional small reward for moving toward the ball
    
    # Penalty for unnecessary movement when the ball is far
    if not features["ball_approaching"] and abs(features["paddle_ball_distance_y"]) > 50:
        reward -= 0.01  # Small penalty for unnecessary movement
    
    return reward