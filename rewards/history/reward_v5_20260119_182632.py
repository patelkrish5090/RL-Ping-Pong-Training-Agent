"""
Reward Function v5
Generated: 2026-01-19T18:26:32.854035

REASONING:
The agent is losing badly and missing most incoming balls because it is not effectively tracking the ball's Y position. I will add a shaping reward to encourage the agent to stay aligned with the ball's vertical position.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: -18.90
- Best Score: -17
- Worst Score: -21
- Win Rate: 0.0%
- Average Episode Length: 1223 steps

### Ball Tracking
- Average Paddle-Ball Distance: 60.5 pixels
- Hit Rate: 9.5%
- Total Hits: 21
- Total Misses: 200

### Observed Behaviors
- ❌ Agent is losing badly (score < -15) - not tracking ball effectively
- 🔴 Paddle very far from ball (>60px) - needs fundamental tracking improvement
- Miss Rate: Very high (>80%)...
"""

def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base game reward
    
    # Penalty for being far from the ball
    distance = features.get("paddle_ball_distance_y", 50)
    if distance > 40:
        reward -= 0.01
    
    # Reward for staying close to the ball's Y position
    alignment_reward = max(0, 1 - (distance / 60))
    reward += alignment_reward * 0.02
    
    return reward