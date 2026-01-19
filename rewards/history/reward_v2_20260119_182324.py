"""
Reward Function v2
Generated: 2026-01-19T18:23:24.234224

REASONING:
The agent is losing because it does not track the ball effectively. I will add a larger alignment bonus and a penalty for being too far from the ball to encourage better tracking.

TRAINING SUMMARY:
## Training Results (Last 10 Episodes)

### Performance Metrics
- Average Score: -20.80
- Best Score: -19
- Worst Score: -21
- Win Rate: 0.0%
- Average Episode Length: 814 steps

### Ball Tracking
- Average Paddle-Ball Distance: 84.7 pixels
- Hit Rate: 1.0%
- Total Hits: 2
- Total Misses: 200

### Observed Behaviors
- ❌ Agent is losing badly (score < -15) - not tracking ball effectively
- 🔴 Paddle very far from ball (>60px) - needs fundamental tracking improvement
- Miss Rate: Very high (>80%) -...
"""

def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base game reward
    
    # === Reward Shaping ===
    
    # 1. Large alignment bonus for staying close to the ball (Y-axis distance)
    distance = features.get("paddle_ball_distance_y", 50)
    alignment_bonus = max(0, (100 - distance) / 100) * 0.2
    reward += alignment_bonus
    
    # 2. Small penalty for being far from the ball
    if distance > 80:
        reward -= 0.05  # Penalty for not tracking
    
    return reward