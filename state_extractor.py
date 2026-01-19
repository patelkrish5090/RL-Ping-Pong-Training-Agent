"""Extract game state from Pong RAM for reward computation"""
import numpy as np
from typing import Dict, Optional

# RAM addresses for Pong (verified from Atari 2600 documentation)
PONG_RAM = {
    "ball_x": 49,
    "ball_y": 54,
    "player_paddle_y": 51,
    "cpu_paddle_y": 50,
    "player_score": 14,
    "cpu_score": 13,
}

def extract_pong_state(ram: np.ndarray) -> Dict:
    """
    Extract game state from 128-byte RAM.
    
    Args:
        ram: 128-byte numpy array from env.unwrapped.ale.getRAM()
    
    Returns:
        Dictionary with game state
    """
    return {
        "ball_x": int(ram[PONG_RAM["ball_x"]]),
        "ball_y": int(ram[PONG_RAM["ball_y"]]),
        "player_paddle_y": int(ram[PONG_RAM["player_paddle_y"]]),
        "cpu_paddle_y": int(ram[PONG_RAM["cpu_paddle_y"]]),
        "player_score": int(ram[PONG_RAM["player_score"]]),
        "cpu_score": int(ram[PONG_RAM["cpu_score"]]),
    }


def compute_derived_features(state: Dict, prev_state: Optional[Dict] = None) -> Dict:
    """
    Compute useful features for reward shaping.
    
    Args:
        state: Current game state
        prev_state: Previous game state (optional)
    
    Returns:
        Dictionary with derived features
    """
    features = {}
    
    if not state:
        return features
    
    ball_y = state.get("ball_y", 100)
    paddle_y = state.get("player_paddle_y", 100)
    
    # Distance between paddle and ball (Y-axis)
    features["paddle_ball_distance_y"] = abs(ball_y - paddle_y)
    
    # Is paddle aligned with ball? (within 10 pixels)
    features["paddle_aligned"] = features["paddle_ball_distance_y"] < 10
    
    # Ball position normalized (0-1)
    features["ball_x_normalized"] = state.get("ball_x", 80) / 160.0
    features["ball_y_normalized"] = ball_y / 210.0
    
    # Paddle position normalized
    features["paddle_y_normalized"] = paddle_y / 210.0
    
    # Score difference
    features["score_diff"] = state.get("player_score", 0) - state.get("cpu_score", 0)
    
    # Velocity features (if we have previous state)
    if prev_state is not None:
        prev_ball_x = prev_state.get("ball_x", state.get("ball_x", 80))
        prev_ball_y = prev_state.get("ball_y", ball_y)
        prev_paddle_y = prev_state.get("player_paddle_y", paddle_y)
        
        features["ball_velocity_x"] = state.get("ball_x", 80) - prev_ball_x
        features["ball_velocity_y"] = ball_y - prev_ball_y
        features["paddle_velocity"] = paddle_y - prev_paddle_y
        
        # Is ball coming toward player? (negative x velocity = toward right paddle)
        features["ball_approaching"] = features["ball_velocity_x"] < 0
        
        # Did we get closer to the ball?
        prev_distance = abs(prev_ball_y - prev_paddle_y)
        curr_distance = features["paddle_ball_distance_y"]
        features["moving_toward_ball"] = curr_distance < prev_distance
    else:
        features["ball_velocity_x"] = 0
        features["ball_velocity_y"] = 0
        features["paddle_velocity"] = 0
        features["ball_approaching"] = False
        features["moving_toward_ball"] = False
    
    return features


def state_to_string(state: Dict, features: Dict) -> str:
    """Convert state and features to human-readable string for LLM"""
    if not state:
        return "No state available"
    
    lines = [
        f"Ball Position: ({state.get('ball_x', '?')}, {state.get('ball_y', '?')})",
        f"Player Paddle Y: {state.get('player_paddle_y', '?')}",
        f"CPU Paddle Y: {state.get('cpu_paddle_y', '?')}",
        f"Score: Player {state.get('player_score', 0)} - CPU {state.get('cpu_score', 0)}",
        f"Paddle-Ball Distance: {features.get('paddle_ball_distance_y', '?')} pixels",
        f"Ball Approaching: {features.get('ball_approaching', '?')}",
        f"Moving Toward Ball: {features.get('moving_toward_ball', '?')}",
    ]
    
    return "\n".join(lines)
