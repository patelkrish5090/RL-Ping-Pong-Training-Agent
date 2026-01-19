"""Configuration for Pong RL Training with LLM Reward Refinement"""
from pathlib import Path

# =============================================================================
# ENVIRONMENT
# =============================================================================
ENV_ID = "PongNoFrameskip-v4"  # Atari Pong
N_ENVS = 8                      # Parallel environments
FRAME_STACK = 4                 # Stacked frames for temporal info

# =============================================================================
# PPO HYPERPARAMETERS (Tuned for Atari)
# =============================================================================
PPO_CONFIG = {
    "learning_rate": 2.5e-4,
    "n_steps": 128,
    "batch_size": 256,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

# =============================================================================
# ITERATIVE REFINEMENT
# =============================================================================
EPISODES_PER_UPDATE = 10        # Train N episodes, then ask LLM
MAX_ITERATIONS = 50             # Maximum LLM updates
TARGET_SCORE = 19.0             # Stop when reached
TIMESTEPS_PER_ITERATION = 100000 # Timesteps between LLM updates

# =============================================================================
# LLM CONFIGURATION (Ollama)
# =============================================================================
LLM_CONFIG = {
    "model": "qwen2.5-coder:7b-instruct",  # Or "deepseek-coder:6.7b"
    "host": "http://localhost:11434",
    "temperature": 0.8,
    "num_ctx": 8192,
}

# =============================================================================
# PONG RAM ADDRESSES (For state extraction)
# =============================================================================
PONG_RAM = {
    "ball_x": 49,
    "ball_y": 54,
    "player_paddle_y": 51,
    "cpu_paddle_y": 50,
    "player_score": 14,
    "cpu_score": 13,
}

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
REWARD_FILE = PROJECT_ROOT / "rewards" / "current_reward.py"
REWARD_HISTORY_DIR = PROJECT_ROOT / "rewards" / "history"
LOG_DIR = PROJECT_ROOT / "logs" / "training"
MODEL_DIR = PROJECT_ROOT / "models"

# Create directories
REWARD_FILE.parent.mkdir(parents=True, exist_ok=True)
REWARD_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
