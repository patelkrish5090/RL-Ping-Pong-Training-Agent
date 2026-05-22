"""
Configuration for LLM-Guided Reward Shaping — Anti-Jamming Channel Selection
MILCOM Research Prototype
"""
from pathlib import Path

# =============================================================================
# WIRELESS ENVIRONMENT PARAMETERS
# =============================================================================
N_CHANNELS = 8                    # Number of available wireless channels
JAMMER_MODES = ["random", "sweep", "reactive"]
N_JAMMED_CHANNELS = 2             # Number of channels jammed in random/sweep modes
SWEEP_WIDTH = 2                   # Number of adjacent channels hit by sweep jammer
REACTIVE_JAM_PROB = 0.95          # Probability reactive jammer attacks last-used channel
REACTIVE_EXTRA_RANDOM = 1         # Extra random channels jammed by reactive jammer
JAMMER_CHANGE_INTERVAL = 100      # Steps before jammer switches strategy
MAX_STEPS_PER_EPISODE = 500       # Steps per training episode
SNR_MEAN = 8.0                    # Mean per-channel SNR (dB)
SNR_STD = 4.0                     # SNR fluctuation standard deviation (dB)
SNR_THRESHOLD = 7.0               # Min SNR for successful TX (dB)
QUEUE_CAPACITY = 20               # Max packet queue depth
ARRIVAL_RATE = 0.35               # Packet arrival probability per arrival slot
MAX_PACKET_ARRIVALS = 2           # Packet arrival slots per step
ENERGY_PER_TX = 1.0              # Energy cost per transmission attempt
SWITCH_DISRUPTION_PROB = 0.15     # Probability a channel switch disrupts the packet
SWITCH_ENERGY_COST = 0.25         # Extra energy cost paid when changing channels
SWITCH_REWARD_PENALTY = 0.05      # Base penalty for switching overhead
SNR_HISTORY_LEN = 10              # Window for rolling success-rate estimates
N_ENVS = 4                        # Parallel environments (MLP is fast)

# =============================================================================
# PPO HYPERPARAMETERS (Tuned for MLP / tabular-ish env)
# =============================================================================
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 64,
    "n_epochs": 10,
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
EPISODES_PER_UPDATE = 20           # Analyze this many recent episodes for LLM
MAX_ITERATIONS = 20                # Maximum LLM reward-refinement iterations
TARGET_PDR = 0.80                  # Stop training when rolling PDR >= this
TIMESTEPS_PER_ITERATION = 20000   # Training steps between LLM updates

# =============================================================================
# LLM CONFIGURATION (Ollama)
# =============================================================================
LLM_CONFIG = {
    "model": "qwen2.5-coder:7b-instruct",  # Or "deepseek-coder:6.7b"
    "host": "http://localhost:11434",
    "temperature": 0.7,
    "num_ctx": 8192,
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
