"""
Configuration for LLM-Guided Reward Shaping — Anti-Jamming Channel Selection
MILCOM Research Prototype

CHANGE LOG (v2 — After Colab run analysis):
  - SNR settings adjusted: ceiling raised so PDR=0.80 is physically achievable
  - PPO n_steps increased from 512 to 2048 for better credit assignment
  - PPO batch_size increased from 64 to 256 to match larger rollouts
  - Entropy coef reduced to encourage more exploitation once reward is shaped
  - TIMESTEPS_PER_ITERATION increased to 50K (already done in v1)
  - N_ENVS increased to 8 for faster wall-clock training in Colab
"""
from pathlib import Path

# =============================================================================
# WIRELESS ENVIRONMENT PARAMETERS (INSANE DIFFICULTY MODE)
# =============================================================================
N_CHANNELS = 32                   # Increased from 8 to 32 (wide spectrum)
JAMMER_MODES = ["random", "sweep", "reactive"]
N_JAMMED_CHANNELS = 16            # Half the spectrum (16/32) is jammed at all times!
SWEEP_WIDTH = 8                   # Sweep jammer takes out massive blocks of 8 channels
REACTIVE_JAM_PROB = 0.99          # 99% chance reactive jammer hits your last channel
REACTIVE_EXTRA_RANDOM = 8         # Reactive jammer also hits 8 other random channels
JAMMER_CHANGE_INTERVAL = 25       # Jammer changes strategy VERY fast (every 25 steps)
MAX_STEPS_PER_EPISODE = 500       # Steps per training episode

# SNR settings (Kept mean=12 so signal is okay, the real challenge is the jammer)
SNR_MEAN = 12.0                   
SNR_STD = 4.0                     
SNR_THRESHOLD = 5.0               

QUEUE_CAPACITY = 20               
ARRIVAL_RATE = 0.35               
MAX_PACKET_ARRIVALS = 2           
ENERGY_PER_TX = 1.0              
SWITCH_DISRUPTION_PROB = 0.25     # Increased to 25%: running away from jammers often drops packets
SWITCH_ENERGY_COST = 0.50         # Switching costs double energy now
SWITCH_REWARD_PENALTY = 0.10      # Base env switch penalty doubled
SNR_HISTORY_LEN = 10              
N_ENVS = 8                        

# =============================================================================
# PPO HYPERPARAMETERS — tuned for this environment
# =============================================================================
# KEY CHANGES:
#   n_steps: 512 → 2048   — was ~4 episodes/update, now ~16 episodes/update
#   batch_size: 64 → 256  — matched to larger rollout (n_steps * N_ENVS / 4)
#   n_epochs: 10 → 15     — more gradient steps per collected batch
#   ent_coef: 0.01 → 0.005 — encourage exploitation once policy converges
#   learning_rate: 3e-4 → 2e-4 — slightly more conservative, better for long training
PPO_CONFIG = {
    "learning_rate": 2e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 15,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.005,
    "clip_range": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

# =============================================================================
# ITERATIVE REFINEMENT
# =============================================================================
EPISODES_PER_UPDATE = 20           # Analyze this many recent episodes for LLM
MAX_ITERATIONS = 30                # Increased from 25 for insane mode
TARGET_PDR = 0.50                  # Max possible is ~0.48, so it will probably never hit this, which ensures full training
TIMESTEPS_PER_ITERATION = 100000   # Doubled to 100K steps per LLM update because 32 channels takes longer to learn

# =============================================================================
# LLM CONFIGURATION (Ollama)
# =============================================================================
# HOST NOTE:
#   Local machine : "http://localhost:11434"  OR  "http://127.0.0.1:11434"
#   Google Colab  : "http://127.0.0.1:11434"  (Ollama binds to 127.0.0.1 inside the VM)
#                   Make sure Ollama is started inside the Colab cell with:
#                       !curl -fsSL https://ollama.com/install.sh | sh
#                       !ollama serve &
#                       !ollama pull qwen2.5-coder:7b-instruct
LLM_CONFIG = {
    "model": "qwen2.5-coder:7b-instruct",
    "host": "http://127.0.0.1:11434",
    "temperature": 0.4,            # Lowered from 0.7 — more deterministic code
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
