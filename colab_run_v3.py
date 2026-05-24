# =========================================================================
# MILCOM Research Prototype — Google Colab Run Script (v3 — Fixed)
# Run these cells in order. Runtime: ~45-60 min on T4 GPU.
# =========================================================================

# --- CELL 1: Install Ollama and pull model ---
"""
!curl -fsSL https://ollama.com/install.sh | sh
!ollama serve &
import time; time.sleep(5)
!ollama pull qwen2.5-coder:7b-instruct
"""

# --- CELL 2: Clone repo and cd into it ---
"""
!git clone https://github.com/<YOUR_REPO>/RL-Ping-Pong-Training-Agent /content/RL-Ping-Pong-Training-Agent
%cd /content/RL-Ping-Pong-Training-Agent
"""

# --- CELL 3: Install dependencies ---
"""
!pip install stable-baselines3 gymnasium shimmy tensorboard
"""

# --- CELL 4: Verify Ollama and env ---
"""
!python -c "
from llm_interface import test_ollama_connection
from wireless_env import WirelessAntiJammingEnv
import numpy as np

print('Ollama:', 'OK' if test_ollama_connection() else 'FAIL — restart cell 1')

env = WirelessAntiJammingEnv(snr_mean=12.0, snr_threshold=5.0, seed=42)
obs, _ = env.reset()
obs2, r, _, _, info = env.step(env.action_space.sample())
from scipy import stats
p_snr = 1 - stats.norm.cdf((5.0 - 12.0) / 4.0)
print(f'P(SNR ok) = {p_snr:.3f}  (PDR ceiling = {p_snr:.3f}  Target 0.80: ACHIEVABLE={p_snr > 0.80})')
print('Features present:', sorted(info[\"features\"].keys()))
print('ENV OK')
"
"""

# --- CELL 5: Run iterative LLM training (main experiment) ---
"""
!python train_iterative.py train --timesteps 50000 --iterations 25 --target-pdr 0.80 --verbose 1
"""

# --- CELL 6: Evaluate final model ---
"""
!python train_iterative.py eval models/antijam_final.zip --episodes 50 --quiet
"""

# --- CELL 7: Evaluate best checkpoint ---
"""
import glob, os
# Find all saved checkpoints
checkpoints = sorted(glob.glob('models/antijam_best_iter*.zip'))
print(f'Checkpoints found: {len(checkpoints)}')
for ckpt in checkpoints:
    print(f'\\n--- {ckpt} ---')
    os.system(f'python train_iterative.py eval {ckpt} --episodes 30 --quiet')
"""

# --- CELL 8: Run baselines for comparison (paper Table) ---
"""
!python baselines.py
"""

# --- CELL 9: Zip and download results ---
"""
!zip -r antijam_results_v3.zip models/ logs/ rewards/history/ rewards/current_reward.py *.py config.py
from google.colab import files
files.download('antijam_results_v3.zip')
"""
