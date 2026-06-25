#!/usr/bin/env bash
#SBATCH -J rl_noisy_h100
#SBATCH -p gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

set -Eeuo pipefail

echo "Starting RL-Noisy H100 job"
echo "Host: $(hostname)"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
date

# Always run from the directory where sbatch was submitted.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

SIF="${PROJECT_DIR}/rl_noisy.sif"

# Verify the container image exists
if [ ! -f "${SIF}" ]; then
    echo "ERROR: Container image not found: ${SIF}" >&2
    echo "Run this first on the login node: bash build_container.sh" >&2
    exit 1
fi

# Ensure directories exist
OLLAMA_MODELS_DIR="${PROJECT_DIR}/.ollama_models"
mkdir -p "${OLLAMA_MODELS_DIR}"
mkdir -p results logs

# Helper: run a command inside the container with GPU passthrough
# --nv            : passes GPU into the container
# --bind          : mounts project dir to /workspace inside container
# --env           : passes environment variables
APPTAINER_RUN="apptainer exec \
    --nv \
    --bind ${PROJECT_DIR}:/workspace \
    --env OLLAMA_MODELS=/workspace/.ollama_models \
    --env PYTHONUNBUFFERED=1 \
    --pwd /workspace \
    ${SIF}"

# Verify container + GPU
echo "Verifying container environment..."
${APPTAINER_RUN} python -c "
import sys, torch
print('python:', sys.version)
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

echo "=========================================================="
echo "Starting Ollama server in background..."
${APPTAINER_RUN} ollama serve > logs/ollama.log 2>&1 &
OLLAMA_PID=$!

echo "Waiting for Ollama server to initialize (15s)..."
sleep 15

echo "Pulling deepseek-r1:32b model (will be fast if already cached)..."
${APPTAINER_RUN} ollama pull deepseek-r1:32b

echo "=========================================================="
echo "Running Experiments"

# Run all paper baselines
echo "Running baselines..."
${APPTAINER_RUN} python -u run_experiment.py --set baselines

# Run all variant A conditions (uses the local ollama server we just started)
echo "Running Variant A..."
${APPTAINER_RUN} python -u run_experiment.py --set variant_a --ollama-url http://localhost:11434

# Run ablation
echo "Running Ablation..."
${APPTAINER_RUN} python -u run_experiment.py --set ablation

# Generate plots
echo "Generating plots..."
${APPTAINER_RUN} python -u run_experiment.py --plots-only

echo "=========================================================="
echo "Job complete. Shutting down Ollama..."
kill ${OLLAMA_PID} || true
date
