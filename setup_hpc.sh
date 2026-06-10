#!/usr/bin/env bash
# ==============================================================================
# setup_hpc.sh - RL-Noisy environment setup for HPC
# ==============================================================================
# Run once from the project directory on the login node:
#
#   bash setup_hpc.sh
# ==============================================================================

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

echo "Starting RL-Noisy HPC setup"
echo "Project directory: ${PROJECT_DIR}"

# Create required directories
echo "Creating required directories..."
mkdir -p results data logs .ollama_models

if [ ! -f "rl_noisy.sif" ]; then
    echo ""
    echo "WARNING: Apptainer container 'rl_noisy.sif' not found."
    echo "Please run: bash build_container.sh"
else
    echo "Container rl_noisy.sif exists."
fi

echo ""
echo "Setup complete. Directories created."
echo "If container is built, submit with: sbatch h100_job.slurm"
