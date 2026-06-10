#!/usr/bin/env bash
# ==============================================================================
# build_container.sh - Build the RL-Noisy Apptainer container
# ==============================================================================
# Run ONCE on the login node. Takes ~10-20 minutes to download and build.
# Usage: bash build_container.sh
# ==============================================================================

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

SIF_FILE="${PROJECT_DIR}/rl_noisy.sif"
DEF_FILE="${PROJECT_DIR}/rl_noisy.def"

echo "Building RL-Noisy Apptainer container..."
echo "Definition file: ${DEF_FILE}"
echo "Output SIF: ${SIF_FILE}"

# Build the container from the .def file
# --fakeroot allows building without real root privileges
apptainer build --fakeroot "${SIF_FILE}" "${DEF_FILE}"

echo ""
echo "Build complete: ${SIF_FILE}"
echo "Size: $(du -sh ${SIF_FILE} | cut -f1)"

# Run the built-in test
echo "Running container self-test (Python imports)..."
apptainer exec "${SIF_FILE}" python -c "import torch, gymnasium, requests; print('Self-test passed!')"

echo ""
echo "Container is ready!"
echo "Submit your job with: sbatch h100_job.slurm"
