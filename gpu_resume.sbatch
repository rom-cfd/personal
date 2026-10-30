#!/bin/bash
# Resume the 1000-epoch GELU run on one GPU node. Submit from the repo root:
#   sbatch gpu_resume.sbatch
# Re-submitting after an interruption continues from the latest checkpoint.
#SBATCH --job-name=zhang1000
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/zhang_5p3m_gelu_1000/g0.1/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Compute nodes have only Python 3.6; use the shared conda env built from requirements.gpu.txt.
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec "$PYTHON" -u -m kinetics.gpu_resume "$@"
