#!/bin/bash
# Train the ten-member 1000-epoch GELU ensemble on the two GPUs of one node. Submit from the repo root:
#   sbatch ensemble_train.sbatch
# Re-submitting after an interruption resumes every seed from its latest checkpoint.
#SBATCH --job-name=zhang_ens
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --output=results/zhang_5p3m_gelu_ens/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec "$PYTHON" -u -m kinetics.large_zhang_ensemble "$@"
