#!/bin/bash
# Liu et al. (2026) UR=10 fine-tune of one 1000-epoch seed on one GPU. Submit from the repo root:
#   sbatch --export=ALL,SEED=11[,CONFIG=configs/zhang_unroll10_ppin.json] unroll_train.sbatch
# Re-submitting the same SEED resumes from its latest checkpoint.
#SBATCH --job-name=unroll10
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/zhang_5p3m_unroll10/slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH
export TF_FORCE_GPU_ALLOW_GROWTH=true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec "$PYTHON" -u -m kinetics.unroll_training --config "${CONFIG:-configs/zhang_unroll10.json}" --seed "$SEED" --resume
