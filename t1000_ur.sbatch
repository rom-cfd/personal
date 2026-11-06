#!/bin/bash
# T >= 1000 K pipeline: assemble parents, then the ten UR=10 fine-tunes as 6 lanes on 2 GPUs.
#SBATCH --job-name=t1k_ur
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --output=results/t1000_logs/ur-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
CUDA_VISIBLE_DEVICES= "$PYTHON" -u -m kinetics.t1000_assemble
exec "$PYTHON" -u -m kinetics.t1000_ur_lanes
