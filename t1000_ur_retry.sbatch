#!/bin/bash
# Rerun of the physics-in-loop UR=10 fine-tunes that crashed at startup in job 132745 (seeds 11 22 44:
# shared config.json.tmp race in atomic_json, fixed 2026-11-03). Parents are the best-validation models.
#SBATCH --job-name=t1k_urfix
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --output=results/t1000_logs/urfix-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec "$PYTHON" -u -m kinetics.t1000_ur_lanes --configs configs/t1000/ur10_phys.json --seeds 11 22 44 --lanes 3
