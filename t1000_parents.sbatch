#!/bin/bash
# T >= 1000 K pipeline: five parent seeds (stage 1: 10 epochs lr 1e-3; stage 2: to 1000 epochs lr 1e-4,
# halve after 20 epochs without lower training loss) as five lanes on the two GPUs of one node.
#   sbatch t1000_parents.sbatch          (re-submitting resumes every seed from its checkpoint)
#SBATCH --job-name=t1k_parents
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --output=results/t1000_logs/parents-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec "$PYTHON" -u -m kinetics.large_zhang_ensemble --seeds 11 22 33 44 55 --lanes 5 --fresh-seed11 \
  --source results/t1000_gelu/prep/g0.1 --stage1 configs/t1000/stage10.json --stage2 configs/t1000/stage1000.json \
  --output results/t1000_gelu --configs configs/t1000/ens
