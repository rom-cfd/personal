#!/bin/bash
# Uncertainty-triggered online fine-tuning (L2-SP 1e-3, Adam 1e-5), 36 cases x 5 seeds in lockstep; worker 3 (case 78 + 8 regular cases) split off on one besq node.
#SBATCH --job-name=t1k_oft
#SBATCH --partition=besq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=results/t1000_logs/oft-w3-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
"$PYTHON" -u -m kinetics.t1000_online_finetune --cases 78 5 18 29 41 48 64 74 88 > results/t1000_logs/oft-132817-w3.out 2>&1
echo OFT_W3_DONE
