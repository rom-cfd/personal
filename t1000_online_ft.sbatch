#!/bin/bash
# Uncertainty-triggered online fine-tuning (L2-SP 1e-3, Adam 1e-5), 36 cases x 5 seeds in lockstep; 4 workers x 8 threads on one besq node.
#SBATCH --job-name=t1k_oft
#SBATCH --partition=besq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=results/t1000_logs/oft-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
"$PYTHON" -u -m kinetics.t1000_online_finetune --cases 7 0 14 23 33 45 57 71 76 > results/t1000_logs/oft-${SLURM_JOB_ID}-w0.out 2>&1 &
"$PYTHON" -u -m kinetics.t1000_online_finetune --cases 9 4 17 24 39 46 63 73 81 > results/t1000_logs/oft-${SLURM_JOB_ID}-w1.out 2>&1 &
"$PYTHON" -u -m kinetics.t1000_online_finetune --cases 52 69 10 19 32 44 53 68 75 > results/t1000_logs/oft-${SLURM_JOB_ID}-w2.out 2>&1 &
wait
echo OFT_DONE
