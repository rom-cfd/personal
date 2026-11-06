#!/bin/bash
# EWC consolidation of the gym data (c sweep x 5 seeds) on 2 GPUs, then offline strategy-5-style evaluation of every model
# on the 36 test/val cases (CPU, 8 at a time). c is chosen afterwards on the validation rollouts only.
#SBATCH --job-name=t1k_ewc
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=results/t1000_logs/ewc-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH; export TF_NUM_INTRAOP_THREADS=8 OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u -m kinetics.t1000_ewc --seeds 11 22 33 > results/t1000_logs/ewc-$SLURM_JOB_ID-gpu0.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u -m kinetics.t1000_ewc --seeds 44 55 > results/t1000_logs/ewc-$SLURM_JOB_ID-gpu1.out 2>&1 &
wait; echo TRAIN_DONE
SRC=results/t1000_long10ms_data/trajectories.npz; REF=results/t1000_reference/reference20ms.npz
ev() { [ -f "$1/evaluation_wang_members/seed$2/manifest.json" ] && return 0; rm -rf "$1/evaluation_wang_members/seed$2"
       CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 "$PYTHON" -u -m kinetics.member_rollouts --root "$1" --seed "$2" --physics wang --source $SRC --reference $REF \
         > "results/t1000_logs/ewc-$SLURM_JOB_ID-eval-$(basename $(dirname $1))-s$2.out" 2>&1 || echo "FAILED eval $1 $2"; }
n=0
for c in c0 c0.01 c0.1 c1 c10; do for s in 11 22 33 44 55; do
  [ -f results/t1000_ewc/$c/g0.1/seed$s/completed.json ] || { echo "MISSING $c $s"; continue; }
  ev results/t1000_ewc/$c/g0.1 $s & n=$((n+1)); [ $((n%8)) -eq 0 ] && wait
done; done; wait
echo EWC_EVAL_DONE
