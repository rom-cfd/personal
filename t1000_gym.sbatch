#!/bin/bash
# 0-D ignition gym (train+val episodes, weights carried, 2 passes) on one GPU, then offline strategy-5-style
# evaluation of the end-of-pass models on all 36 test/val cases (no online updates, no ODE in the rollout).
#SBATCH --job-name=t1k_gym
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=results/t1000_logs/gym-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH; export TF_NUM_INTRAOP_THREADS=8 OMP_NUM_THREADS=8
R=""; [ -f results/t1000_gym/state.json ] && R=--resume
"$PYTHON" -u -m kinetics.t1000_gym --passes 2 $R || { echo FAILED gym; exit 1; }
SRC=results/t1000_long10ms_data/trajectories.npz; REF=results/t1000_reference/reference20ms.npz
ev() { [ -f "$1/evaluation_wang_members/seed$2/manifest.json" ] && return 0; rm -rf "$1/evaluation_wang_members/seed$2"
       CUDA_VISIBLE_DEVICES= "$PYTHON" -u -m kinetics.member_rollouts --root "$1" --seed "$2" --physics wang --source $SRC --reference $REF \
         > "results/t1000_logs/gym-$SLURM_JOB_ID-eval-$(basename $(dirname $1))-s$2.out" 2>&1 || echo "FAILED eval $1 $2"; }
for P in 1 2; do
  for s in 11 22 33 44 55; do ev results/t1000_gym/pass$P/g0.1 $s & done; wait
done
echo GYM_EVAL_DONE
