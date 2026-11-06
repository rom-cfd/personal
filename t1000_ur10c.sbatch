#!/bin/bash
# UR=10 physics-in-loop consolidation of the gym windows with replay (1:1 and 1:3) x 5 seeds on 2 GPUs (one task per GPU),
# then best-validation roots and offline strategy-5-style evaluation on the 36 test/val cases.
#SBATCH --job-name=t1k_urc
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=results/t1000_logs/urc-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH; export TF_NUM_INTRAOP_THREADS=8 OMP_NUM_THREADS=8
T=""; for r in 1 3; do for s in 11 22 33 44 55; do T="$T configs/t1000/ur10c_r$r.json:$s"; done; done
"$PYTHON" -u -m kinetics.t1000_ur_lanes --lanes 2 --logs results/t1000_logs/urc --tasks $T || { echo FAILED training; exit 1; }
"$PYTHON" -u -c "from kinetics.t1000_best_roots import best_root
for r in (1,3):best_root(f'results/t1000_ur10c_r{r}/g0.1',f'results/t1000_ur10c_r{r}_best/g0.1')
print('BEST_ROOTS_READY',flush=True)" || { echo FAILED best roots; exit 1; }
SRC=results/t1000_long10ms_data/trajectories.npz; REF=results/t1000_reference/reference20ms.npz
ev() { [ -f "$1/evaluation_wang_members/seed$2/manifest.json" ] && return 0; rm -rf "$1/evaluation_wang_members/seed$2"
       CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 "$PYTHON" -u -m kinetics.member_rollouts --root "$1" --seed "$2" --physics wang --source $SRC --reference $REF \
         > "results/t1000_logs/urc-$SLURM_JOB_ID-eval-$(basename $(dirname $1))-s$2.out" 2>&1 || echo "FAILED eval $1 $2"; }
for r in 1 3; do for s in 11 22 33 44 55; do ev results/t1000_ur10c_r${r}_best/g0.1 $s & done; done; wait
echo URC_DONE
