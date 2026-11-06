#!/bin/bash
# Round 2 of gym -> UR10 consolidation, starting from the round-1 consolidation (UR10, replay 1:1):
#  1. gym (train+val episodes, 2 passes, same trigger/perturbation/L2-SP recipe) with the round-1 members -> results/t1000_gym2;
#     --force-round: every episode without a triggered round gets one round at the first 50 K rise (every case contributes data)
#  2. 11-step Cantera windows of the round-2 pairs; pooled with the round-1 gym windows; 1:1 replay -> results/t1000_gymur2_r1
#  3. UR10 physics-in-loop consolidation x 5 seeds from the round-1 members -> results/t1000_ur10c2_r1(_best)
#  4. offline strategy-5-style rollouts of the best-validation members on the 36 test/val cases
#SBATCH --job-name=t1k_round2
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=results/t1000_logs/round2-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH; export TF_NUM_INTRAOP_THREADS=8 OMP_NUM_THREADS=8
P1=results/t1000_ur10c_r1_best/g0.1; G2=results/t1000_gym2
if [ ! -f $G2/manifest.json ]; then
  R=""; [ -f $G2/state.json ] && R=--resume
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u -m kinetics.t1000_gym --root $P1 --out $G2 --passes 2 --force-round $R || { echo FAILED gym; exit 1; }
fi
echo GYM2_DONE
"$PYTHON" -u -m kinetics.t1000_gymur_data --workers 30 --ratios 1 --parent $P1 --out-prefix results/t1000_gymur2_r \
  --gyms results/t1000_gym/rounds:results/t1000_gymur_windows.npz $G2/rounds:results/t1000_gymur2_windows.npz || { echo FAILED windows; exit 1; }
echo WINDOWS2_DONE
T=""; for s in 11 22 33 44 55; do T="$T configs/t1000/ur10c2_r1.json:$s"; done
"$PYTHON" -u -m kinetics.t1000_ur_lanes --lanes 2 --logs results/t1000_logs/urc2 --tasks $T || { echo FAILED training; exit 1; }
"$PYTHON" -u -c "from kinetics.t1000_best_roots import best_root
best_root('results/t1000_ur10c2_r1/g0.1','results/t1000_ur10c2_r1_best/g0.1');print('BEST_ROOT_READY',flush=True)" || { echo FAILED best root; exit 1; }
SRC=results/t1000_long10ms_data/trajectories.npz; REF=results/t1000_reference/reference20ms.npz
ev() { [ -f "$1/evaluation_wang_members/seed$2/manifest.json" ] && return 0; rm -rf "$1/evaluation_wang_members/seed$2"
       CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 "$PYTHON" -u -m kinetics.member_rollouts --root "$1" --seed "$2" --physics wang --source $SRC --reference $REF \
         > "results/t1000_logs/round2-$SLURM_JOB_ID-eval-s$2.out" 2>&1 || echo "FAILED eval $1 $2"; }
for s in 11 22 33 44 55; do ev results/t1000_ur10c2_r1_best/g0.1 $s & done; wait
echo ROUND2_DONE
