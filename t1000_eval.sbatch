#!/bin/bash
# Five approaches x five seeds, 10/20-ms rollouts on the T0 >= 1000 K cases, one besq node, sequential.
#SBATCH --job-name=t1k_eval
#SBATCH --partition=besq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=results/t1000_logs/eval-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH; export CUDA_VISIBLE_DEVICES=
SRC=results/t1000_long10ms_data/trajectories.npz; REF=results/t1000_reference/reference20ms.npz
run() { [ -f "$1/evaluation_$3_members/seed$2/manifest.json" ] && return 0
        rm -rf "$1/evaluation_$3_members/seed$2"
        "$PYTHON" -u -m kinetics.member_rollouts --root "$1" --seed "$2" --physics "$4" --source $SRC --reference $REF || echo "FAILED $1 $2 $4"; }
# Every evaluation uses best-validation weights: t1000_gelu/g0.1 is assembled from best/ (t1000_assemble);
# the UR roots get best-validation copies here.
"$PYTHON" -u -m kinetics.t1000_best_roots ur || { echo "FAILED best roots"; exit 1; }
for s in 11 22 33 44 55; do
  run results/t1000_gelu/g0.1 $s raw none        # 1) no UR, no physics
  run results/t1000_gelu/g0.1 $s wang wang       # 2) no UR, physics
  run results/t1000_ur10_best/g0.1 $s raw none        # 3) UR10, no physics
  run results/t1000_ur10_best/g0.1 $s wang wang       # 4) UR10 + physics
  run results/t1000_ur10_phys_best/g0.1 $s wang wang  # 5) UR10 trained with physics in the loop, + physics
done
echo EVAL_DONE
