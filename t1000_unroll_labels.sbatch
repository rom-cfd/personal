#!/bin/bash
# 11-step Cantera labels for the T >= 1000 K data, exact float64 initial states, one 32-core besq node.
#SBATCH --job-name=t1k_ur11
#SBATCH --partition=besq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=results/t1000_logs/unroll_labels-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH; export CUDA_VISIBLE_DEVICES=
export UNROLL_SRC=results/t1000_gelu/prep/g0.1/preprocessing UNROLL_OUT=results/t1000_unroll11 UNROLL_DATA=results/t1000_data
[ -f $UNROLL_OUT/initial_states.npy ] || "$PYTHON" -u -m kinetics.unroll_data --prepare
"$PYTHON" -u -m kinetics.unroll_data --all-chunks --workers 32
"$PYTHON" -u -m kinetics.unroll_data --repair --workers 32
"$PYTHON" -u -m kinetics.unroll_data --assemble
