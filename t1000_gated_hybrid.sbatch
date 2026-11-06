#!/bin/bash
# Disagreement-gated hybrid prototype (strategy 5 + Cantera), 36 cases x 5 seeds, three variants, one besq node.
#SBATCH --job-name=t1k_gate
#SBATCH --partition=besq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --output=results/t1000_logs/gate-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
"$PYTHON" -u -m kinetics.t1000_gated_hybrid --threshold inf || echo FAILED gate_off
"$PYTHON" -u -m kinetics.t1000_gated_hybrid --threshold 0.3 --handover 50 || echo FAILED h50
"$PYTHON" -u -m kinetics.t1000_gated_hybrid --threshold 0.3 --handover 200 || echo FAILED h200
"$PYTHON" -u -m kinetics.t1000_ode_cost || echo FAILED ode_cost
echo GATE_DONE
