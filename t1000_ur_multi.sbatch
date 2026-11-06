#!/bin/bash
# T >= 1000 K UR=10 fine-tunes, all ten tasks in ONE job on 3 GPU nodes (6 GPUs), one task per GPU at a time.
# Every task resumes from its own checkpoint (unroll_training --resume); parents are the best-validation models.
#   node 1: ur10_phys s11, s22      node 2: ur10_phys s33, s44
#   node 3: ur10_phys s55 on one GPU; ur10 (raw) s11 -> s22 -> s33 -> s44 -> s55 in sequence on the other
#SBATCH --job-name=t1k_ur
#SBATCH --partition=gpuq
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --output=results/t1000_logs/ur3-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH
P=configs/t1000/ur10_phys.json; R=configs/t1000/ur10.json
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
TASKS=("$P:11 $P:22" "$P:33 $P:44" "$P:55 $R:11 $R:22 $R:33 $R:44 $R:55")
for i in 0 1 2; do
  srun --nodes=1 --ntasks=1 --nodelist="${NODES[$i]}" --gres=gpu:2 --cpus-per-task=32 \
    bash -c 'echo NODE $(hostname) GPUS $CUDA_VISIBLE_DEVICES; exec "$0" -u -m kinetics.t1000_ur_lanes --lanes 2 --tasks $1' \
    "$PYTHON" "${TASKS[$i]}" > "results/t1000_logs/ur3-$SLURM_JOB_ID-node$i.out" 2>&1 &
done
fail=0; for j in $(jobs -p); do wait $j || fail=1; done
echo UR_ALL_DONE fail=$fail; exit $fail
