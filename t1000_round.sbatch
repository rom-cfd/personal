#!/bin/bash
# Round ROUND (>= 3) of the iterated gym -> UR10 consolidation loop (round 2: t1000_round2.sbatch; same steps):
#  1. gym (train+val episodes, 2 passes, R > 0.3 trigger + one forced round per episode at the first 50 K rise,
#     L2-SP fine-tune) with the previous round's best-validation members -> results/t1000_gym$ROUND
#  2. 11-step Cantera windows of this round's pairs, pooled with ALL earlier rounds' gym windows (aggregation),
#     1:1 replay of the original UR10 windows -> results/t1000_gymur${ROUND}_r1
#  3. UR10 physics-in-loop consolidation x 5 seeds from the previous round's members -> results/t1000_ur10c${ROUND}_r1(_best)
#  4. offline strategy-5-style rollouts of the best-validation members on the 36 test/val cases (test never visited)
#  5. unless results/t1000_rounds.STOP exists, submit round ROUND+1. The loop ends when the user asks to stop, or when the
#     gym of a round triggers no round at all (no pre-ignition point with R > 0.3): the previous round's model is final.
#SBATCH --job-name=t1k_round
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=results/t1000_logs/round-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${ROUND:?set ROUND}"; [ "$ROUND" -ge 3 ] || { echo "ROUND must be >= 3"; exit 1; }
echo "ROUND $ROUND job $SLURM_JOB_ID"
PYTHON=${PYTHON:-$HOME/anaconda3/envs/kinetics-gpu/bin/python}
unset LD_LIBRARY_PATH; export TF_NUM_INTRAOP_THREADS=8 OMP_NUM_THREADS=8
PREV=results/t1000_ur10c$((ROUND-1))_r1_best/g0.1; G=results/t1000_gym$ROUND
for s in 11 22 33 44 55; do [ -f $PREV/evaluation_wang_members/seed$s/manifest.json ] || { echo "FAILED previous round $PREV seed $s not evaluated"; exit 1; }; done
if [ ! -f $G/manifest.json ]; then
  R=""; [ -f $G/state.json ] && R=--resume
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u -m kinetics.t1000_gym --root $PREV --out $G --passes 2 --force-round $R || { echo FAILED gym; exit 1; }
fi
echo GYM_ROUND_DONE $ROUND
# stopping rule (user, 2026-11-05): the loop ends when the gym finds no pre-ignition point with R > 0.3, i.e. zero
# triggered rounds in both passes (forced rounds do not count). Round ROUND-1's model is then the final model.
TRIG=$("$PYTHON" -c "import json;L=json.load(open('$G/state.json'))['log'];print(sum(x['rounds']-int(x.get('forced',False)) for x in L))")
echo "TRIGGERED_ROUNDS $ROUND $TRIG"
if [ "$TRIG" = 0 ]; then echo "CONVERGED: no R > 0.3 point in round $ROUND gym; final model = $PREV"; touch results/t1000_rounds.STOP; exit 0; fi
GY="results/t1000_gym/rounds:results/t1000_gymur_windows.npz"
for k in $(seq 2 $ROUND); do GY="$GY results/t1000_gym$k/rounds:results/t1000_gymur${k}_windows.npz"; done
"$PYTHON" -u -m kinetics.t1000_gymur_data --workers 30 --ratios 1 --parent $PREV --out-prefix results/t1000_gymur${ROUND}_r --gyms $GY || { echo FAILED windows; exit 1; }
echo WINDOWS_ROUND_DONE $ROUND
CFG=configs/t1000/ur10c${ROUND}_r1.json
"$PYTHON" - "$ROUND" "$CFG" <<'PY' || { echo FAILED config; exit 1; }
import json,sys
n=int(sys.argv[1]);c=json.load(open('configs/t1000/ur10c2_r1.json'))
c.update(method=f'Round {n}: UR=10 physics-in-loop consolidation of the pooled round-1..{n} gym windows with 1:1 replay of the original UR10 windows; parent = round-{n-1} best-validation members',
         parent_root=f'results/t1000_ur10c{n-1}_r1_best/g0.1',data=f'results/t1000_gymur{n}_r1',output=f'results/t1000_ur10c{n}_r1')
json.dump(c,open(sys.argv[2],'w'),indent=2)
PY
T=""; for s in 11 22 33 44 55; do T="$T $CFG:$s"; done
"$PYTHON" -u -m kinetics.t1000_ur_lanes --lanes 2 --logs results/t1000_logs/urc$ROUND --tasks $T || { echo FAILED training; exit 1; }
"$PYTHON" -u -c "from kinetics.t1000_best_roots import best_root
best_root('results/t1000_ur10c${ROUND}_r1/g0.1','results/t1000_ur10c${ROUND}_r1_best/g0.1');print('BEST_ROOT_READY',flush=True)" || { echo FAILED best root; exit 1; }
SRC=results/t1000_long10ms_data/trajectories.npz; REF=results/t1000_reference/reference20ms.npz
B=results/t1000_ur10c${ROUND}_r1_best/g0.1; bad=0
for s in 11 22 33 44 55; do
  ( [ -f $B/evaluation_wang_members/seed$s/manifest.json ] && exit 0; rm -rf $B/evaluation_wang_members/seed$s
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 "$PYTHON" -u -m kinetics.member_rollouts --root $B --seed $s --physics wang --source $SRC --reference $REF \
      > results/t1000_logs/round$ROUND-$SLURM_JOB_ID-eval-s$s.out 2>&1 ) &
done; wait
for s in 11 22 33 44 55; do [ -f $B/evaluation_wang_members/seed$s/manifest.json ] || { echo "FAILED eval seed $s"; bad=1; }; done
[ $bad = 0 ] || exit 1
echo ROUND_DONE $ROUND
if [ -f results/t1000_rounds.STOP ]; then echo "STOP file present; not submitting round $((ROUND+1))"
else sbatch --export=ALL,ROUND=$((ROUND+1)) t1000_round.sbatch && echo "SUBMITTED round $((ROUND+1))"; fi
