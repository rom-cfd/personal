Six-hour serial continuation of120 lambda-sweep models
====================================================

The user requested publication of the completed matched10k lambda sweep before
starting its long continuation, then explicitly authorized a maximum six hours
for120 models at lambda1,100,1000 including evaluation and figures. The completed
sweep is committed and pushed on bw at f61eff1e9fe44512a68ddecc5b431a9b792a526c.
Use the existing checkout. No new lambda0 or lambda10 training is requested.

Continue the120 existing10000-update checkpoints exactly: same weights, full
Adam variables/iteration, NumPy and TensorFlow RNGs, preprocessing, data8815
pairs, branch/trunk partitions, lambda, batch512, clip5 and absolute LR schedule.
LR remains.0001 beyond9000. Reuse the frozen SoftTrainer and its original source
signatures. New modules use weight_long_* names; never edit the origin trainer
to bypass a provenance check. Resume accepts complete observed checkpoints even
if an interruption preceded a terminal record. Already-finished first-pass
models do not receive a fresh initial allocation after resume.

Joint stopping is identical to the previous lambda10 continuation: train-only
total loss, data MSE and physical element L1 must ALL pass. Observe every1000,
check every5000, at least50000 total updates, two adjacent20000-update windows
with20 actual observations each, relative q25/median/q75 changes at most1%/1%/3%,
three consecutive passing checks spanning10000 updates. Do not fabricate missing
observations or label a flat total with cancelling component trends as plateau.
No production update ceiling is added. A time cap remains a time cap.

One absolute21600-second clock is shared by all models and phases. Reserve3600
seconds for diagnostics/evaluation/figures. Divide remaining training time among
remaining first-pass tasks, in lambda1→100→1000 order, and ANN→P-only→T/P→T/P/phi,
ten seeds each. This normally yields about150 seconds per member, with fewer
updates than the older40-member lambda10 long run. Early plateaus free time for
remaining models; spare training time may revisit caps without resetting state.
Do not reset the six-hour budget on resume. Some models may not reach the
minimum observation window before their allocation ends; report this honestly.

All training, model evaluation and plotting are sequential. CPU intra2/inter1,
OMP/OpenBLAS/MKL2, CUDA=-1 and oneDNN0 before import. Record original and added
updates, terminal reasons, active/optimizer/checkpoint time and whole-stage wall
time. Freeze all120 endpoints before test evaluation. Reuse the original dense
and self-fed raw/repaired evaluators, same-state10/20-member means, strict decode
and physics rules. No new penalty, preprocessing, augmentation or ODE fallback.

For EACH lambda give18 established7x10 profiles, each followed by its matched
NRMSE table: Cantera, baseline lambda0 at10k, that lambda's10k and long endpoint.
Raw/repaired each have36-page PDFs and PNG/CSV exports. Mixed long teams include
both corresponding long ANN10 and long DeepONet10. Summary medians use explicitly
listed common completed ICs of every displayed variant; failures have no full
score. Supplemental within-team tables use that team's common cohorts and are
not architecture rankings across different IC sets. Cross-weight tables compare
the new three lambdas' short/long42 variants on common cohorts. The older lambda10
long study used much more per-model time and different updates, so keep it as a
separate historical reference. Long endpoints have unequal updates; do not call
their ranking a fixed-update or fully converged lambda comparison.

NRMSE remains the user's util.py default RMSE(Cantera,NN)/RMS(Cantera), equal
sample weights and unscaled dimensionless ratios. Retain absolute RMSE, reference
RMS, mean/std/quartiles and completion counts. Original histories are preserved;
loss plots show continuation from10000 and every individual tail, with geometric
means only on common observed updates. Compare weighted total losses only within
the same lambda. Export separate MSE and unweighted element losses.

Before reporting verify all actual Adam iterations, unchanged original signatures,
preprocessors and history prefixes, replay every joint plateau check, and check
all original-result hashes. Write new outputs under results/weight_long_study.
Tests use results/weight_long_checks and finish before the immutable run snapshot.

Previous experiments also ran in this managed cloud environment. For this run,
launch a standard-library-only supervisor in a new OS session. It starts the
TensorFlow pipeline in another new session, keeps an exclusive per-run process
lock, retains the original budget on resume, writes console/exit records and
terminates an overrun at the absolute six-hour expiry. This removes dependence
on the interactive command session; it cannot keep a stopped/deleted cloud
environment alive. The supervisor performs no NN work and does not compete with
the serial compute timing. The pipeline automatically evaluates and reports after
training, without requiring a subsequent UI command.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=-1 \
MPLCONFIGDIR=/tmp/kinetics-mpl XDG_CACHE_HOME=/tmp/kinetics-cache \
/workspace/kinetics-venv/bin/python -m kinetics.weight_long_study \
  --config configs/weight_long_study.json --phase prepare
```

After checks, launch scripts/run_weight_long_supervisor.py with --config and a
unique --attempt directory under the prepared output using Popen with
start_new_session=True, stdin=DEVNULL and stdout/stderr to that attempt's console
log. Write the PID/session/command to start.json. Inspect root execution.jsonl,
budget.json, terminal checkpoints, FINAL_ARTIFACTS.json and the attempt exit.json.
This new long experiment is not automatically committed/pushed after completion.
