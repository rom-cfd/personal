T/P/initial-phi DeepONet experiment
=================================

The user requested a trunk with pressure, current temperature and equivalence
ratio, one matched-ANN 10,000-update endpoint and one plateau continuation, with
a six-hour maximum. The complete preceding study was committed and pushed to
`bw` as `798a673f863c0aa6f59ab6b69d684815743145a6` before implementation or
training. GitHub's advertised `refs/heads/bw` was checked against the local SHA.
The backup includes all 11,159 previously untracked files: models, intermediate
and terminal Adam/RNG checkpoints, histories, audits and evaluation/plot outputs.

Use the existing checkout on `bw` and `/workspace/kinetics-venv/bin/python`.
Original modules and results are immutable. New code uses `tp_phi_*.py`; outputs
go below `results/tp_phi_study`, and implementation checks below
`results/tp_phi_checks`. Do not resume the old six-hour study to reset its budget.

Physical input rows are `[T, Y9, phi_initial, P_Pa]`. Phi comes from the dataset's
known IC descriptor `initials[trajectory,2]`, using each training pair's saved
parent trajectory. It remains constant for each self-fed trajectory, including
raw paths with elemental drift and batches that shrink after an IC fails. The
network feature order is the original eleven standardized state/logP features,
followed by training-only standardized phi. Common preprocessing matches the ANN
exactly; phi is not put into the target or the branch. The branch is current Y9
with widths64/64 and70 coefficients; the trunk is current T/logP/phi with width32
and7 basis values; product+bias gives10 residuals. Parameters=9,719. The fixed
9-species/28-reaction, 2us-target, 10ms benchmark and 54/18/18 split are unchanged.

First train all ten seeds11,22,...,110 sequentially to10k and save full state.
Then restore each exact checkpoint and continue sequentially. No optimizer,
minibatch stream, LR boundaries, history, or total clock is reset. Batch512,
replacement sampling, Adam, globalnormclip5 and absolute LR schedule .001 through
6000, .0003 through9000, .0001 thereafter match the original ANN rules. The
serial TensorFlow optimizer loop does not train independent members in parallel.
The scalar loop, serial loop and resumed loop are checked for array-equal weights,
Adam variables and next NumPy minibatches.

The unchanged plateau rule observes full-train MSE every1k; from50k, every5k,
compare q25/median/q75 of consecutive20k windows using relative thresholds
1%/1%/3%, require20 actual observations per window and three consecutive passes
spanning10k. Missing/nonfinite observations cannot pass. No additional update cap
is imposed in this experiment. Stops are matched budget, plateau, or time limit.

The six-hour pipeline clock starts when `--phase run` begins, after implementation
and preflight tests. It covers matched training, continuation, frozen endpoint
diagnostics, evaluations and reports;40 minutes are reserved for evaluation and
reporting. Remaining training time is divided among the ten members in sequence;
earlier plateau frees time for remaining members. The saved absolute expiry is
retained on resume. A capped member is never marked converged. Training may
finish before six hours if all members reach plateau.

All endpoints are frozen before reading test performance. Evaluate ANN10,
phi10k10, ANN+phi10k20, philong10, ANN+philong20, each raw/repaired and val/test
(20 arms). Mixed teams evaluate every member at their own same current state.
Existing physical guards, projection/enthalpy repair and invalid whole-candidate
exclusion rules are unchanged. No ODE fallback, teacher feedback, future-state
phi recomputation, medoid, or family reweighting is used. Full-horizon errors use
complete traces only and identical-IC intersections are reported separately.
Conservation is imposed by repair, entropy is diagnostic, and test ICs were
already observed in previous research. Phi is redundant with conserved elemental
ratios for an ideal closed mixture; its explicit trunk representation adds a
stable IC descriptor, not evidence of a new physical degree of freedom.

Set all execution environment variables before importing TensorFlow:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=-1 \
MPLCONFIGDIR=/tmp/kinetics-mpl XDG_CACHE_HOME=/tmp/kinetics-cache \
/workspace/kinetics-venv/bin/python -m kinetics.tp_phi_study \
  --config configs/tp_phi_study.json --phase prepare
```

After prepare, run with the same variables, `--phase run --resume`. Reproduction
requires a new output path. Resume verifies source/data/config/runtime and every
checkpoint hash before restoring optimizer and RNG. It never creates a fresh
six-hour budget for an existing run. Reports include actual timing/workload,
all observed loss tails, all test IC T+9-species profiles in PNG/PDF/CSV, raw and
repaired failure events, one-step and trajectory metrics, IC intersections, and
independently replayed plateau checks. NN feature/decode subcomponents should not
be added again to their `neural_decode` aggregate.

Historical long P-only/T/P endpoints had a different allocated training time and
different update counts. New long-vs-old comparisons cannot isolate architecture
from training time. The10k comparison against the same ANN10 is the matched-update
control. No test-based parameter changes are permitted during this run.
