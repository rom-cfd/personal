Matched10k element-penalty weight comparison
==========================================

The user requested lambda1,100,1000 training and results after the lambda10
continuation study. Compare weights at the same10000 updates per model first:
three weights x four families x ten seeds =120 new members,1200000 updates.
The existing lambda0 and lambda10 models and traces are the frozen10k controls.
The differently budgeted lambda10 long endpoints are not a matched control.
All new starts are fresh; seed-specific initialization and minibatch streams
match across weights. ANN10, P-only10, ANN+P-only20, T/P10, ANN+T/P20,
T/P/initial-phi10, ANN+T/P/initial-phi20 use the corresponding lambda's ANN and
DeepONet models. No ANN100 comparison or extra lambda10 retraining is included.

Keep the original data,8815 selected pairs,2us targets,10ms horizon, train-only
Box-Cox gamma.1 preprocessing, network partitions/widths, Adam, batch512,
clipping5 and LR boundaries unchanged. The objective is the original float32
residual MSE plus lambda times the original float64 physical H/O/N element L1.
The signed Box-Cox inverse extension remains loss-only; strict actual decoding
and raw/repaired inference are unchanged. Existing soft_constraint_* modules
remain immutable so original checkpoint signatures continue to verify. New code
uses penalty_sweep_* names and results/penalty_weight_sweep/run_001 outputs.

Train120 members serially, then freeze all endpoints before evaluating42 new
test arms. Run diagnostics, evaluation and plotting serially too. TensorFlow
intra2/inter1, OMP/OpenBLAS/MKL2, oneDNN0 before import, CPU only. Record per-member
optimizer/trace/checkpoint timing and per-weight/stage wall time. The authorized
fixed update count is the normal stop criterion;3h training and4h whole-pipeline
guards prevent runaway jobs and never permit extra updates. Resume retains the
original absolute clock and all optimizer/RNG states. A guard stop is incomplete,
not convergence. This task does not request another long/plateau training run.

Each test IC gets the established7x10 graph (Cantera plus five lambda curves)
and its matched35-row NRMSE table. Raw/repaired each get36-page PDFs and PNGs.
NRMSE is the supplied util.py default, RMSE(Cantera,NN)/RMS(Cantera), raw ratio,
equal sample weights, full horizon only. No failed trace is padded. Main summary
uses the explicitly listed common completed-IC intersection of all35 variants;
empty intersection gives N/A. Predeclare a supplemental within-team comparison
using each team's completed-IC intersection across all five lambdas. Different
within-team IC sets must not be used to rank architectures. Export individual
values, medians, means, standard deviations, quartiles, completion counts and
absolute RMSE/reference RMS. Compare unweighted MSE and physical element L1;
weighted totals at different lambdas have different scales.

These test ICs have already been inspected in earlier experiments. This is an
exploratory sensitivity comparison, not independent final validation of a lambda
chosen using its test scores. No training schedule or endpoint selection uses
test outcomes. No augmentation, extra physical penalty or ODE fallback is added.

Tests include exact continuation and finite losses at each new weight in all
four families, identical initialization/minibatch streams, strict provenance,
non-resetting budgets, util.py scoring, failed-trace handling, empty global
intersections and the7x10/six-curve layout. Before reporting verify all120 actual
Adam iterations, exact preprocessing against matched lambda10 members, source
hashes and every preexisting result file. Reports and per-mode ZIP bundles are
immutable, and no new commit/push is part of this training request.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=-1 \
MPLCONFIGDIR=/tmp/kinetics-mpl XDG_CACHE_HOME=/tmp/kinetics-cache \
/workspace/kinetics-venv/bin/python -m kinetics.penalty_sweep_study \
  --config configs/penalty_weight_sweep.json --phase prepare
```

Use the same environment with --phase run --resume after tests. No prior output
is overwritten, and an identical resume cannot create a new compute budget.
