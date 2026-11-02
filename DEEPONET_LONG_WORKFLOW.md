# Sequential long-training and five-team comparison

This new study preserves every existing model/result. Work remains on `bw` in
the existing checkout. User authorization covers both families' long training
and evaluation, with a **six-hour total window**, all operations sequential.
No commit/push or environment publication is included.

P-only starts from the new `results/p_only_rebuild/run_001` 10k checkpoints with
actual Adam/RNG continuation. T/P starts a new reproduction from initialization,
because the saved historical 10k files lack Adam/RNG state. Old T/P weights and
results are preserved. The same ten audited ANN members are reused for every
comparison. Data, train-only preprocessing, gamma=.1, species order, 2 us dt,
10 ms horizon, batch512 with replacement, global norm clip5 and absolute
6000/9000 LR boundaries remain fixed. LR after 10k is .0001.

The train-only plateau monitor observes full-training residual MSE every 1000
updates. Starting at 50k, it checks every 5000 updates. At update u it compares
the observed values in `(u-40000,u-20000]` and `(u-20000,u]`, requiring 20 actual
observations per window. Relative changes in q25/median/q75 must be at most
1%/1%/3% in absolute value. Three consecutive passing checks span 10k sustained
updates. Missing/nonfinite data or a failed check reset the sustained condition.
An old zero quantile and positive new quantile explicitly fail. These thresholds
are a reproducible stopping convention, not proof of mathematical convergence.
No test result is used to stop, select, or tune a model.

The first run fixes an absolute six-hour expiry, retained on resume. Forty
minutes are reserved for evaluation/reporting, including five minutes for the
report itself. Remaining training time is divided among remaining members.
P-only seeds11..110 run first, followed by T/P seeds11..110. Spare time after
the first pass can continue time-limited members. A model can stop for `plateau`,
`time_limit`, or `update_limit`; these are distinct. The update safety ceiling is
1M, four times the incomplete historical 250k attempt, bounded by the agreed
wall-time cap. Capped models are not described as plateaued. Kernel and final
checkpoint writes finish at operation boundaries, so the deadline is not an OS
hard kill. Reconnect/idle time counts against the conservative absolute window.

SerialSteps runs one member's Adam updates in a TensorFlow loop with
parallel_iterations=1. Tests verify array-equal weights, Adam state, and next
NumPy minibatch relative to scalar updates and checkpoint resume for both
families. This reduces Python dispatch overhead without concurrent members.
TF intra/inter threads are 2/1 and OMP/BLAS threads2. Full loss uses the same
float64 transformed residual targets, recorded separately from minibatch loss.
Tracing, RNG generation, optimizer, full loss, checkpoint, and member wall time
are retained. Historical ANN grouped time is not treated as a matched speed test.

Configuration: `configs/deeponet_long_study.json`. New output:
`results/deeponet_long_study/run_001`. All origin/data/source/runtime hashes are
saved before training. Each new checkpoint is immutable with completion hashes;
partial checkpoints remain preserved. Final endpoint records include update,
stop reason, plateau window checks, Adam/RNG state, histories, timing, and origin.

```bash
cd /workspace/personal
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=-1
export MPLCONFIGDIR=/tmp/kinetics-mpl XDG_CACHE_HOME=/tmp/kinetics-cache
/workspace/kinetics-venv/bin/python -m kinetics.deeponet_long_study --phase prepare
# The user authorized this complete six-hour sequential pipeline:
/workspace/kinetics-venv/bin/python -u -m kinetics.deeponet_long_study --phase run --resume
```

Phases train/evaluate/report can also be invoked separately with --resume;
they share the same fixed budget. Existing run provenance must match. Original
P-only/legacy T/P runners are not used to modify old experiments.

The five teams are ANN10, T/P10, P-only10, ANN+T/P20, ANN+P-only20. Every team
feeds its own current state back to all its members at every step. Raw and
repaired are independent experiments; repaired uses the original guard,
candidate composition/element/enthalpy repair, equal valid-row mean and mean
repair. Family valid counts are retained; no ODE fallback, medoid or reference
feedback is introduced. All 18 val and 18 previously observed test ICs use all
5000 transitions. Dense one-step uses all 90k pairs per split.

Evaluation writes separate neural, decode, feature encoding, candidate repair,
mean repair, warm-up, rollout, dense one-step and diagnostics timings. Shared
30-model loading is recorded once. Train/validation residual losses are computed
after endpoints are frozen. Ignition/refined delays, completion, T/Y errors,
mass/element/enthalpy drift and entropy decrease use the existing definitions.
Incomplete traces have no full-horizon RMSE and are never padded. Reports give
completion and common completed IC comparisons, with per-IC average versus
pooled RMSE distinguished. Temperature and nine species panels for every test
IC include the available six curves including Cantera, separately raw/repaired.
PNG/PDF, profile CSVs, full observed training histories, plateau checks and source
hashes are retained. Full individual loss curves remain visible beyond the
common observed updates used in family min/max/geometric-mean bands. Bands are
observed ranges, not confidence intervals. Repair-imposed conservation is not
interpreted as learned conservation or proof of accurate chemistry.
