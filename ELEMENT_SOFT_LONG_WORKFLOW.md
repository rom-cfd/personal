Continuation of all40 soft-mu10 models, maximum six hours
=======================================================

The user requested long training for every soft-constraint NN and added these
curves to the existing comparisons. They explicitly set the whole training,
evaluation and reporting cap to six hours. Duplicate copies of that answer do
not grant a second budget. The run continues the40 existing10000-update models,
not freshly initialized networks: ANN, P-only, T/P, T/P/initial-phi,10 seeds each.
Origin weights, Adam slots/iteration, NumPy/TF RNGs, preprocessing, histories,
mu10 and absolute LR schedule are all preserved. Beyond update9000 LR remains
.0001. The signed inverse extension is confined to the same original soft loss;
actual inference, strict inverse validity and raw/repaired physics are unchanged.

New modules use the `element_long_` prefix so that the frozen10k trainer's
`soft_constraint_*.py` source signature is not changed by adding continuation
tools. Do not edit the original soft trainer to make its old states load.
`SoftTrainer.restore` still verifies the complete origin/runtime/config signature.
Each continuation checkpoint additionally binds its plateau monitor to the
immutable new run snapshot. Full checkpoints still have the same six-file
format with a completion marker written last; none are overwritten.

Stopping now requires three components simultaneously: total loss, data MSE,
and physical element L1. For each component use full-train observations every
1000 updates, checks every5000, at least50000 total updates, two adjacent20000
windows with20 actual observations each, relative q25/median/q75 changes no more
than1%/1%/3%, and three consecutive passing checks spanning10000 updates.
The full state histories before10000 are retained, but no observations are
invented for sparse windows. A flat total with cancelling component trends does
not pass. Missing/nonfinite observations reset the relevant stream. No test
performance is used for stopping, LR changes, mu changes or endpoint selection.

The six-hour absolute clock starts at `--phase run` after implementation tests.
Forty minutes are reserved for frozen endpoint diagnostics, evaluation, figures
and file verification. The remaining19200 seconds are divided fairly among the
remaining models in the sequence ANN10 → P-only10 → T/P10 → T/P/phi10. Each
normally gets about480 seconds if none reaches plateau early. Earlier plateau
frees time for later members; any unused training time can revisit capped models
without resetting state. No extra update ceiling is imposed. Reasons are joint
plateau or time limit, and are never relabeled as convergence. Resume keeps the
original absolute expiry and cannot create a new six-hour budget.

Train, evaluation and plotting stay sequential with CPU TensorFlow intra2 /
inter1, OMP/OpenBLAS/MKL2, oneDNN0. Reuse frozen no-constraint10k and soft10k
control traces and their train/validation diagnostics. Evaluate only the new
long endpoints: seven teams, raw/repaired, on the18 previously observed test ICs,
plus full train/validation per-member diagnostics after endpoints are frozen.
Mixed teams use ANN10 and DeepONet10 from the corresponding stage, so long
mixed teams include the long ANN models too. No new training samples, Cantera
references, preprocessing, inference guards or ODE fallback are introduced.

Graphs retain7 model rows x10 variables and now show four curves: Cantera,
unconstrained10k, soft10k and soft-long. Each IC's graph is followed by its NRMSE
table; raw/repaired each have36-page PDFs, PNGs and CSVs. NRMSE follows user
util.py exactly: RMSE(Cantera,NN)/RMS(Cantera), equal sample weights, raw ratio.
Failed traces are not padded and receive no full-horizon score. Summary medians
use the explicitly listed common complete-IC intersection of all21 variants,
with completion counts; an empty intersection is N/A. Include all observed
individual loss tails and summaries only at common observed updates.

Before reporting recheck original-result hashes, terminal model/Adam/RNG markers
and replay every joint plateau check. All new outputs are below
`results/element_soft_long`, with independent run/environment/budget snapshots.
Implementation tests go below `results/element_long_checks`. New commit/push is
not authorized by this training request.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=-1 \
MPLCONFIGDIR=/tmp/kinetics-mpl XDG_CACHE_HOME=/tmp/kinetics-cache \
/workspace/kinetics-venv/bin/python -m kinetics.element_long_study \
  --config configs/element_soft_long.json --phase prepare
```

Use the same environment with `--phase run --resume` after preflight. Original
data/results must remain immutable; use a new output for a new run. Historical
unconstrained long runs had different time budgets/update counts, and should
not be presented as a controlled architecture/time comparison here.
