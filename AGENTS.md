# Development workflow

After the initial user-requested publication to `main`, use `kisung` for subsequent
work. Switch to `kisung` before editing; do not push subsequent changes directly
to `main` unless the user explicitly asks. Use the existing checkout, not a new
worktree, unless the user requests one.

Run Python via `/workspace/kinetics-venv/bin/python` in this cloud environment.
Read README.md and the report relevant to the experiment before changing it.
Keep original datasets/results immutable; write new experiments to new output
paths. Fit preprocessing and sampling decisions on training/validation only.

## Cantera versus NN reporting

The user wants every case comparison delivered as a matched pair: the established
7-row x 10-variable plot and a Cantera-versus-NN NRMSE table for the same IC,
horizon, physical correction mode, model checkpoints and short/long variants.
Provide IC-level tables and a summary over an explicitly listed common set of
completed ICs; include completion counts instead of comparing different IC sets.
The user supplied `references/user_nrmse/util.py` and explicitly requests its
`_nrmse_1d(A, B, norm="rms")` default: A=Cantera, B=NN. Use
sqrt(mean((A-B)**2)) / sqrt(mean(A**2)), with equal weight for every aligned time
point, and show the raw dimensionless ratio (no percent multiplier). This
supersedes the earlier range-normalization assumption. Preserve the uploaded
function's constant/zero-reference behavior: identical constants return0;
nonzero constant references, including N2, are valid; zero-RMS references with
nonzero predictions have undefined NRMSE and should be marked N/A. Keep absolute
RMSE and reference RMS in exports. Incomplete predictions have no full-horizon
NRMSE; do not pad failed traces or substitute prefix errors. Summary tables
currently show the median of IC-wise NRMSE; export mean/std/count and individual
values separately. Keep this definition unless the user specifies another one.
