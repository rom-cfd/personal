# Pure-MSE Box-Cox gamma screening

Use the existing `bw` checkout, interpreter `/workspace/kinetics-venv/bin/python`,
and unchanged dependencies. Original datasets, sources and previous results stay
immutable. New outputs are below `results/gamma_sweep/run_001`.

The user requested removal of soft constraint and variation of the Box-Cox
exponent. Fixed candidates: .05, .1, .2, .3, .5 and1. Gamma1 is linear species
scaling. Gamma0 would require a separate policy for zero species and is excluded.
Reuse the40 unconstrained gamma.1 models/test traces recorded under
`results/soft_constraint_10k/run_001/run.json:baseline_members` (these are the
lambda0 controls, not its lambda10 models). Train the other200 models from
scratch for up to10000 updates each, with the same150-second model time limit
as the previous study: four families, ten paired seeds11..110 per gamma.

Keep8815 selected train pairs, fixed2us targets,10ms horizon, architecture,
seed-specific initialization/minibatch streams, Adam/LR boundaries/batch512/clip5
and raw/repaired inference unchanged. Each mixed team uses its gamma's ANN10
and corresponding DeepONet10 at the same current state. No soft objective,
signed loss inverse or penalty term enters the optimizer.

The unchanged legacy constructors initialize the original architecture at.1;
the new factory then applies the requested gamma and analytically refits all
state/input standardizers on the same training pairs BEFORE any optimization.
For phi models, initial-phi normalization stays fitted to training inputs and
phi stays fixed during rollout. Original predictor modules are not edited.
Their loaders already reconstruct the exponent from saved metadata.

New gamma checkpoints carry weights, fitted scales, Adam slots/iteration,
NumPy/TF RNGs, histories, update, absolute LR and data/config/source hashes.
Resume preserves the original wall clock. Training, diagnostics, evaluation
and plotting run sequentially with oneDNN0, CPU intra2/inter1, BLAS/OMP2.
The detached stdlib supervisor owns a lock and enforces the total clock; app
closure does not attach the worker to the interactive launcher. Stopping or
deleting the cloud environment still ends its processes.

The user authorized the same per-model time limit and requested commit/push
before starting. The previous run allocated18000 training seconds to120 models,
about150 seconds each (observed member times129.9–164.0s at operation boundaries).
Give every new model its own150-second clock, starting before construction.
Stop at10000 updates if reached first. Persist the first member start/deadline;
resume does not reset it. A time-limited endpoint is saved with its actual
update count and the next model proceeds. Never label a capped endpoint10k.
Checks occur at optimizer-block/measurement/checkpoint boundaries, as before;
record actual wall times rather than promising hard kernel preemption.

The outer watchdog is derived from200×150s plus600s training overhead, a3-hour
evaluation guard and30min reporting guard:43200s in total. This guard does not
extend an individual model's150s allowance. Each phase also enforces its own
ceiling. Completed members return their unused time; no extra updates or
long-training stage is added. Training generally finishes much earlier because
10000 updates are a ceiling. Do not divide a global six-hour pool by200 models.

Freeze all240 endpoints (200 new+40 control) before evaluation. Run all validation
arms first; select each team's representative gamma by repaired10ms completion
count, then median temperature NRMSE on common ICs among maximum-coverage ties.
Exact score ties choose the lower exponent. Save `validation_selection.json`
before running any new test arm. Report all10 variables; temperature is the
declared selection metric, not a combined accuracy score for all species.
Previously inspected test ICs remain an exploratory comparison.

Gamma changes both transformation and training-target scales. Raw transformed
MSE magnitudes across gammas are not physical-accuracy rankings. Learning curves
show full-train MSE divided by each member's initial MSE; raw values are exported.
Historical gamma.1 ANN group time is distinct from fresh serial training time.

Plots contain7 rows×10 variables and Cantera+six gamma curves for every IC.
Each profile is followed by the exact matched NRMSE table. User util.py RMS
normalization, incomplete-horizon exclusion, explicit common-IC medians and
completion counts remain unchanged. Export val/test, raw/repaired; within-team
comparisons across all gammas; and test comparison of each validation-selected
gamma with the .1 reference on pairwise common completed ICs. No gamma is chosen
using test losses, completion or plots.

Prepare only after final tests and successful commit/push:

```bash
TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/gamma-sweep-mpl \
XDG_CACHE_HOME=/tmp/gamma-sweep-cache \
/workspace/kinetics-venv/bin/python -m kinetics.gamma_sweep_study --phase prepare
```

Launch a unique attempt with `python -m scripts.run_gamma_sweep_supervisor`,
`start_new_session=True`, stdin `/dev/null`, stdout/stderr to its console log.
The user has requested publication of all work through this preparation,
including the completed weight-long results. Publish to `origin/bw` before
starting production. Subsequent gamma outputs are new local results until the
user requests another publication.
