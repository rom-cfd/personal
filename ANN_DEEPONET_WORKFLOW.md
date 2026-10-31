# ANN10 / T/P-DeepONet10 / mixed20

Implementation on `bw`. The first ten-member DeepONet training run is complete
(details below). The user has now requested a checkpoint commit/push followed by
the planned comparison evaluation. This checkpoint contains the code, tests,
configuration, ten trained models and training graphs. No package or environment
changes were required.

## What is compared

- `ann10`: the first ten audited MLPs from `results/committee100_long10ms`.
- `deeponet10`: ten independently initialized temperature/pressure-query branch/trunk models.
- `mixed20`: the exact same ten ANN and ten DeepONet members, all evaluated at
  the **mixed team's own current state** every step. This is not an average of
  two independently evolved traces, nor an average of model weights.

Every team has its own self-fed trajectory. Cantera supplies only the initial
state to a rollout and the reference for evaluation. Fixed dt remains 2 us;
the full horizon remains 10 ms / 5000 transitions. There is no medoid, majority
vote, reference-state replacement or chemical ODE fallback.

For repaired mode, all candidates get the same decoder guard, nonnegative
element projection and current-enthalpy/constant-pressure temperature recovery.
Valid candidates are averaged equally, then repaired again. If some are invalid,
they are excluded as whole rows; events log `valid_ann`, `valid_deeponet`, and
`valid_count`. Thus the effective family ratio may depart from 10:10. No valid
candidate or a failed final repair stops that IC, without fabricating a trace.
Conservation imposed by repair is not a learned property or proof of accurate
chemistry. Entropy is diagnostic, not a candidate acceptance criterion.

Raw mode has no guard or repair and requires every member to have a finite
decoded prediction. It is a separate self-fed rollout. Invalid raw compositions
are not normalized to manufacture successful thermodynamic diagnostics.

## Model and fairness

The original MLP is 11-64-64-64-10 tanh, 9738 parameters. The new model is:

- branch: standardized current Y9 (9 values), 64-64 tanh then 70 linear
  coefficients, reshaped into 10 components x 7 latent coefficients;
- trunk: standardized current temperature and log-pressure (2 values), 32 tanh
  then 7 linear thermodynamic basis values;
- ten residual outputs: component-wise coefficient/basis inner products plus
  ten biases. Total: 9687 parameters per member.

The trunk is nonzero-initialized, the branch coefficient layer and output bias
are zero-initialized. Initial residuals are zero, without freezing both factors.
Pressure is constant within an IC, variable across ICs; trunk temperature is the
current temperature at each step, not the initial case temperature. This is a finite-input
DeepONet-style parameterized transition model, not a claim of learning an
operator on arbitrary input functions or a freely queried temporal solution.

Both families use the existing train-only Box-Cox gamma=.1 transformation,
output/input standardizers, transformed residual targets and all ten output
components. The 8815 selected pairs always target the next 2-us state. Split is
54/18/18 whole ICs. Seeds are 11,22,...,110. Per member: 10000 Adam updates,
batch 512 sampled with replacement, lr .001/.0003/.0001 at 60%/90% boundaries,
global norm clipping 5. DeepONets train independently with separate RNG/Adam
states. There is no validation checkpoint selection or test-driven tuning.

The ten old ANN models are reused only after checking the dataset/model hashes,
recorded update budget/configuration and re-fitted training-only preprocessing.
Their `.keras` and preprocessing JSON are read-only. Both families represent
100000 member-updates; only the DeepONet family requires new training. The
mixed team adds inference cost, not new trained members. Historical allocated
ANN group time is labeled separately from newly measured DeepONet training time.

## Commands and safeguards

Use the existing checkout `/workspace/personal` and `bw` tracking `origin/bw`.
Do not make a worktree unless requested. Do not commit/push/merge to main,
delete or overwrite remote branches, or force push. Commit/push only when the
user requests it. Never delete files without explaining the target/reason and
obtaining explicit confirmation.

Use `/workspace/kinetics-venv/bin/python` (Python 3.12), CPU TensorFlow 2.20.0 and
Cantera 3.2.0. Commands below assume:

```bash
cd /workspace/personal
export MPLCONFIGDIR=/tmp/kinetics-mpl
export XDG_CACHE_HOME=/tmp/kinetics-cache
export CUDA_VISIBLE_DEVICES=-1
```

The default command only audits sources and prepares a new run manifest:

```bash
/workspace/kinetics-venv/bin/python -m kinetics.ann_deeponet_study --phase prepare
```

It never trains by default. Output must be a new run below
`results/ann_deeponet_mean10/`; the supplied configuration uses `run_001`.
If it exists, the command refuses to overwrite it. Copy the configuration and
choose a new run name, or use `--resume` only for a run made by this runner.
Resume requires identical code, configuration, data, reference and ANN hashes.
Old study/report scripts must not be pointed at these run directories.

After a **separate user-approved compute budget**, execute the phases below,
replacing each `SECONDS` token with that phase's approved positive wall seconds.
These are future execution instructions, not a request to run them now.

```bash
/workspace/kinetics-venv/bin/python -m kinetics.ann_deeponet_study --phase train --resume --budget-seconds SECONDS
/workspace/kinetics-venv/bin/python -m kinetics.ann_deeponet_study --phase evaluate --split val --resume --budget-seconds SECONDS
# Freeze all design choices using validation, then request final test execution:
/workspace/kinetics-venv/bin/python -m kinetics.ann_deeponet_study --phase evaluate --split test --resume --budget-seconds SECONDS
/workspace/kinetics-venv/bin/python -m kinetics.ann_deeponet_study --phase report --resume --budget-seconds SECONDS
```

Budget checks apply during training, rollout, one-step batches, diagnostics and
reporting. They stop at operation boundaries, not by preempting a TensorFlow
kernel or file write mid-operation. Execution history records each phase's
actual wall time; `--budget-seconds` applies per invocation, not cumulatively
across repeated invocations. Do not restart a phase to evade an approved total
budget. Prior experiments' 90-minute permission does not authorize new runs.

Only complete saved members are resumed; optimizer state is not restored for
an interrupted partial member. Existing unregistered model files or partial
evaluation directories are preserved and cause an error rather than overwritten.
Completed evaluation hashes are checked before skipping them on resume. Reports
always use a new directory and clearly identify missing/unrun evaluations.

## Saved evidence

Run manifests include configuration, current Git HEAD/status, dependency versions,
data/model/preprocessing/source-code hashes, and reference ignition hashes.
No tokens, credentials or process environment dumps are included.

For each of three teams, raw/repaired mode, and requested val/test split:

- full autoregressive traces for every IC, valid-family counts and failure events;
- 2-ms and 10-ms trajectory metrics, temperature/species RMSE, coverage,
  physical validity, ignition classification/delay and unresolved cases;
- one-step errors on every dense pair of that split (90000 pairs), chunked,
  including per-species error and invalid candidate fraction;
- mass/element drift, enthalpy drift, entropy decrease diagnostics, final
  temperature/reference/equilibrium differences and min/max temperature;
- warm-up, common model loading, neural/decode, candidate/final correction,
  rollout and one-step times. Common loading time is not charged three times.

Incomplete traces have no full-horizon RMSE; prefix errors remain labeled.
Existing 0.2-us ignition references are read-only and are checked against ICs,
resolution and hashes. They do not change the model's 2-us timestep.

Reporting writes a summary CSV, training-cost metadata and all-IC temperature
and nine species figures. A short line ends where its prediction failed; no
padding or replacement with Cantera is performed. Plots can be produced from
validation alone and must not be described as final test results.

## Verification and handoff

```bash
/workspace/kinetics-venv/bin/python -m pytest -q
/workspace/kinetics-venv/bin/python -m pip check
```

Tests exercise parameter count/preprocessing parity, both-network gradients,
current-temperature/pressure partition and sensitivity, save/reload, independent seeded training, raw/repair
rules, same-state mixed feedback, old ANN correction equivalence, data leakage,
output preservation, deadlines and a tiny three-team evaluation/report. Gradient
tests use a few updates on synthetic data, not research training.

Main sources: `kinetics/deeponet.py`, `kinetics/mixed_ensemble.py`,
`kinetics/ann_deeponet_study.py`, `kinetics/ann_deeponet_report.py` and
`configs/ann_deeponet_mean10.json`.

Base HEAD at implementation start: e89c656eeeac0e93bafb4322b9482c7dfec70168.
Training loss is available; no validation/test comparison results exist yet.
Remaining work: run validation/frozen test under the approved scope/budget,
generate comparisons. The checkpoint commit includes the new code and completed
training run; use `git log` and the reported remote SHA to identify it. Future
evaluation outputs are separate from that checkpoint. Git push does not preserve
untracked/ignored outputs automatically; explicitly include or separately archive
needed models/results when handing off to a new environment.

## Completed first training run

Run: `results/ann_deeponet_mean10/run_001`. All ten seeds completed exactly
10000 updates, batch 512, with 9687 parameters per member. Branch receives
current Y9; trunk receives current T and log(P). The learning-rate schedule was
.001 for updates 1-6000, .0003 for 6001-9000, .0001 for 9001-10000. Each history
records update 1, every 50 updates and the first updates after LR changes
(203 observations per member).

Actual command used:

```bash
/workspace/kinetics-venv/bin/python -u -m kinetics.ann_deeponet_study --phase train --budget-seconds 1200
```

Training phase wall time was 210.572 seconds; sum of member training times was
207.624 seconds. The tool connection was interrupted, but the training process
continued and all ten saved members were verified after reconnection. Do not
restart this training as a new run just because a tool connection was lost.

Models and preprocessing are in `run_001/models/`, registered with hashes in
`run_001/deeponet_members.json`. Source fingerprints still match. All 799
pre-existing result files were rehashed and unchanged.

Training reports can be reproduced without retraining:

```bash
/workspace/kinetics-venv/bin/python -m kinetics.ann_deeponet_training_report --root results/ann_deeponet_mean10/run_001 --budget-seconds 120
```

The completed report is
`results/ann_deeponet_mean10/run_001/training_reports/report_1793476697862741811/`.
It contains `training_loss_and_lr.png`, `training_loss_per_member.png`,
`observed_history.csv`, `summary.csv`, `REPORT.md`, and output hashes.

The final full-training-set transformed residual MSE across members ranges
from 1.79646e-4 to 2.42370e-4 (median 2.24369e-4). The zero-residual baseline
on the same train data is 1.44309e-2. These are training diagnostics, not
validation/test errors or evidence of accurate 5000-step chemistry rollout.
No `run_001/evaluations` directory exists yet. The next step is separate
validation of ANN10, DeepONet10 and mixed20, followed by frozen test evaluation.
The training snapshot records its original base Git HEAD. A later checkpoint
commit can preserve these files without changing their source/model fingerprints.

## Figure style and reproducible plotting

`scripts/plot_ann_deeponet.py` uses a white background, blue/orange/green method
colors, dotted major/minor grids and separate method/statistic legends. A solid
line is the geometric mean, dashed is max, dotted is min and shading is the
observed range (not a confidence interval). Training statistics are across ten
members at actually observed common updates. Error statistics are across available
ICs at each time; coverage is shown and failed prefixes are never padded.
Zero values are omitted on logarithmic axes, not replaced by invented errors.

```bash
/workspace/kinetics-venv/bin/python scripts/plot_ann_deeponet.py --kind training
# After evaluation completes:
/workspace/kinetics-venv/bin/python scripts/plot_ann_deeponet.py --kind comparison
```

Both commands create a new `run_001/plots/` directory with PNG/PDF files,
the plotted statistics CSV and hashes of the plotting script and outputs.
