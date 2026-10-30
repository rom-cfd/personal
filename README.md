# Nine-species hydrogen kinetics preprocessing study

## Full-scale follow-up: 5.3 million pairs and GELU

The current request scales generation to **180,000 independent states at each
of 25 log scales + 400,000 hot initial states with two consecutive labels**:
exactly 5,300,000 pairs. The new architecture is **11–1600–800–400–11, GELU**,
with ten complete shuffled epochs and physics OFF. See [run details](ZHANG_5P3M_RUN.md).
The full-scale run must not be confused with the completed reduced pilot below;
full-scale model results are not yet available.

Generation is sharded, process-parallel and resumable. Large arrays and optimizer
checkpoints stay in the cloud workspace; source, configurations, checksums and
result reports can be published without pushing gigabyte-scale arrays to GitHub.

```bash
python -m kinetics.multiscale_large_data --config configs/zhang_5p3m_data.json --resume
python -m kinetics.large_zhang_training --config configs/zhang_5p3m_gelu.json --resume
```

## Previous reduced study: Zhang multi-scale sampling, physics OFF

The requested follow-up uses independent log-space sampling from Zhang 2022
Section 2.5, not perturbations of manifold states. The NN follows the paper's
full-state increment equations: separately standardized inputs
`[T, P, BCT(Y)]` and targets `delta[T, P, BCT(Y)]`, with L1 loss. The previous
64/64/64 tanh architecture is retained. Temperature and pressure are self-fed
from the NN, with no enthalpy closure, projection, clipping, or renormalization.

The study trains ten members per gamma (.1/.2/.5), plus ten matched raw-I/O
manifold controls at gamma=.1. It assesses both **10 ms and 20 ms** final times.
All test methods have **0/18 physically admissible completions**: longer numerical
survival is not evidence of a valid chemistry integrator. Data size, pressure
coverage, normalization assumptions, and other differences from the paper are
documented in the report.

- [18-profile gallery and training-data coverage](ZHANG_MULTISCALE_GALLERY.md)
- [Full report](ZHANG_MULTISCALE_RAW_REPORT.md)

```bash
/workspace/kinetics-venv/bin/python -m kinetics.multiscale_data --config configs/zhang_multiscale_matched_data.json
/workspace/kinetics-venv/bin/python -m kinetics.zhang_raw_study
/workspace/kinetics-venv/bin/python -m kinetics.zhang_raw_report
/workspace/kinetics-venv/bin/python -m pytest -q --junitxml=results/zhang_multiscale_raw/pytest.xml > results/zhang_multiscale_raw/pytest.log
/workspace/kinetics-venv/bin/python -m kinetics.zhang_raw_verify
```

Repeated runs require new configured output paths. Historical physics-on
experiments and `configs/active_model.json` below remain comparison artifacts;
they are not used by this physics-off runner.

## Per-step minimum physics-correction selection

`kinetics.min_correction_study` reuses the frozen ten-member gamma=.2/.5 tanh
ensembles. At each self-fed timestep it corrects all proposals, then selects the
valid candidate with the smallest train-scaled RMS change in T and nine Y values
from its own ANN proposal. This measures the correction, not the change from the
current state. The existing decoder guard precedes this measurement. No new
training or ODE fallback is used. See [MIN_CORRECTION_REPORT.md](MIN_CORRECTION_REPORT.md)
for comparisons, failure coverage and per-step member-selection figures.

```bash
/workspace/kinetics-venv/bin/python -m kinetics.min_correction_study
/workspace/kinetics-venv/bin/python -m kinetics.min_correction_report
```

Repeat with a new `--output`; give the reporter the matching `--root` and a new
`--report`. Existing weights, datasets and experiment outputs remain unchanged.

## Box–Cox preprocessing ablation: γ=0.2 and 0.5

The `pre-process` experiment returns to the earlier **64/64/64 tanh** network.
It freshly trains ten independent members per gamma on the same immutable
10-ms, train-only downsampled data, with 10,000 updates per member. Wang hard
correction uses the matching gamma. The best of all twenty members by full
validation one-step physical error is frozen as one common single-model control
before rollout/test evaluation; per-gamma best singles are auxiliary controls.
Mean10 and medoid10 run all 5,000 self-fed steps, with no chemical ODE fallback.

```bash
/workspace/kinetics-venv/bin/python -m kinetics.preprocess_study --config configs/preprocess.json
/workspace/kinetics-venv/bin/python -m pytest -q --junitxml=results/preprocess_tanh10/pytest.xml > results/preprocess_tanh10/pytest.log
/workspace/kinetics-venv/bin/python -m kinetics.preprocess_verify
/workspace/kinetics-venv/bin/python -m kinetics.preprocess_report
```

Results: [PREPROCESS_REPORT.md](PREPROCESS_REPORT.md). Repeated experiments need
a new output path in a copied config; `--resume` checks existing configuration,
input/model hashes and frozen selections. Historical experiment outputs remain
unchanged. The separate paper/GELU pilot below is not used for this ablation.

Current active formulation: **next-state Y prediction with Wang hard correction**,
as recorded in [configs/active_model.json](configs/active_model.json). The previous
default remains `medoid100`; mean100 and the validation-selected singleton are
also recorded. The signed physical-increment experiment is retained as an
unsuccessful physics-off comparison in [DELTA_RAW_REPORT.md](DELTA_RAW_REPORT.md).

In the active pipeline, the ANN also has a temperature output, but that output
is discarded during corrected inference. Current T,Y determine mixture enthalpy;
the corrected next Y and this conserved enthalpy at fixed pressure determine
next T. Composition alone does not determine temperature.

The requested 2025-paper alignment is a separate prepared model family:
[implementation audit](WANG_IMPLEMENTATION_AUDIT.md),
[configuration](configs/wang2025_paper.json), and `kinetics.wang_paper_study`.
It inputs T/P/G and outputs only nine direct next-G species values, with
1600/800/400 GELU, raw G-space L1 and the paper's two-stage training schedule.
The current 10-update pilot is undertrained and is not an active model.
Full training requires the agreed compute budget; unpublished author details
are explicit assumptions, so bit-identical reproduction is not claimed.

Latest 100-member / 10 ms study: [LONG_COMMITTEE_REPORT.md](LONG_COMMITTEE_REPORT.md); [training downsampling audit](LONG_DATA_REPORT.md).

Latest committee comparison and literature brief: [COMMITTEE_REPORT.md](COMMITTEE_REPORT.md), [LITERATURE_RESEARCH_BRIEF.md](LITERATURE_RESEARCH_BRIEF.md).

Latest augmentation and 0-D ignition results: [REFINEMENT_REPORT.md](REFINEMENT_REPORT.md).

Cantera 3.2 + TensorFlow CPU 2.20 benchmark for H2/H/O/O2/OH/H2O/HO2/H2O2/N2.
The repository was empty: no original ANN, dataset or Box–Cox implementation was
available. Gamma=0.1 is a new reference baseline, not a reproduced legacy model.

## Reproduce

Python 3.12 is used. Each cloud task is already isolated; use this checkout and do
not create a Git worktree unless explicitly requested.

```bash
python3 -m venv /workspace/kinetics-venv
/workspace/kinetics-venv/bin/python -m pip install -r requirements.lock.txt
/workspace/kinetics-venv/bin/python -m pytest -q
MPLCONFIGDIR=/tmp/kinetics-mpl XDG_CACHE_HOME=/tmp/kinetics-cache \
  /workspace/kinetics-venv/bin/python -m kinetics.study --output results/reproduction
/workspace/kinetics-venv/bin/python -m kinetics.analysis --output results/reproduction
/workspace/kinetics-venv/bin/python -m kinetics.posteriori --study results/reproduction
/workspace/kinetics-venv/bin/python -m kinetics.reference_resolution --study results/reproduction
/workspace/kinetics-venv/bin/python -m kinetics.report --study results/reproduction --report REPRODUCTION_REPORT.md
```

The study refuses existing output paths and never overwrites datasets. To reuse
an exact generated dataset, add `--dataset results/verified/dataset.npz`. The checksum
is checked. Dataset configurations must match; defaults are in `configs/study.json`.
No network or external services are needed after dependency installation. CPU
thread counts are capped; no GPU/HPC jobs are submitted. The default training
budget guard is 1800 wall seconds, checked every epoch (rollout/analysis overhead
is additional). Final comparison has six methods and three seeds each.

`kinetics.transforms.Transform` supports positive and odd signed variants and
explicit inverse-domain failures. For this next-state study only nonnegative
mass fractions enter a transform. The network learns a residual in transformed
coordinates and reconstructs next T,Y; it does not learn physical signed rates.
Temperature is scaled linearly; pressure is log-scaled. All fitted standardizers
and epsilon statistics use training trajectories only. Read the research report
for precise formulas, metrics, limits and measured outcomes.

`Predictor.load('results/verified/models/boxcox_g0.1_seed11')` reloads a fitted model
and preprocessing. Call `.predict(x)` on rows `[T, Y_H2, ..., Y_N2, P_Pa]`.
The returned rows are `[T_next, Y_next...]` at dt=2 microseconds. Raw predictions
are never silently clipped; NaN marks inverse-domain violations. Do not deploy
these exploratory models as a chemistry integrator without further validation.

Results include dataset/split metadata and SHA256, exact configuration, models,
per-epoch histories, validation-only selection records, held-out predictions,
physical metrics, concentration bins, conservation diagnostics, raw and repaired
rollouts, timing, perturbation tests and figures. `results/study` is an interrupted
pilot retained for provenance; `results/final` is superseded by `results/verified`, whose 18 saved models passed reload checks. The final report is [RESEARCH_REPORT.md](RESEARCH_REPORT.md).

A posteriori evaluation is self-fed at fixed 2 us, without teacher forcing. It
compares 0.4 ms and 2 ms full profiles and ignition timing against independently
integrated Cantera reactors. A 0.2 us reference-only diagnostic refines ignition
measurement; training, test and ANN step sizes stay 2 us. Incomplete profiles
are excluded from full-horizon error and separately recorded as failures, with
coverage and physical admissibility reported. In the original unconstrained study, all models fail the 2 ms physical
admissibility criterion; no trained model is ready for use as a kinetics integrator.

## Perturbation refinement: current milestone

Read [REFINEMENT_REPORT.md](REFINEMENT_REPORT.md) for the new study. The original
research results remain in `results/verified`; augmented data/models are in
`results/refinement`, equal-update unaugmented controls in
`results/refinement_controls`, and fixed held-out evaluations in
`results/refinement_test`. Training-only perturbations receive fresh Cantera
labels at the same fixed 2 us timestep. An optional common projection/enthalpy
closure is evaluated separately from raw ANN output.

The agreed validation ignition target was reached across three seeds. The
validation-selected single model uses gamma=.1 and seed 33. On held-out ICs it
classifies ignition/no-ignition correctly in 17/18 cases with median ignition
delay error 13.0%, and physically completes all 18 trajectories over 2 ms. It
misses one long induction case. This is progress on the benchmark, not validation
for use in CFD or evidence that augmentation alone fixes raw ANN instability.

To reproduce without overwriting results, copy `configs/refinement.json`, change
its `output` to `results/repeat_refinement`, and run with the venv interpreter:

```bash
python -m kinetics.refinement --config configs/repeat_refinement.json
python -m kinetics.refinement_controls --config configs/repeat_refinement.json \
  --root results/repeat_refinement --output results/repeat_controls
python -m kinetics.ensemble_validation --root results/repeat_refinement
python -m kinetics.refinement_test --root results/repeat_refinement \
  --controls results/repeat_controls --output results/repeat_test
```

The test runner refuses existing output directories and requires the validation
family to meet the configured ignition target first. Its frozen selection is
written before reading held-out data. `kinetics.refinement_report` regenerates
the current milestone report/figures from the current default result paths.

```python
from kinetics.study import Predictor
from kinetics.physics import PhysicalPredictor
model = PhysicalPredictor(Predictor.load(
    'results/refinement/models/final_moderate_boxcox_g0.1_seed33'))
# x rows: [T_K, Y_H2, Y_H, Y_O, Y_O2, Y_OH, Y_H2O, Y_HO2, Y_H2O2, Y_N2, P_Pa]
# result rows: [T_next_K, next nine Y values], fixed dt=2e-6 s
next_state = model.predict(x)
```

## Ten-member committee on matched-budget original data

This experiment intentionally uses **no augmentation**. Every member retains the
same 64/64/64 tanh architecture, Box–Cox gamma=.1, and 10,000-update budget. Seeds
11/22/33 reuse the matched original controls; seven additional seeds complete ten.
The committee total represents 100,000 updates, not a single-model training budget.

```bash
/workspace/kinetics-venv/bin/python -m kinetics.committee_study --config configs/committee.json
/workspace/kinetics-venv/bin/python -m kinetics.committee_report
```

Existing result directories are refused. For reproduction, copy the JSON and
change its output path; source data and three existing control models are reused.
Medoid distance is selected using validation only, then frozen before test.

```python
from kinetics.committee import Committee
model = Committee.load('results/committee_original/methods/medoid10_scaled.json')
next_state = model.predict(x)  # x: [T_K, Y9, P_Pa], output: [T_next_K, Y9_next]
representative_seeds = model.last_diagnostics['selected_seed']
```

This object already performs physical correction; no extra wrapper is required.
Every member receives the same current state. An actual corrected candidate is
chosen, not a coordinate median. There is no ODE fallback or hysteresis. If no
candidate is valid, the output is NaN and the rollout fails explicitly.

The ten-member mean and medoid share the same candidate correction for the main
comparison. Legacy averaging-before-correction is a separate arm. On this test,
the ten-member mean **after candidate correction** outperformed medoid selection.
Read the report for full profiles, failures, switching, budgets and limitations.

## 100 members and 10 ms: current run

Fresh Cantera trajectories extend training, validation and test to **10 ms**.
Sampling thins training INPUT TIMES only; every target stays exactly **2 us**
ahead. Of 270,000 candidate training transitions, 8,815 are retained, including
376 equilibrium anchors (4.3%, versus 80.5% in the dense training pool). Slow
induction is protected by comparison with HP chemical equilibrium, not merely a
small dT/dt. Validation/test autoregressive profiles keep every 2 us sample.

All 100 models are newly trained on this data, with the same 64/64/64 architecture,
Box–Cox gamma=.1 and 10,000 updates per member. The first ten form the nested
same-data control. Previous 0.4 ms models remain unchanged for historical study.

```bash
# Each command uses /workspace/kinetics-venv/bin/python in the cloud environment.
python -m kinetics.long_data --output results/long10ms_data
python -m kinetics.long_data_report
python -m kinetics.long_committee_study --config configs/committee100_10ms.json   --budget-seconds 5400
python -m kinetics.long_committee_report
python -m kinetics.verify_long_committee
python -m pytest -q
```

This run had explicit approval for up to 90 CPU minutes. The runner defaults to a
30-minute cap; do not increase a task's approved compute budget without approval.
Existing data outputs are refused; copy configs and use new paths to reproduce.
`--resume` validates source/config/model hashes and continues completed groups.
`--phase train --train-limit 8` supports an inexpensive first training group.

Independent models are computationally batched for training; gradients, norm
clipping, minibatch streams and Adam moments stay separate. The vectorized
physical backend uses the mechanism's NASA-7 polynomials and the same element
projection/enthalpy equations, validated against Cantera/scalar projection. It
never integrates the chemical ODE at inference. Numerical scalar/thermodynamic
fallbacks are distinct from an ODE fallback.

```python
from kinetics.committee import Committee
model = Committee.load('results/committee100_long10ms/methods/medoid100.json')
next_state = model.predict(x)  # x rows: [T_K, Y9, P_Pa]; fixed dt=2e-6 s
selected_seed = model.last_diagnostics['selected_seed']
```

Reports include full and regime-specific profiles, ignition tails, failed versus
censored outcomes, conservation, and entropy diagnostics. Increasing member count
does not guarantee lower shared bias or long-time profile error.

After initial publication to `main`, subsequent development uses `kisung` as
requested by the user. See AGENTS.md.

## Wang ANN-hard correction on the same trained members

The paper's Box–Cox tangent-space projection (Eqs. 5–9) is available as
`physics_backend='wang_boxcox'`. It replaces the previous nonnegative weighted
Y-space QP; the common decoder guard and fixed-pressure enthalpy recovery remain.
Negative or element-inconsistent corrected candidates are rejected without
clipping or an old-QP fallback. This is a frozen-weight correction comparison,
with no retraining or test-driven parameter tuning.

```bash
python -m kinetics.wang_study --output results/wang_hard10ms
python -m kinetics.log_time_profiles --root results/wang_hard10ms
python -m kinetics.wang_report
```

Use a new output path for repeated experiments. Results and limitations are in
[WANG_HARD_REPORT.md](WANG_HARD_REPORT.md). The plotting command excludes t=0
from log-axis display only; full-time error metrics retain the original data.

## Best a priori single member with the same Wang correction

`single_apriori_study` ranks all 100 frozen models on all 90,000 validation
one-step pairs using the existing physical-unit score, then freezes one seed
before any rollout or test assessment. The single-member runtime calls only
that network at every step, with the same Wang correction and no ensemble/ODE
fallback. Previously saved mean100/medoid100 traces are the controls.

```bash
python -m kinetics.single_apriori_study
python -m kinetics.single_apriori_report
```

Read [SINGLE_MODEL_REPORT.md](SINGLE_MODEL_REPORT.md) for selection details,
complete/failing trajectories, ignition and full-profile comparisons. New runs
require a new `--output`; the report accepts `--root` and `--report`.

## Signed physical increments with physics disabled

The new target is `signed_boxcox(Y_next - Y_now)`, distinct from the older
`boxcox(Y_next) - boxcox(Y_now)`. The signed transform is
`sign(v) * abs(v)**0.1 / 0.1`; temperature increments are linearly standardized.
All preprocessing statistics use training data only. The DNN predicts both
delta T and delta Y from (T,P,Y), then adds them to the current state.

```bash
python -m kinetics.delta_training --config configs/delta100_raw.json
python -m kinetics.delta_evaluation --root results/delta100_raw
python -m kinetics.delta_target_diagnostics --root results/delta100_raw
python -m kinetics.verify_delta --root results/delta100_raw
python -m kinetics.delta_report --root results/delta100_raw
```

All 100 members are freshly trained with the previous architecture and member
update budget. Single-member selection uses validation one-step physical error
only. Raw mean/medoid include negative finite candidates; there is no clipping,
normalization, element/enthalpy projection or chemistry fallback. Negative states
continue to be fed back and are counted as violations. Numerical termination is
separate from physical admissibility. Training/evaluation share a 30-minute CPU
cap; no GPU/HPC is used. Prior datasets, weights and results remain unchanged.
See [DELTA_RAW_REPORT.md](DELTA_RAW_REPORT.md) for results and limitations.
