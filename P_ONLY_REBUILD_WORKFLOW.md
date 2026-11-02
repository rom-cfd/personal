# Rebuilt P-only transition model

`PressureDeepONet` is a new implementation on `bw`, not recovered weights from
the inaccessible earlier runtime. Original ANN/T/P code, data and results are
preserved. It represents a finite-state DeepONet-style transition at fixed 2 us,
not an arbitrary-function operator-learning result.

The branch receives **current T and Y9**, standardized with the existing
train-only Box–Cox gamma=.1 pipeline. The trunk receives **log(P) only**.
Branch 10→64→64→70, trunk 1→32→7, component-wise 10×7 contraction and ten
biases give **9,719 parameters**. Hidden activations are tanh. Branch coefficients
and output bias start at zero; the trunk starts nonzero. Both networks receive
nonzero gradients after the initial coefficient update.

All preprocessing/target values match the ANN/T/P baseline exactly. T is T/1000;
Box–Cox is Y**.1/.1 with the existing inverse-domain rules. Pressure is log(P).
The output is the common transformed-coordinate residual for next T,Y9.
No augmentation, new species order, new dt, or test-based fitting is introduced.

## Current scope

Default preparation verifies the existing 90×5001 trajectories, 54/18/18 split,
8,815 fixed-2us train pairs, and the first ten ANN provenance/preprocessing values.
It saves **one untrained seed11 initialization**, config, data/source hashes,
package versions, Git SHA/status, thread settings and planned seed list.
`prepared.json` explicitly records update=0 and `prepared_untrained`. This is
not a ten-member trained ensemble or a new measured performance result.

```bash
cd /workspace/personal
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=-1
export MPLCONFIGDIR=/tmp/kinetics-mpl XDG_CACHE_HOME=/tmp/kinetics-cache
/workspace/kinetics-venv/bin/python -m kinetics.p_only_study --phase prepare
```

Config: `configs/p_only_rebuild.json`. Output: `results/p_only_rebuild/run_001`.
Existing output paths are refused unless `--resume` is explicit and the full
config/data/ANN/code fingerprint matches. Copy the config to select a new run.
Historical T/P manifests fingerprint every kinetics/*.py, so adding new modules
changes the directory-wide fingerprint even though original bytes are intact.
Do not run historical training/evaluation `--resume` against those old results;
new experiments require their own output/provenance. No old manifest is edited.

## Ready-to-run first training, pending an agreed budget

The new runner is limited to the first **10×10,000** updates and never starts
training by default. Seeds are 11,22,…,110; batch512 with replacement; Adam;
global gradient-norm clip5. LR uses absolute updates 1–6000=.001,
6001–9000=.0003, 9001 onward=.0001, so resumed/extended targets do not move the
boundaries. There is no validation checkpoint selection or plateau selection.

After the user agrees to the positive wall-time budget, substitute that number
for SECONDS:

```bash
/workspace/kinetics-venv/bin/python -m kinetics.p_only_study \
  --phase train --resume --budget-seconds SECONDS
```

The first train invocation records a fixed absolute expiry. Reconnects and idle
time count against this conservative window; subsequent invocations cannot
reset it or change its duration. Expired runs stop. Budget checks occur between
kernels/batches; checkpoint flush can finish after the deadline. This is not an
operating-system hard kill. A later budget extension would need an explicit new
workflow/user decision; it is not implemented as a silent resume override.

Each member owns its Adam slots/iteration, NumPy minibatch Generator and a saved
TensorFlow Generator. There are no stochastic TF operations/dropout in training;
ambient NumPy/Python/TF random streams are not used for optimizer updates.
Every 1000 updates the runner records full-train residual MSE against float64
transformed targets, distinct from the logged float32 minibatch loss. Losses and
LR observations are saved without fabricated intermediate values.

Checkpoints are new immutable directories with model/preprocessing, Adam arrays,
RNG states, update, LR, histories, elapsed time, disabled plateau monitor state,
config/data/source/version hashes and stop reason. A `complete.json` hash marker
is written last. Partial directories remain preserved and unregistered. Resume
checks all files, optimizer layout/iteration and provenance. Completed members
point to the verified model/checkpoint in individual `members/seed*.json` files.
These are new exports with real Adam/RNG state, unlike old weights-only files.

## Verification and remaining work

Tests cover parameter count and exact common preprocessing, current-state input
partition, both-network learning, save/load, seeded independence, actual
fresh-process checkpoint resume with array-equal weights/Adam/next minibatch,
corruption and data/config mismatch rejection, fixed LR boundaries, budget expiry
without reset, and an actual P-only member in same-state mixed20 feedback.

```bash
/workspace/kinetics-venv/bin/python -m pytest -q
```

Research training, validation/test comparisons, full time profiles, long-training
plateau policy, and performance benchmarking remain separate tasks. MixedMean
already accepts this model as family `deeponet`, with the original raw/repaired
candidate and mean rules. Use this loader for inference:

```python
from kinetics.p_only_deeponet import PressureDeepONet
p = PressureDeepONet.load('results/p_only_rebuild/run_001/initial_untrained_seed11')
# This example loads an UNTRAINED initialization, not a trained chemistry model.
next_T_Y9 = p.predict(x)  # x: current T,Y9,P; next state is 2 us ahead
```

No commit/push or published environment change is implied by local files.
