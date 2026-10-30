# Full-scale Zhang sampling and GELU training

**Status: data generation in progress; full-scale training/evaluation is not yet
complete.** Do not use the reduced pilot's results as full-scale results.

The user requested the paper's 5.3-million-pair scale, ten complete epochs, and
the previously prepared large paper-model architecture including GELU. Physics
remains OFF; evaluate both 10 ms and 20 ms after training. CPU/GPU and ensemble
scheduling must follow the user's latest resource preference.

## Dataset

- 25 log scales, k=1…25, **180,000 independent initial states per scale**:
  4,500,000 one-step pairs.
- **400,000 hot initial states**, each with two consecutive one-step labels:
  800,000 pairs. Total exactly **5,300,000 pairs / 4,900,000 initial states**.
- No perturbation of old trajectories; all labels are independently integrated
  with Cantera CVODE, rtol=1e-10, atol=1e-20.
- Preserve the preceding study's 9-species/28-reaction mechanism, dt=2 us,
  pressure extension 0.5–5 atm and explicit N2-preserving simplex normalization.
  Those remain differences/assumptions relative to the original paper.
- No invented HRR filter; HRR is recorded. Existing finite-label/temperature
  support checks remain and any rejected candidates are counted/replaced.

Configuration: [zhang_5p3m_data.json](configs/zhang_5p3m_data.json).
Generator: [multiscale_large_data.py](kinetics/multiscale_large_data.py).

Generation uses four CPU workers and 2,450 independently seeded shards.
A completed shard is registered only after an atomic data write and SHA256.
Its configuration and generator source digest are checked before reuse.
A process lock prevents duplicate managers. Each shard starts a new ReactorNet;
within a shard the documented Cantera state-reset API avoids repeated setup.
Reset-solver labels are tested against fresh integrations, and every shard
checks mass and elemental conservation. No approximate chemistry solver is used.

```bash
cd /workspace/personal
/workspace/kinetics-venv/bin/python -m kinetics.multiscale_large_data \
  --config configs/zhang_5p3m_data.json --resume
```

Local progress is `results/zhang_5p3m_data/progress.json`. A GitHub copy, when
published, is only a snapshot. The final manifest will record all shard hashes
and contiguous array hashes. Large part files and arrays are ignored by Git;
regenerate them with the published configuration or use the retained workspace.

## Network and training

- Input: train-standardized `[T, linear P, BCT(Y)]`.
- Output: separately train-standardized `delta[T, P, BCT(Y)]`.
- Network: **11–1600–800–400–11**, exact GELU, **1,624,811 parameters**.
- Raw NN temperature/pressure are self-fed. No element projection, enthalpy
  closure, decoder clipping or mass renormalization.
- Loss: L1; batch512; ten full epochs. An epoch shuffles all 5,300,000 rows
  without replacement, including the final partial batch.
- **10,352 updates/epoch, 103,520 updates/model, exactly 53,000,000 examples seen**.
  This differs from sampling 10,000 minibatches with replacement.
- Gamma=.1 is the paper reference. A single-model first-stage configuration is
  saved; ensemble/gamma scheduling is a separate resource choice.

Configuration: [zhang_5p3m_gelu.json](configs/zhang_5p3m_gelu.json).
Trainer: [large_zhang_training.py](kinetics/large_zhang_training.py).

Input/output population statistics are fitted on the full training set using
streaming moments. Standardized float32 training arrays are memory-mapped on
disk. Only the retained full-training data are used to fit the normalizers;
the fixed validation/test trajectories are not used in preprocessing.

Model weights, Adam variables, optimizer iteration and progress are checkpointed
together. The epoch permutation is reconstructed from seed and epoch, so a
mid-epoch resume does not repeat or skip training examples. The separate model
seed binding allows a finished pilot seed to be retained if the queue is later
extended. Config/data changes are rejected instead of silently mixed.

```bash
/workspace/kinetics-venv/bin/python -m kinetics.large_zhang_training \
  --config configs/zhang_5p3m_gelu.json --resume
```

`--run-updates N` checkpoints and pauses a bounded session. `--prepare-only`
prepares and checks the full-data normalization cache without training.

## Resource measurement and limits

The current environment has four CPU cores of quota and no GPU. A short large
GELU benchmark, while generation was also using CPU, measured about 0.076 seconds
per batch512 update with two TensorFlow threads. Linear extrapolation is about
2.2 hours/model for ten epochs; contention, four-thread execution, I/O and hardware
make this an estimate, not a runtime guarantee. Ensemble training can take much
longer. The complete dataset must still be generated before full-data training.

The old small pilot's 30-minute budget is not a full-scale completion claim.
The new generation/training configurations have explicit resumable wall-time
bounds; crossing one stops with reusable checkpoints, not fabricated results.

Uploaded paper PDFs are analysis inputs and are not republished in the repository.
