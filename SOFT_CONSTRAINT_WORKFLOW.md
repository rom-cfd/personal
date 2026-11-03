Soft element constraint, matched10k experiment
============================================

The user requested all currently compared NN families freshly trained10000
updates each, with the element soft constraint from two uploaded methane
training scripts. A follow-up explicitly selected the step2 penalty weight10.
This means40 independently trained members: ANN, P-only DeepONet, T/P DeepONet,
T/P/initial-phi DeepONet, each with seeds11,22,...,110. The original ANN10 is the
first10 of the existing ANN100; neither the baseline nor the new study uses
ANN100 in its seven comparison teams. Mixed teams contain corresponding ANN10
and DeepONet10, so both families are newly soft-trained in the soft mixed arms.

The user references are preserved verbatim in `references/soft_constraint/`.
Step1 uses actual Mu1 for species reconstruction, and step2 uses actual Mu10 for
reaction rates; both factory defaults are100 but those defaults are overridden
by their callers. The chosen penalty reduction is the same in both scripts:

```
data_loss = mean((delta_pred - delta_true)**2)
element_loss = mean_batch(sum_elements(abs(Y_pred @ A.T - Y_true @ A.T)))
total_loss = data_loss +10 * element_loss
```

Here Y is the predicted next state in physical mass-fraction units, and Y_true
is the Cantera next-state composition. A contains mass fractions of H/O/N for
the established9 hydrogen species and is rebuilt from Cantera and verified
against the original dataset matrix. Temperature is included in data MSE but
excluded from the element term. There are no extra positivity, energy, entropy
or mass penalties. All elements together imply a mass-sum penalty indirectly.

This adopts the element soft-loss idea and weight from the methane scripts,
while retaining this project's state-transition task. The methane53-species
data, subset/reconstruction architecture, removal of N2, minmax normalization,
rate labels, epoch counting and LR callback are not applied to this experiment.
All original data, split, selected8815 pairs, fixed2us target, 10ms reference,
no augmentation, Box-Cox gamma.1, fitted train-only standardizers, branch/trunk
partitions, widths/latent7, minibatch512 replacement sampling, paired seeds,
Adam, clipping5 and absolute LR boundaries6000/9000 remain unchanged.

For a predictor output delta, inverse standardization is applied to
`current_standardized + delta`. The physical loss inverse is
`sign(z) * (gamma * abs(z))**(1/gamma)`. On z>=0 this is the original Box-Cox
inverse. On negative encoded predictions this is a finite differentiable
extension, following the signed-power approach in step2. It avoids silently
clipping away violations or generating NaN gradients. No projection or physical
repair is applied inside the training loss. Actual model inference uses the
original strict inverse; invalid negative encoded states are still invalid in
raw evaluation. Domain-violation rates are recorded explicitly.

Network inputs/weights and residual data loss remain float32 as before. The
current-state reference, inverse transformations, elemental matrix and physical
penalty are evaluated in float64 to avoid adding an artificial float32
conservation noise floor. Gradients flow back into the original float32 weights.
Training, evaluation and plotting are sequential, CPU-only, TensorFlow intra2 /
inter1, OMP/OpenBLAS/MKL2, oneDNN0. Serial TensorFlow loops do not run members
concurrently. Mu0 is tested against the original scalar Adam updates with
array-equal weights, optimizer slots and next minibatches for all four families.
Mu10 checkpoint continuation is similarly tested for all four families.

The newly authorized training budget is exactly40x10000=400000 updates, not an
extension of any historical long run. All starts are fresh. As conservative
failure guards the runner records a one-hour training wall safety ceiling and
two-hour pipeline ceiling. These are stopping guards, not permission to run
extra experiments. If a guard is reached, preserve checkpoints, report incomplete
work and do not reset the absolute clock on resume. Fixed10k is the only normal
training completion criterion; no plateau stopping or extra LR tuning occurs.

Before test evaluation all40 endpoints are frozen. Baseline and soft full-train
and validation MSE, physical element L1 and inverse-domain failure rates are
reported for each member. The seven new teams are ANN10, P-only10,
ANN+P-only20, T/P10, ANN+T/P20, T/P/phi10, ANN+T/P/phi20. Each is evaluated raw
and repaired on the18 previously observed test ICs, with dense one-step and
independent self-fed5000-step rollouts. No test-driven choice of mu, endpoints,
models, postprocessing or preprocessing is made. Existing baseline traces are
reused after hash checks; only four missing raw P-only/T/P baseline test arms
are computed. There are no new Cantera trajectories and no ODE fallback.

The report shows seven rows x ten variables, three curves in every cell:
Cantera, baseline10k, soft10k. Unlike the older short-vs-long figure, the ANN row
also has baseline and soft curves. Each profile page is followed by the same
IC's NRMSE table. Raw and repaired each have36-page PDFs and per-IC PNGs. NRMSE
is the user's util.py default, RMSE(Cantera,NN)/RMS(Cantera), unscaled, equal
weight per time point. Incomplete traces have no full-horizon score. Summary
medians compare the identical completed-IC intersection of all14 variants.
If the intersection is empty, show N/A and actual completion counts. Include
the individual metrics, means/std/quartiles and all observed loss components.
Conservation imposed by inference repair must not be claimed as learned.

Use a new namespace and preserve all preceding modules and results:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=-1 \
MPLCONFIGDIR=/tmp/kinetics-mpl XDG_CACHE_HOME=/tmp/kinetics-cache \
/workspace/kinetics-venv/bin/python -m kinetics.soft_constraint_study \
  --config configs/soft_constraint_10k.json --phase prepare
```

Run with the same variables and `--phase run --resume` after implementation
checks. `run.json` binds configuration, dataset, model origins and source hashes.
Checkpoints contain weights, preprocessing, Adam/iteration, independent NumPy and
TensorFlow RNGs, loss-component histories, LR, disabled plateau policy and
runtime/source signatures. Completion markers are written last and no checkpoint
is overwritten. Existing result hashes are checked after reporting. New commits,
pushes and publication are separate actions and are not part of this request.
