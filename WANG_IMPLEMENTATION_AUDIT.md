# Wang 2025 implementation audit

The existing models apply the Wang conservation equations to our earlier ANN.
They are **not an end-to-end reproduction of Wang 2025**. Temperature was already
an input. No old dataset, model, or experiment has been modified by this audit.

## Verified sources

- Attached paper: Wang et al., Combustion and Flame 275 (2025) 114105,
  DOI https://doi.org/10.1016/j.combustflame.2025.114105, §§2.1–2.2, §3.1.1,
  Appendix A.
- Paper-linked public repository, inspected commit
  `bc07e5a210ba357b8a5350b460a751a68a002174`:
  https://github.com/tianhanz/DNN-Models-for-Chemical-Kinetics/tree/bc07e5a210ba357b8a5350b460a751a68a002174
- Public inference:
  [modelclass.py](https://github.com/tianhanz/DNN-Models-for-Chemical-Kinetics/blob/bc07e5a210ba357b8a5350b460a751a68a002174/modelclass.py#L100)
  and `FortranDNN_Module.F`.
- Public stored settings: both `Model/model528w_msDsH2_constP/checkpoint/settings5000.json`
  and `Model/model642w_msDsClipESH2_constP/checkpoint/settings5000.json`.
  Dataset/model binaries were not downloaded.

## Paper versus current implementation

| Item | Paper 2025 | Current corrected model |
|---|---|---|
| Input temperature | T(t) included | Included; was never omitted |
| Input variables | T, P, G(Y) | Standardized T/1000, G(Y), log(P) |
| Box–Cox | (Y^0.1−1)/0.1 | Y^0.1/0.1; affine-equivalent after standardization |
| Species target | Eq. (2): G(Y_next) | G(Y_next)−G(Y_current) in normalized coordinates |
| Temperature output | §§2.1–2.2: obtained from energy equation | Temperature is also trained, then discarded under hard correction |
| Output dimensionality | Eq. (2) gives 9 species; §3.1.1 separately says 11 output neurons | 10: T plus 9 species |
| Hidden layers | 1600, 800, 400 | 64, 64, 64 |
| Activation | GELU | tanh |
| Training loss | L1 in G-space | MSE of normalized transformed residual |
| Optimizer | Adam | Adam, with an additional per-member norm clip of 5 |
| Stage 1 | 2500 epochs, batch1024, lr1e-4 | 10,000 total updates per member, batch512 |
| Stage 2 | 2500 epochs, batch131072, lr1e-5 | lr .001 → .0003 → .0001 by update fraction |
| Hard element correction | Eqs. (5)–(9) | Same constraint/objective, using row scaling and pseudoinverse |
| Decoder clipping | Not specified in paper | Transformed output guard enforces decoded Y in [0,1] |
| Corrected negative species | No explicit positivity constraint in Eqs. (5)–(9) | Reject candidate; no second QP or clipping |
| Thermal recovery | Energy equation and EOS | NASA-7 Newton solve, Cantera HPY numerical fallback, 200–6000 K admissibility |
| Framework | Python3.10.10, PyTorch2.0.0 | Python3.12.14, TensorFlow2.20.0 |
| Committee | Paper does not specify 100-member mean/medoid policy | Our additional comparison policy |

The Box–Cox affine offset is mathematically removed by a consistent standardizer;
the target, loss, network and decoder changes above are substantive differences.
The appendix pseudocode residual sign conflicts with Eq. (7) and its final
addition. The existing correction follows Eq. (7), E=A(Y_current−Y_proposed).

## Intentional data exceptions requested by the user

Keep all existing data and splits: 9-species/28-reaction Cantera mechanism,
54/18/18 initial-condition splits, 10-ms trajectories, dt=2 us, and 8,815 selected
training transitions. No relabeling or overwrite is authorized by this request.
Paper data instead use a 9-species/19-reaction mechanism, dt=1 us, 1.1 million
H2/air samples, and 90/10 sample split. These are declared data differences,
so this can be a method reproduction on our data, not reproduction of the
paper's reported numerical results.

## Public code does not resolve the paper's ambiguity

The repository README references both the 2022 multiscale-sampling paper and
Wang 2025. Its `netOneStep` implements:

1. Transform species in a full T/P/Y state, and standardize with saved Xmu/Xstd.
2. Run a network and undo Ymu/Ystd output scaling.
3. Multiply this output by dt and add it to the transformed **full state**.
4. Invert the species transform; temperature/pressure are network outputs.

This is a transformed-state-rate model, not the species-next-state plus energy
recovery path described in Wang 2025 §§2.1–2.2. The inspected inference files do
not implement the ANN-hard projection. Both stored configurations declare
9→1600→800→400→9, `mygelu` (tanh GELU approximation), L1, and dt1 us; they do not
establish the 11-variable, 9-species ANN-hard model described by the 2025 paper.
No training-loop source is present in the inspected tree. Stored normalization
numbers from another dataset cannot be reused with our training data.

Consequently, exact agreement between the paper, released inference and our
implementation cannot be asserted. In particular, output dimensions/targets,
2025 normalization fitting, initialization, optimizer epsilon, GELU variant,
small-negative policy and clipping need either the actual 2025 training/inference
source or explicit documented choices. Missing details must not be described as
confirmed author implementation.

## Resolved reference and prepared implementation

The user selected **2025 paper equations/descriptions** and **TensorFlow**, with
mathematical operations/training rules matched to the paper. The legacy public
`netOneStep` implementation is not the selected target.

Prepared implementation: [wang_paper.py](kinetics/wang_paper.py),
[full/pilot runner](kinetics/wang_paper_study.py), and
[complete declared configuration](configs/wang2025_paper.json).

- Input: train-standardized linear T, P(atm), literal Box–Cox(Y), dimension11.
- Output: **9 direct raw G_next values**, no temperature/pressure/residual head.
  This follows Eq.(2), with the paper's contradictory 11-output statement noted.
- Temperature: current T,P,Y → current enthalpy; hard-corrected next Y plus
  conserved enthalpy/fixed pressure → next T through Cantera HPY.
- Network: 1600/800/400 exact GELU, **1,624,009 parameters**.
- Loss: raw G-space L1, no output normalization.
- Adam: bias-corrected m/v and epsilon outside sqrt(v_hat), betas .9/.999,
  epsilon1e-8; implemented directly because Keras' epsilon-hat convention differs.
- Stages: 2500 epochs batch1024/lr1e-4, then 2500 epochs batch131072/lr1e-5.
  Our retained dataset has only8815 pairs, so the second-stage actual batch is8815.
  Optimizer moments are retained across stages. No gradient clipping is added.
- Species correction: literal Eqs.(5)–(9), direct full-rank solve, no abundance
  floor, positivity QP or decoder clipping. Mathematically undefined inverses,
  singular constraints, negative corrected proposals or failed HPY are rejected.

**Not confirmed by the publication:** input scaler fit/ddof, 2025 initialization,
GELU variant, Adam epsilon/reset details and invalid-candidate policy. The config
names these assumptions rather than describing them as verified author behavior.
Backend RNG/roundoff also differs. TensorFlow mathematical reproduction is the
requested scope; bit-for-bit author-code equivalence cannot be claimed.

Tests cover literal transforms and their inverse-domain/finite-precision limits,
the projection equations, Adam epsilon placement, species-only direct targets,
element/enthalpy recovery and serialization. A **10-update validation-only pilot**
completed in4.9seconds; all104 pilot proposals were rejected (71 projection,
33 thermal). This undertrained pilot verifies failure handling, **not predictive
accuracy**, and has not replaced any active model.

## Resource estimate and pending full training

[Measured CPU benchmark](results/wang_implementation_audit/cost_benchmark.json):
median update costs were .0536s at batch1024 and .4926s at batch8815. Including
the final623-point batch per stage1 epoch, the complete schedule extrapolates to
**.665hours (~40minutes) per model, ~66.5hours for100 serial members**. This is a
short TensorFlow benchmark, not a runtime guarantee; validation/I/O are extra.
The benchmark uses Keras Adam for cost measurement, while the candidate's actual
optimizer follows the bias-corrected PyTorch equations described above.

The user has been asked whether to run **one full model plus10-ms evaluation
within90CPUminutes**, or hold full training. The full100-member run exceeds the
prior short-run approval and has not been launched. Existing active weights
remain the earlier corrected models until a new model is trained/evaluated.

```bash
python -m kinetics.wang_paper_study                    # short pilot only
# Only after explicit budget approval:
python -m kinetics.wang_paper_study --full --budget-seconds 5400
python -m pytest -q
```

Output directories are immutable; use a new config output path for repeats.
Interrupted full jobs save model weights, Adam slots, RNG/progress and history;
automatic resume is not yet implemented. No original datasets are overwritten.
