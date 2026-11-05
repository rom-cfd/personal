# Pure-MSE gamma continuation after the completed10k comparison

The user requested long training/evaluation AFTER the current gamma10k
evaluation, retaining the previous per-model time allowance. All six gamma
values (.05,.1,.2,.3,.5,1), four families and ten seeds are included:240 models.
Each model gets150 seconds of NEW active work in this phase. Stop earlier on
the original scalar full-train MSE plateau policy; do not add a hard update cap.
No soft-constraint term is introduced. LR after update9000 remains.0001.

Do not run any new training or heavy tests while the10k pipeline is measuring
inference time. Implementation is staged below the current result directory,
outside its frozen source globs; install new gamma_long_* sources only after
the10k supervisor exits0 and FINAL_ARTIFACTS.json exists. Original gamma_sweep_*
and all previous outputs remain byte-identical. No new push is authorized by
the follow-up; the completed pre-experiment publication is b75d680 on origin/bw.

Origins:200 new gamma checkpoints have exact Adam/RNG state. Gamma.1 P-only10
and T/P/phi10 also have complete legacy state; source hashes and configurations
are validated by their original restorers before transfer of every Adam slot,
iteration and NumPy/TF RNG to the pure-MSE continuation trainer.

Gamma.1 ANN10 and T/P10 lack optimizer state (20 models). Rebuild ONLY these20
to10000 updates with the same seeds/data/architecture/LR schedule, save exact
state, then continue. This is a rebuilt paired10k origin, not exact continuation
of the historical weights. Its bootstrap time counts against its150 seconds.
Keep historical10k controls intact; evaluate the rebuilt ANN/T-P and three
affected mixed teams at10k before any long comparison. Unaffected P-only/phi
short traces and other gamma short traces are reused with hash checks.

The allowance is cumulative ACTIVE time across explicitly closed bootstrap and
continuation visits. Queue time and evaluation are excluded. Persist open-visit
start/deadline; an interrupted open visit is charged on recovery, never reset.
Do not give bootstrapped models150 seconds a second time. Kernel/checkpoint
boundary overhead is recorded just as in prior runs.

Full-train MSE is observed every1000 updates. Starting at50000, check every5000:
two disjoint20000-update windows must each contain20 observations; q25/median/q75
absolute relative changes must be <=1%/1%/3%, sustained for10000 updates (three
consecutive checks; earliest plateau60000). No test/validation stopping.
Checkpoints preserve weights, scales, optimizer, RNGs, histories, monitor,
lineage and cumulative allowance. Save every10000 updates and at termination.

Freeze all240 endpoints, then evaluate validation first and freeze long gamma
selection before test. Selection remains repaired completion count, then median
temperature NRMSE on common ICs among maximum-completion ties. All ten variables
are reported; this is not an all-species scalar optimum.

For each IC and mode, deliver short7x10 + its NRMSE table, then long7x10 + its
NRMSE table. Both stages share variable-column y scales. Each graph has Cantera
and six gamma curves. Short gamma.1 uses the paired rebuilt ANN/T-P origins.
Export common-all84 and within-team-all12 cohorts, paired same-gamma short/long
cohorts, and validation-selected-gamma test comparisons. Explain the historical
gamma.1 versus paired-short distinction. Raw and repaired stay separate.

The nominal240x150s active training ceiling is10h. A derived outer watchdog
also includes bootstrap short evaluation, setup, long evaluation and reporting.
No parallel model work or budget reset; app-independent supervisor and read-only
observer use the same established CPU flags and interpreter.
