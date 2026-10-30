# Development workflow

Follow an explicitly requested branch. The current Box–Cox gamma=.2/.5,
ten-member tanh experiment uses `pre-process`, branched from the latest `kisung`;
continue that experiment on `pre-process`. For other subsequent work, use
`kisung` unless the user asks otherwise. Do not push subsequent changes directly
to `main` unless explicitly requested. Use the existing checkout, not a new
worktree, unless the user requests one.

The current follow-up is Zhang multi-scale sampling with full T/P/BCT(Y)
increments and physics OFF. Continue it on `pre-process`; preserve historical
physics-on experiments. The latest user request now enables enthalpy-based
temperature recovery from updated composition. Do not add element projection,
clipping or output mass renormalization. Preserve previous raw-temperature results.
Thermodynamic properties use Cantera's normalized-composition convention only;
the ANN's saved/self-fed species remain unchanged. Assess the requested
10-ms and 20-ms final times and report incomplete/physically invalid runs.

The latest requested scale is 5,300,000 pairs and 1000 complete training epochs,
continuing the completed ten-epoch model and Adam state in a new output path.
Use validation ReduceLROnPlateau and retain the best validation model plus snapshots.
Batch size remains 512 by the user's explicit choice after the 1024/2048 benchmark.
The user has now requested GPU takeover. CPU training is paused at the handoff
checkpoint (110,520 updates / 10.676197836 epochs). Do not restart CPU training
unless asked. See GPU_HANDOFF.md for restoring the split payload and GPU resume.
Use the existing paper-model architecture 1600/800/400 with GELU for this new
run, not the reduced 64/64/64 tanh pilot. Check current processes/locks before
resuming. Preserve and verify finished shards and Adam checkpoints. GPU/single
model/ensemble scheduling depends on the user's latest resource preference.

Run Python via `/workspace/kinetics-venv/bin/python` in this cloud environment.
Read README.md and the report relevant to the experiment before changing it.
Keep original datasets/results immutable; write new experiments to new output
paths. Fit preprocessing and sampling decisions on training/validation only.
