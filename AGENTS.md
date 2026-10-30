# Development workflow

Follow an explicitly requested branch. The current Box–Cox gamma=.2/.5,
ten-member tanh experiment uses `pre-process`, branched from the latest `kisung`;
continue that experiment on `pre-process`. For other subsequent work, use
`kisung` unless the user asks otherwise. Do not push subsequent changes directly
to `main` unless explicitly requested. Use the existing checkout, not a new
worktree, unless the user requests one.

The current follow-up is Zhang multi-scale sampling with full T/P/BCT(Y)
increments and physics OFF. Continue it on `pre-process`; preserve historical
physics-on experiments. Use raw NN temperature and pressure, with no enthalpy
closure, projection, clipping or mass renormalization. Assess the requested
10-ms and 20-ms final times and report incomplete/physically invalid runs.

Run Python via `/workspace/kinetics-venv/bin/python` in this cloud environment.
Read README.md and the report relevant to the experiment before changing it.
Keep original datasets/results immutable; write new experiments to new output
paths. Fit preprocessing and sampling decisions on training/validation only.
