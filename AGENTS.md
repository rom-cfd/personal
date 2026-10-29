# Development workflow

After the initial user-requested publication to `main`, use `kisung` for subsequent
work. Switch to `kisung` before editing; do not push subsequent changes directly
to `main` unless the user explicitly asks. Use the existing checkout, not a new
worktree, unless the user requests one.

Run Python via `/workspace/kinetics-venv/bin/python` in this cloud environment.
Read README.md and the report relevant to the experiment before changing it.
Keep original datasets/results immutable; write new experiments to new output
paths. Fit preprocessing and sampling decisions on training/validation only.
