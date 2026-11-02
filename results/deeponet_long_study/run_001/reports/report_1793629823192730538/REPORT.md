# Sequential P-only / T/P long-training comparison

Endpoints were frozen using training loss and the declared plateau policy only. Test ICs were already observed in previous studies; no test-based stopping or tuning was used.

P-only continued the new 10k Adam/RNG states. T/P was newly reproduced from initialization because the historical optimizer/RNG states were not saved. Original files remain unchanged.

Saved endpoints: 20; plateau: 0; time limit: 20; update limit: 0.
Completed evaluation arms: 20/20. Missing arms are unrun, not successful.

summary.csv reports full-horizon RMSE only on complete trajectories, alongside completion. common_completed_IC_comparison.csv compares identical completed IC sets and lists the ICs. Pooled temperature RMSE and the arithmetic mean of per-IC RMSE are separate columns. All test IC figures include T and nine species, with Cantera and the available five teams. Failed prefixes end without padding. Raw and repaired are separate experiments.

Repair imposes composition/element/enthalpy conditions; this is not learned conservation. Entropy is a diagnostic, not a filtering rule. Mean teams evaluate all members at their own current state. Invalid rows are excluded only in repaired mode, with family counts retained.

Timing separates model loading, warm-up, neural inference, decoding, candidate repair, mean repair, rollout, dense one-step evaluation and diagnostics. Training uses the same sequential engine and CPU thread settings, but plateau endpoints have different update counts. Historical grouped ANN time is not used as a direct speed comparison.

Preexisting result files preserved: 1607. Report seconds: 262.446.
No commit/push or published snapshot change was performed.

- ann10: 18/18 complete, T RMSE mean on completed ICs=179.13 K; pooled=277.692 K.
- ann_p_only20: 18/18 complete, T RMSE mean on completed ICs=234.691 K; pooled=414.528 K.
- ann_tp20: 18/18 complete, T RMSE mean on completed ICs=464.492 K; pooled=598.377 K.
- p_only10: 18/18 complete, T RMSE mean on completed ICs=360.874 K; pooled=554.118 K.
- tp10: 16/18 complete, T RMSE mean on completed ICs=646.95 K; pooled=836.887 K.
