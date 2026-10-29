# 10 ms training-data generation and equilibrium downsampling

Full Cantera trajectories cover **0–10 ms**, with dt=2 microseconds and 5,000
transitions per initial condition. The original 90 initial conditions and
trajectory-disjoint train/validation/test membership are unchanged.

Only training input times are thinned. A selected input at j always has target
j+1, not the next retained sample. The supervised prediction interval is therefore
**2 microseconds for every pair**, even when input sampling times have large gaps.
Validation and test retain the entire 5,001-point profile.

| Quantity | Count |
|---|---:|
| Dense training candidates | 270,000 |
| Dense equilibrium candidates | 217,421 (80.5%) |
| Selected training pairs | 8,815 |
| Selected non-equilibrium/induction | 8,439 |
| Selected equilibrium anchors | 376 (4.3%) |

A state is considered near equilibrium only when it agrees with the Cantera HP
equilibrium at the same initial enthalpy, pressure and elemental inventory:
temperature within 1 K, species L1 difference <=1e-4, and each species within 5%
with a 1e-8 floor. Both current/next states must satisfy this, persistently for
25 transitions (50 microseconds). Slow initial induction far from HP equilibrium
is NOT classified as equilibrium simply because dT/dt is small.

Each training trajectory keeps at most 320 non-equilibrium input times, combining
initial states, time stratification and concentration/temperature arc length.
At most eight equilibrium anchors are retained, including late-time points.
Anchors are kept because removing all equilibrium examples could worsen long-term
drift. Selection may change training-distribution weights and must be assessed by
full-profile validation, not assumed to improve the ANN automatically.

No perturbation augmentation is used in this dataset. Its statistics and scalers
must be fit afresh; the old 0.4-ms-trained ten networks are comparison artifacts,
not reused as newly trained members.

Original source SHA256: `56457bf445a4970ca15361eb35f81cd218c01f76934a87652b8cba3b2a0ad7fe`.
Dense 10-ms SHA256: `8d41f19f2145e12f9e89ee3eaa88fb0b1b93a84dcaaee65dcaa2e8b264077563`.
Selected pair SHA256: `44db17b16e167ea389631dacb8015528f76cbd5c2072e9fb919701e54d50fa3e`.

- [Sampling summary](results/long10ms_data/sampling_summary.csv)
- [Every selected input and target time](results/long10ms_data/selected_inputs.csv)
- [Counts](results/long10ms_data/figures/sampling_counts.png)
- [Retained time distribution](results/long10ms_data/figures/selected_times.png)
- [Representative training profiles](results/long10ms_data/figures/sampling_examples.png)
- [Generator](kinetics/long_data.py)
