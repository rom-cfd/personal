# Literature query: robust representative selection in neural stiff-chemistry ensembles

Please identify closely related published work and critique the strategy below.
Provide verifiable paper titles, authors, years, DOI/arXiv/publisher links, and
separate direct precedents from merely related ideas. Do not invent citations.
We are particularly interested in negative results and failure modes.

## Objective

Replace repeated stiff chemical-kinetics ODE integration with a neural fixed-step
flow map, prioritizing a posteriori robustness over the lowest cost of a single
network. Large speedups can motivate paying for multiple networks, but the often
mentioned roughly 1000x speedup is a motivation, NOT a measured result of our
current Python/Cantera implementation. The small nine-species benchmark may have
a very different cost balance from a large detailed chemistry mechanism.

## Physical problem and network

- Adiabatic constant-pressure homogeneous 0-D hydrogen/air reactor using Cantera
  h2o2 chemistry with Ar removed: H2, H, O, O2, OH, H2O, HO2, H2O2, N2; 28 reactions.
- Initial T=900–1600 K, P=1–5 atm, equivalence ratio=0.5–2.0.
- Learn [T(t), Y_1(t),...,Y_9(t), P] -> [T(t+dt),Y_1(t+dt),...,Y_9(t+dt)].
  This is next mass-fraction/state prediction, NOT reaction-rate prediction.
- Fixed dt=2 microseconds in training, validation, test, and neural rollout.
- TensorFlow fully connected DNN: 64/64/64 tanh hidden units, linear residual
  head in standardized transformed coordinates; 9,738 learned parameters.
- Species preprocessing uses Box–Cox gamma=0.1. We implement Y^gamma/gamma,
  an affine-offset equivalent of (Y^gamma-1)/gamma after standardization, to
  avoid subtractive cancellation near zero. Scaling is fit on training data only.
- Original trajectories cover 0–0.4 ms; autoregressive evaluation extends to 2 ms.
  This mismatch can leave long induction-time states insufficiently represented.
- Original dataset: 90 independent initial conditions, trajectory-disjoint
  train/validation/test = 54/18/18; 18,000 fixed-dt transitions in total.

## What earlier experiments showed

One-step physical errors do not predict long-term stability reliably. Unconstrained
ANN rollouts can violate positivity, mass/element balances, and ignition behavior.
Box–Cox gamma=0.1 can amplify extremely small concentration perturbations; increasing
gamma or using log1p/asinh/shifted Box–Cox changes the tradeoff rather than solving
all problems. Transformed-space MSE must not be confused with physical accuracy.

Training-only augmentation was tested by perturbing valid training states and
independently integrating each perturbed state with Cantera for dt to obtain fresh
labels. Small and moderate perturbations helped some full profiles; larger
perturbation was not uniformly better. These augmented results are a separate
experimental arm; augmentation is NOT the default for the next committee study.

An optional explicit physical correction imposes nonnegative species/element
inventories and recovers temperature from current enthalpy at constant pressure.
This uses Cantera thermodynamics (HPY), not a chemical ODE solve. Conservation is
imposed, not learned. A three-seed ensemble sometimes improved test ignition and
profiles, but a long-induction case remained poorly timed even when ignition
classification was correct. Model agreement can therefore accompany shared bias.

## Current experiment: ten homogeneous judges and a representative candidate

Default is the **matched-budget original** setting: original, unaugmented training
data, identical architecture, Box–Cox gamma=0.1, and 10,000 optimizer updates per
member (Adam, batch 512, common learning-rate schedule). Models differ in seed,
initial weights, and minibatch stream. Three existing fits are reusable; seven
additional seeds complete the ten-member committee. Equal budget means per member;
total ensemble training/inference cost is larger than a single model.

At every timestep:

1. Feed the SAME current reactor state to all ten DNNs. Do not evolve ten separate
   trajectories and then compare predictions at different states.
2. Decode each predicted composition with an explicit physical range guard,
   project it to nonnegative composition with the current H/O/N inventory, and
   recover its temperature from the current mixture enthalpy at fixed pressure.
3. Exclude physically/thermodynamically invalid candidates. If none remain,
   record integration failure; do not substitute a reference answer.
4. Compute distances between the corrected candidate states, balancing temperature,
   major species and trace species using training-only scales/concentration floors.
5. Select an actual candidate medoid:

   i* = argmin_i sum_{j in valid candidates} d(z_i, z_j).

   The distance is Euclidean in selected scaled coordinates, not squared distance.
   This is neither a coordinate-wise median nor an average of state components.
6. Use this single selected state as the common next state. Repeat at the same dt.

Primary trial has no ODE fallback, no adaptive timestep, no hysteresis, no
architecture mixing, and no dynamic model training during rollout. Medoid ties
are resolved deterministically. Selection identities and switch rates are logged.

Two candidate distance definitions can be compared using validation only: scaled
physical T/Y distance and a training-scaled soft-log species distance. Soft-log
DISTANCE does not change the network's Box–Cox PREPROCESSING. Freeze the chosen
metric before held-out evaluation.

## Fair comparisons and diagnostics

Compare a fixed matched-original single model, the original three-member mean,
the ten-member mean, and the ten-member medoid. Because physical correction is
nonlinear, distinguish:

- average decoded proposals then repair (the legacy mean-ensemble order);
- repair individual candidates then average composition and restore enthalpy;
- repair individual candidates then choose an actual medoid.

The last two share the same valid candidate pool and isolate aggregation from
correction-order effects. No augmented data should enter this baseline comparison.
Heterogeneous architectures are a possible later ablation, not the current default.

Assess:

- True a posteriori self-fed 0-D simulations against independent Cantera reactors.
- Ignition/no-ignition classification; delay by T0+400 K crossing and max dT/dt.
- Entire temperature and all-species profiles; major/minor and concentration bins.
- Positivity, total mass, element and enthalpy consistency; failure/coverage rates.
- Input-noise robustness; representative switching and valid-member counts.
- Whether agreement/disagreement correlates with actual error on validation states.
- Actual inference/repair/selection timing. Do not assume a neural speedup.

Incomplete rollouts have undefined full-horizon profile error and are reported as
failures with coverage, rather than scored only on a short accurate prefix.
Non-ignition within the horizon is right-censoring, not proof of permanent
non-ignition. A few-microsecond ignition delay has limited resolution at dt=2 us;
reference-only finer sampling helps characterize measurement uncertainty.

## Specific literature questions

1. Prior work on neural fixed-step flow maps/direct state prediction for stiff
   reacting systems, distinct from neural reaction-rate models and neural ODEs?
2. Prior work on medoid/most-central-member regression ensembles, dynamic ensemble
   selection, expert selection, and robust aggregation of coupled physical states?
3. Does selecting an actual candidate preserve useful chemical correlations that
   averaging destroys, especially around ignition or strongly nonlinear manifolds?
4. How do correlated ensemble errors and shared training-data bias invalidate
   agreement as a confidence estimate? What diagnostics/calibration are supported?
5. Can representative switching create spurious dynamics or loss of numerical
   consistency? Are hysteresis, temporal regularization or stability guarantees known?
6. How should species-dependent scales/floors and thermochemical constraints enter
   the consensus metric without letting major species dominate or over-weighting
   numerical noise in trace species?
7. What is known about homogeneous versus architecture-diverse ensembles for
   physical surrogate rollout robustness, under comparable training budgets?
8. What experiments would convincingly test this hypothesis, including long
   induction delays, off-manifold states, perturbation augmentation and long rollouts?

Please critique this plan rather than assuming medoid selection is superior to
averaging. Identify what evidence would falsify the proposed advantage.

## First ten-member pilot results (now executed)

All results below use the unaugmented matched-budget original setting and the same
member-wise physical correction, with a fixed 18-IC, 2-ms held-out comparison.
Scaled-state medoid was chosen over the soft-log-distance alternative using
validation only, where its combined ignition/profile score was better than mean.

On held-out data, however, mean-after-member-correction had 94.4% ignition
classification, 5.5% median delay error on 12 igniting pairs, and 130 K mean
per-trajectory temperature-profile RMSE. Medoid had 88.9%, 9.1% on 13 igniting
pairs, and 219 K respectively. Both physically completed all 18 trajectories.

There is an important tradeoff: medoid captured a previously difficult long delay
at 1.177 ms versus Cantera 1.213 ms (3.0% error), which the means missed within
2 ms. But medoid produced two false ignitions within the evaluation horizon;
mean-after-correction produced none and missed that one true ignition. Therefore
conditional delay medians alone do not establish superiority.

The medoid representative switched on about 28.8% of adjacent accepted test
steps. Validation one-step disagreement versus scaled physical species error had
Spearman correlation about 0.899, which is not evidence of calibrated uncertainty
on self-fed long rollouts. These observations should inform a critical literature
review, not be treated as confirmation of medoid superiority.

## Scaling follow-up: 100 members and 10 ms

The next study has now trained 100 fresh same-architecture models on new 10-ms
Cantera trajectories. Every supervised target and autoregressive step is still
2 us. Training input times are thinned using chemical HP-equilibrium proximity,
time stratification and T/species arc length: 270,000 training candidates become
8,815 pairs, and the equilibrium fraction falls from 80.5% to 4.3%. Slow induction
is not treated as equilibrium merely because reaction/temperature rates are low.
Validation/test profiles remain fully sampled over 5,000 transitions.

The first ten freshly trained members form a same-data nested control. On the
18-IC 10-ms test, medoid100 classifies ignition correctly in 94.4% with median
conditional delay error about 5.4% and mean per-IC T-profile RMSE about 274 K.
Mean100 has similar classification and about 5.0% delay error / 275 K profile error.
The new ten-member controls have lower full-profile errors (about 179 K mean,
211 K medoid) and 100% classification on this small test. Thus increasing count
is not a uniform improvement.

All methods complete these rollouts under imposed mass/element/enthalpy closure,
but some 100-member traces cool severely and decrease specific entropy. This is
an actual kinetic/thermodynamic failure, not evidence of a trustworthy integrator.
The latest report separates induction, reaction/relaxation and equilibrium errors,
conditional ignition errors and tails, failure/censoring, and entropy diagnostics.
Please include entropy-consistent learned flow maps and long-time equilibrium
stability in the literature critique; do not equate hard conservation with valid
reaction dynamics.
