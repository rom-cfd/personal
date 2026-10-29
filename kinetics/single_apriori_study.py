"""Select one of the 100 frozen members by validation one-step accuracy only.

Run: python -m kinetics.single_apriori_study
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/kinetics-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/kinetics-cache')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .committee import Committee
from .committee_study import run_rollouts
from .long_committee_study import horizon_summary
from .metrics import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/single_apriori_wang'))
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()
    root = args.output
    if root.exists():
        raise FileExistsError('Preserve existing experiments; use a new output')
    if args.batch_size < 1:
        raise ValueError('positive batch size required')
    root.mkdir()
    for directory in ('methods', 'traces'):
        (root / directory).mkdir()
    old = Path('results/wang_hard10ms')
    cfg = json.loads((old / 'config.json').read_text())
    (root / 'config.json').write_text(json.dumps({**cfg, 'output': str(root)}, indent=2))
    protocol = dict(selection_split='val', selection_kind='one-step a priori only',
        criterion='existing metrics.evaluate selection_score, after Wang correction',
        score='mean(log1p(min(abs(Ypred-Ytrue)/max(Ytrue,1e-8),1e12))) + 10*major_Y_RMSE + 10*mass_sum_MAE + 100*invalid_row_fraction',
        major_species_threshold=1e-3, ties='ascending seed', ranking_uses='all 18 validation trajectories, every one of 5000 adjacent fixed-dt pairs',
        transformed_mse_used_for_selection=False, posteriori_used_for_selection=False,
        test_used_for_selection=False, retraining=False, dt=2e-6, horizon=.01,
        physics_backend='wang_boxcox', no_other_candidate_fallback=True,
        max_execution_seconds=1800, batch_size=args.batch_size)
    (root / 'selection_protocol.json').write_text(json.dumps(protocol, indent=2))
    started = time.perf_counter()
    def budget():
        if time.perf_counter()-started > 1800:
            raise TimeoutError('30 minute CPU experiment cap exceeded')
    with np.load(cfg['source']) as data:
        states, pressures, split = data['states'], data['pressures'], data['split']
        matrix, dt, names = data['element_matrix'], float(data['dt']), data['species']
    assert dt == protocol['dt']
    ids = np.flatnonzero(split == 'val')
    n = states.shape[1]-1
    x = np.c_[states[ids, :-1].reshape(-1, 10), np.repeat(pressures[ids], n)]
    y = states[ids, 1:].reshape(-1, 10)
    committee = Committee.load(old / 'methods/medoid100.json')
    records = json.loads(Path('results/committee100_long10ms/members.json').read_text())
    for r in records:
        assert hashlib.sha256(Path(r['path']+'.keras').read_bytes()).hexdigest() == r['sha256']
    predictions = np.empty((len(committee.seeds), len(x), 10), dtype=float)
    transformed_error = np.zeros(len(committee.seeds))
    base = committee.base
    for begin in range(0, len(x), args.batch_size):
        budget()
        end = min(begin+args.batch_size, len(x)); xx = x[begin:end]
        delta = committee.forward(base.features(xx).astype('float32')).numpy().astype('float64')
        current_z = base.norm.forward(base.encode_state(xx[:, :10]))
        target_delta = base.norm.forward(base.encode_state(y[begin:end])) - current_z
        transformed_error += np.sum((delta-target_delta[None])**2, axis=(1, 2))
        z = base.norm.inverse(current_z[None] + delta)
        decoded = base.transform.inverse(np.clip(z[..., 1:], 0, base.transform.forward(np.ones((1, 9)))))
        raw = np.concatenate([z[..., :1]*1000, decoded], axis=-1)
        if begin == 0:
            np.testing.assert_allclose(raw[:, :8], committee.proposals(xx[:8]), atol=1e-10, rtol=1e-12)
        predictions[:, begin:end] = committee.repair(raw, xx)
        if begin % (args.batch_size*10) == 0:
            print('APRIORI_VALIDATION', end, '/', len(x), flush=True)
    ranking = []
    for j, seed in enumerate(committee.seeds):
        budget()
        ranking.append(dict(seed=int(seed), samples=len(x),
            transformed_mse=float(transformed_error[j]/(len(x)*10)),
            **evaluate(y, predictions[j], x, matrix)))
    rank = pd.DataFrame(ranking).sort_values(['selection_score', 'seed'])
    if not np.isfinite(rank.iloc[0].selection_score):
        raise ValueError('No finite validation selection score')
    rank.to_csv(root / 'validation_member_ranking.csv', index=False)
    seed = int(rank.iloc[0].seed); index = list(committee.seeds).index(seed)
    path = committee.paths[index]; metric = committee.metric
    frozen = dict(**protocol, selected_seed=seed, model_path=path,
                  model_sha256=hashlib.sha256(Path(path+'.keras').read_bytes()).hexdigest(),
                  validation_score=float(rank.iloc[0].selection_score),
                  validation_ids=ids.tolist(), validation_pairs=len(x),
                  selection_wall_seconds=time.perf_counter()-started)
    # Persist this choice BEFORE looking at any rollout or test performance.
    (root / 'selection_frozen.json').write_text(json.dumps(frozen, indent=2))
    print('FROZEN_SINGLE', seed, frozen['validation_score'], flush=True)
    selected_validation_predictions = predictions[index].copy()
    del predictions, committee
    p = Committee([path], [seed], metric, rule='single', physics_backend='wang_boxcox')
    p.save(root / 'methods/single_best.json')
    summaries, details, species, timing, probes, corrections, entropy_rows, checks = [], [], [], [], [], [], [], []
    for partition in ('val', 'test'):
        budget()
        ids = np.flatnonzero(split == partition)
        fine = pd.read_csv(Path('results/committee100_long10ms') / f'reference_ignition_{partition}.csv').set_index('trajectory')
        px0 = np.c_[states[ids, 0], pressures[ids]]
        first = p.predict(px0)
        p.timing = {k: 0. for k in p.timing}
        p.vector_physics.counts = {k: 0 for k in p.vector_physics.counts}
        begin = time.perf_counter()
        def progress(step, active):
            budget(); print('ROLLOUT', partition, step, active, flush=True)
        _, rows, sp, events, traces = run_rollouts(p, states[ids], pressures[ids], ids, matrix, dt, progress)
        timing.append(dict(method='single_best', split=partition, seconds=time.perf_counter()-begin, **p.timing))
        corrections.append(dict(method='single_best', split=partition, **p.vector_physics.counts))
        np.savez_compressed(root / 'traces' / f'single_best_{partition}.npz',
                            **{f'trajectory_{int(i)}': t for i, t in zip(ids, traces)})
        pd.DataFrame(events).to_csv(root / f'selection_single_best_{partition}.csv', index=False)
        np.testing.assert_allclose(first, np.array([t[1] if len(t)>1 else np.full(10, np.nan) for t in traces]), equal_nan=True, atol=1e-8, rtol=1e-10)
        restored = Committee.load(root / 'methods/single_best.json')
        np.testing.assert_allclose(first, restored.predict(px0), equal_nan=True, atol=1e-8, rtol=1e-10)
        checks.append(dict(split=partition, first_step_match=True, reload_match=True,
                           only_selected_seed=all(e['selected_seed'] in (-1, seed) for e in events)))
        for row in rows:
            tau = float(fine.loc[row['trajectory'], 'tau400'])
            row['reference_tau400_refined'] = tau
            row['tau400_relative_error_refined'] = abs(row['predicted_tau400']-tau)/tau if np.isfinite(tau) and row['prediction_ignites'] else np.nan
        summary = horizon_summary(rows)
        summary['median_ignition_error_refined'] = float(pd.DataFrame(rows).tau400_relative_error_refined.median())
        common = dict(method='single_best', split=partition, horizon=.01, seed=seed)
        summaries.append(dict(**common, **summary))
        details.extend(dict(**common, **r) for r in rows)
        species.extend(dict(**common, **{**r, 'species': str(names[r['species']])}) for r in sp)
        j = np.unique(np.r_[np.arange(8), np.linspace(0, n-1, 24).astype(int)])
        px = np.c_[states[ids][:, j].reshape(-1, 10), np.repeat(pressures[ids], len(j))]
        py = states[ids][:, j+1].reshape(-1, 10)
        pred = p.predict(px)
        probes.append(dict(**common, **evaluate(py, pred, px, matrix)))
        if partition == 'val':
            # The singleton runtime path must agree with the ranking path.
            sampled = selected_validation_predictions.reshape(len(ids), n, 10)[:, j].reshape(-1, 10)
            np.testing.assert_allclose(pred, sampled, equal_nan=True, atol=2e-5, rtol=2e-7)
        for i, trace in zip(ids, traces):
            values, enthalpy = [], []
            for state in trace:
                p.gas.TPY = state[0], pressures[i], state[1:]
                values.append(p.gas.entropy_mass); enthalpy.append(p.gas.enthalpy_mass)
            values = np.asarray(values); changes = np.diff(values)
            entropy_rows.append(dict(method='single_best', split=partition, trajectory=int(i), points=len(trace),
                min_temperature=float(trace[:, 0].min()),
                decreasing_fraction=float(np.mean(changes < -1e-6)) if len(changes) else np.nan,
                largest_step_drop=float(max(0., -changes.min())) if len(changes) else np.nan,
                minimum_delta_from_initial=float((values-values[0]).min()),
                max_enthalpy_drift_J_kg=float(np.max(np.abs(np.asarray(enthalpy)-enthalpy[0])))))
        print('SINGLE_RESULT', partition, summary, flush=True)
    for filename, values in [('summary', summaries), ('trajectories', details), ('species_profiles', species),
                             ('timing', timing), ('one_step_probes', probes), ('correction_diagnostics', corrections),
                             ('entropy_diagnostics', entropy_rows), ('reload_checks', checks)]:
        pd.DataFrame(values).to_csv(root / f'{filename}.csv', index=False)
    for filename in ('summary', 'one_step_probes', 'timing'):
        ensemble = pd.read_csv(old / f'{filename}.csv')
        ensemble = ensemble[ensemble.method.isin(['mean100', 'medoid100'])]
        single = pd.read_csv(root / f'{filename}.csv')
        pd.concat([ensemble, single], ignore_index=True).to_csv(root / f'comparison_{filename}.csv', index=False)
    manifest = dict(status='completed', selected_seed=seed, baseline=str(old), source=cfg['source'],
                    source_sha256=hashlib.sha256(Path(cfg['source']).read_bytes()).hexdigest(),
                    execution_seconds=time.perf_counter()-started, no_retraining=True,
                    source_hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(Path('kinetics').glob('*.py'))})
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('COMPLETED', manifest['execution_seconds'], flush=True)


if __name__ == '__main__':
    main()
