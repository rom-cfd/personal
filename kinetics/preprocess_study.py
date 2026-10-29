"""Gamma ablation on the original 64/64/64 tanh next-state ANN.

Train ten independent members per gamma; freeze validation one-step singletons
for ALL gammas before any rollout or test assessment. All outputs are new.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from .batched_training import train_group
from .committee import Committee, fit_metric
from .committee_study import run_rollouts
from .data import samples
from .long_committee_study import horizon_summary
from .metrics import evaluate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def rank_members(committee, x, y, matrix, batch_size, budget):
    """Every validation pair, identical corrected prediction path to inference."""
    predictions = np.empty((len(committee.seeds), len(x), 10))
    for begin in range(0, len(x), batch_size):
        budget()
        end = min(begin + batch_size, len(x))
        xx = x[begin:end]
        predictions[:, begin:end] = committee.repair(committee.proposals(xx), xx)
    rows = [dict(seed=int(seed), samples=len(x), **evaluate(y, pred, x, matrix))
            for seed, pred in zip(committee.seeds, predictions)]
    ranking = pd.DataFrame(rows).sort_values(['selection_score', 'seed'])
    if not np.isfinite(ranking.iloc[0].selection_score):
        raise ValueError('No finite single-member validation score')
    return ranking, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/preprocess.json'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg['output'])
    if root.exists() and not args.resume:
        raise FileExistsError('Preserve results: choose a new output or --resume')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != cfg:
        raise ValueError('Cannot resume with a changed configuration')
    write_json(root / 'config.json', cfg)
    started = time.perf_counter()
    previous = json.loads((root / 'progress.json').read_text()) if (root / 'progress.json').exists() else {}
    spent = previous.get('elapsed_seconds', 0.)
    def budget():
        if spent + time.perf_counter() - started > cfg['max_seconds']:
            raise TimeoutError('Experiment 30-minute wall-time cap exceeded')
    def progress(phase):
        write_json(root / 'progress.json', dict(phase=phase, elapsed_seconds=spent+time.perf_counter()-started))
    with np.load(cfg['source']) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(cfg['training_pairs']) as archive:
        pairs = {key: archive[key] for key in archive.files}
    matrix = data['element_matrix']
    x, y = pairs['x'], pairs['y']
    train = np.flatnonzero(data['split'] == 'train')
    assert set(pairs['parent'][:, 0]) <= set(train)
    assert float(data['dt']) == float(pairs['dt']) == cfg['dt']
    assert np.isclose((data['states'].shape[1]-1)*cfg['dt'], cfg['horizon'])
    # Check exact parent/next-state identity as well as split membership.
    for xx, yy, (i, j) in zip(x, y, pairs['parent']):
        np.testing.assert_array_equal(xx[:10], data['states'][i, j])
        np.testing.assert_array_equal(yy, data['states'][i, j+1])
        assert xx[-1] == data['pressures'][i]
    hashes = {key: sha(cfg[key]) for key in ('source', 'training_pairs')}
    if (root / 'input_hashes.json').exists() and json.loads((root / 'input_hashes.json').read_text()) != hashes:
        raise ValueError('Input data changed')
    write_json(root / 'input_hashes.json', hashes)
    protocol = dict(selection_split='val', selection='all validation one-step pairs, corrected physical-unit selection_score',
        ties='ascending seed', test_used_for_selection=False, posteriori_used_for_selection=False,
        gamma_selected_from_test=False, singleton_scope='common control: best of all twenty NEW members; auxiliary best within each gamma',
        correction='gamma-matched Wang projection; reject negative or infeasible candidates',
        temperature='discard ANN T; recover from conserved enthalpy at fixed P',
        decoder_guard='transformed Y clipped to the domain corresponding to [0,1] before correction',
        ensemble_definitions=['mean10 after individual correction and thermal closure', 'medoid10 actual corrected candidate'],
        ode_fallback=False, hysteresis=False, validation_downsampling=False,
        architecture='64/64/64 tanh, 9738 parameters/member', updates_per_member=cfg['updates_per_member'])
    write_json(root / 'selection_protocol.json', protocol)
    metric = fit_metric(x[:, :10], cfg['species_distance_floor'])
    write_json(root / 'distance_metric.json', metric)
    vx, vy, val_ids = samples(data, 'val')
    frozen_all = {}
    try:
        # Complete training and a priori choices before reading ANY test scores.
        for gamma in cfg['gammas']:
            label = f'g{gamma:g}'
            out = root / label
            for folder in ('models', 'methods', 'traces'):
                (out / folder).mkdir(parents=True, exist_ok=True)
            records = json.loads((out / 'members.json').read_text()) if (out / 'members.json').exists() else []
            for record in records:
                assert record['sha256'] == sha(record['path']+'.keras')
                assert record['metadata_sha256'] == sha(record['path']+'.json')
            remaining = [s for s in cfg['seeds'] if s not in {r['seed'] for r in records}]
            history = pd.read_csv(out / 'history.csv').to_dict('records') if (out / 'history.csv').exists() else []
            groups = pd.read_csv(out / 'training_groups.csv').to_dict('records') if (out / 'training_groups.csv').exists() else []
            for begin in range(0, len(remaining), cfg['group_size']):
                budget()
                seeds = remaining[begin:begin+cfg['group_size']]
                print('TRAIN_START', gamma, seeds, flush=True)
                members, seconds, h = train_group(dict(kind='boxcox', gamma=gamma), seeds,
                    cfg['updates_per_member'], x, y, cfg, started+cfg['max_seconds']-spent, progress=True)
                for seed, member in zip(seeds, members):
                    path = out / 'models' / f'boxcox_{label}_seed{seed}'
                    if Path(str(path)+'.keras').exists():
                        raise FileExistsError('Unregistered existing model; preserve before retry')
                    member.save(path)
                    records.append(dict(seed=seed, gamma=gamma, path=str(path), updates=cfg['updates_per_member'],
                        sha256=sha(str(path)+'.keras'), metadata_sha256=sha(str(path)+'.json'), training_pairs_sha256=hashes['training_pairs']))
                    write_json(out / 'members.json', records)
                history.extend(h)
                groups.append(dict(seeds=','.join(map(str, seeds)), seconds=seconds, updates=cfg['updates_per_member']))
                pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
                pd.DataFrame(groups).to_csv(out / 'training_groups.csv', index=False)
                progress('training')
                del members
                gc.collect()
            records.sort(key=lambda r: cfg['seeds'].index(r['seed']))
            assert len(records) == 10
            paths = [r['path'] for r in records]
            p = Committee(paths, cfg['seeds'], metric, physics_backend=cfg['physics_backend'], collect_mean_diagnostics=False)
            assert p.vector_physics.gamma == gamma
            rank, predictions = rank_members(p, vx, vy, matrix, cfg['evaluation_batch_size'], budget)
            rank.to_csv(out / 'validation_member_ranking.csv', index=False)
            seed = int(rank.iloc[0].seed)
            index = cfg['seeds'].index(seed)
            # Verify singleton inference equals the all-member ranking path.
            single = Committee([paths[index]], [seed], metric, rule='single', physics_backend=cfg['physics_backend'])
            probe = np.linspace(0, len(vx)-1, 128).astype(int)
            np.testing.assert_allclose(single.predict(vx[probe]), predictions[index, probe], rtol=2e-7, atol=2e-5, equal_nan=True)
            chosen = dict(gamma=gamma, selected_seed=seed, validation_score=float(rank.iloc[0].selection_score),
                validation_pairs=len(vx), validation_ids=val_ids.tolist(), model_path=paths[index],
                model_sha256=records[index]['sha256'], selection='validation a priori only', test_used=False, posteriori_used=False)
            if (out / 'selection_frozen.json').exists() and json.loads((out / 'selection_frozen.json').read_text()) != chosen:
                raise ValueError('Frozen singleton changed on resume')
            write_json(out / 'selection_frozen.json', chosen)
            single.save(out / 'methods/single_best.json')
            for name, rule in [('mean10', 'mean_after_repair'), ('medoid10', 'medoid')]:
                p.rule = rule
                p.save(out / 'methods' / f'{name}.json')
            frozen_all[label] = chosen
            print('FROZEN_SINGLE', chosen, flush=True)
            progress('selection')
            del p, single, predictions
            gc.collect()
        write_json(root / 'all_selections_frozen.json', frozen_all)
        global_single = min(frozen_all.values(), key=lambda r: (r['validation_score'], r['gamma'], r['selected_seed']))
        write_json(root / 'global_single_frozen.json', global_single)
        print('FROZEN_COMMON_CONTROL', global_single, flush=True)
        summaries, details, species, timings, diagnostics, one_steps, checks = [], [], [], [], [], [], []
        for gamma in cfg['gammas']:
            label = f'g{gamma:g}'
            out = root / label
            for name in ('single_best', 'mean10', 'medoid10'):
                p = Committee.load(out / 'methods' / f'{name}.json')
                assert p.vector_physics.gamma == gamma
                for partition in ('val', 'test'):
                    budget()
                    # Full one-step evaluation, same samples for all methods.
                    px, py, ids = samples(data, partition)
                    pred = np.empty_like(py)
                    for begin in range(0, len(px), cfg['evaluation_batch_size']):
                        budget()
                        end = min(begin+cfg['evaluation_batch_size'], len(px))
                        pred[begin:end] = p.predict(px[begin:end])
                    common = dict(gamma=gamma, method=name, split=partition, horizon=cfg['horizon'])
                    one_steps.append(dict(**common, samples=len(px), **evaluate(py, pred, px, matrix)))
                    p0 = np.c_[data['states'][ids, 0], data['pressures'][ids]]
                    first = p.predict(p0)
                    p.timing = {k: 0. for k in p.timing}
                    p.vector_physics.counts = {k: 0 for k in p.vector_physics.counts}
                    print('ROLLOUT_START', gamma, name, partition, flush=True)
                    began = time.perf_counter()
                    def report(step, active):
                        budget()
                        print('ROLLOUT', gamma, name, partition, step, active, flush=True)
                    _, rows, sp, events, traces = run_rollouts(p, data['states'][ids], data['pressures'][ids], ids, matrix, cfg['dt'], report)
                    timings.append(dict(**common, seconds=time.perf_counter()-began, **p.timing))
                    diagnostics.append(dict(**common, **p.vector_physics.counts))
                    np.savez_compressed(out / 'traces' / f'{name}_{partition}.npz', **{f'trajectory_{int(i)}': t for i, t in zip(ids, traces)})
                    pd.DataFrame(events).to_csv(out / f'selection_{name}_{partition}.csv', index=False)
                    fine = pd.read_csv(Path(cfg['reference_root']) / f'reference_ignition_{partition}.csv').set_index('trajectory')
                    for row in rows:
                        tau = float(fine.loc[row['trajectory'], 'tau400'])
                        row['reference_tau400_refined'] = tau
                        row['tau400_relative_error_refined'] = abs(row['predicted_tau400']-tau)/tau if np.isfinite(tau) and row['prediction_ignites'] else np.nan
                    summary = horizon_summary(rows)
                    summary['median_ignition_error_refined'] = float(pd.DataFrame(rows).tau400_relative_error_refined.median())
                    summaries.append(dict(**common, **summary))
                    details.extend(dict(**common, **r) for r in rows)
                    species.extend(dict(**common, **{**r, 'species': str(data['species'][r['species']])}) for r in sp)
                    restored = Committee.load(out / 'methods' / f'{name}.json')
                    np.testing.assert_allclose(first, restored.predict(p0), rtol=1e-9, atol=1e-10, equal_nan=True)
                    np.testing.assert_allclose(first, np.array([t[1] if len(t)>1 else np.full(10, np.nan) for t in traces]), rtol=1e-9, atol=1e-10, equal_nan=True)
                    fixed = name != 'single_best' or all(e['selected_seed'] in (-1, frozen_all[label]['selected_seed']) for e in events)
                    assert fixed
                    checks.append(dict(**common, reload_match=True, first_step_match=True, singleton_seed_fixed=fixed, correction_gamma=p.vector_physics.gamma))
                    for filename, values in [('summary', summaries), ('trajectories', details), ('species_profiles', species), ('timing', timings),
                            ('correction_diagnostics', diagnostics), ('one_step', one_steps), ('reload_checks', checks)]:
                        pd.DataFrame(values).to_csv(root / f'{filename}.csv', index=False)
                    print('RESULT', common, summary, flush=True)
                    progress('evaluation')
                    del restored, events, pred
                del p
                gc.collect()
        assert hashes == {key: sha(cfg[key]) for key in hashes}
        write_json(root / 'manifest.json', dict(status='completed', input_hashes=hashes,
            fresh_members=20, represented_member_updates=20*cfg['updates_per_member'], train_pairs=len(x),
            validation_pairs=len(vx), test_pairs=len(samples(data, 'test')[0]),
            execution_seconds=spent+time.perf_counter()-started,
            source_hashes={str(p): sha(p) for p in sorted(Path('kinetics').glob('*.py'))}))
        progress('completed')
        print('COMPLETED', spent+time.perf_counter()-started, flush=True)
    except BaseException:
        progress('interrupted')
        raise


if __name__ == '__main__':
    main()
