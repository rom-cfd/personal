"""Budgeted ANN10 / T/P-DeepONet10 / mixed20 fixed-step comparison.

Default phase is prepare (audit only). Training/evaluation require an explicit
wall-time budget and separate user authorization. No existing result is edited.
"""
import argparse
import gc
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from .deeponet import ThermoDeepONet, train_member
from .study import Predictor
from .transforms import Transform, Standardizer
from .mixed_ensemble import MixedMean, rollout, require_shared_preprocessing
from .metrics import evaluate as one_step_metrics
from .posteriori import metrics as trajectory_metrics
from .data import mechanism, SPECIES


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def verified_evaluation(directory):
    directory = Path(directory)
    info = json.loads((directory / 'complete.json').read_text())
    for name, expected in info['output_sha256'].items():
        if sha256(directory / name) != expected:
            raise ValueError(f'Evaluation output changed: {directory / name}')
    return info


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError('Experiment wall-time budget exhausted')


def validate_pairs(data, pairs, dt, horizon):
    if float(data['dt']) != dt or float(pairs['dt']) != dt:
        raise ValueError('Every supervised pair must have the same fixed dt')
    if not np.isclose((data['states'].shape[1] - 1) * dt, horizon):
        raise ValueError('Dense data horizon differs from configuration')
    if list(data['species']) != SPECIES:
        raise ValueError('Species order differs')
    parent = pairs['parent']
    if not set(parent[:, 0]) <= set(np.flatnonzero(data['split'] == 'train')):
        raise ValueError('Training pairs include non-training trajectories')
    if np.any(parent[:, 1] < 0) or np.any(parent[:, 1] >= data['states'].shape[1] - 1):
        raise ValueError('Invalid parent time indices')
    np.testing.assert_array_equal(pairs['x'][:, :10], data['states'][parent[:, 0], parent[:, 1]])
    np.testing.assert_array_equal(pairs['x'][:, -1], data['pressures'][parent[:, 0]])
    np.testing.assert_array_equal(pairs['y'], data['states'][parent[:, 0], parent[:, 1] + 1])


def load_inputs(cfg):
    if cfg['dt'] != 2e-6 or cfg['horizon'] != .01:
        raise ValueError('This experiment keeps the agreed 2us / 10ms problem')
    if len(cfg['seeds']) != 10 or len(set(cfg['seeds'])) != 10:
        raise ValueError('Exactly ten distinct paired seeds required')
    if cfg['modes'] != ['raw', 'repaired'] or cfg['physics_backend'] != 'vectorized_nasa':
        raise ValueError('Use the common raw/repaired comparison and vectorized physics')
    if cfg['test_used_for_selection'] or cfg['one_step_chunk_size'] < 1:
        raise ValueError('Test-driven selection and nonpositive chunks are forbidden')
    if cfg['input_partition'] != 'current_Y9__current_T_logP':
        raise ValueError('Expected current Y9 branch and current T/log(P) trunk')
    with np.load(cfg['source']) as f:
        data = {k: f[k] for k in f.files}
    with np.load(cfg['training_pairs']) as f:
        pairs = {k: f[k] for k in f.files}
    validate_pairs(data, pairs, cfg['dt'], cfg['horizon'])
    if data['states'].shape != (90, 5001, 10) or len(pairs['x']) != 8815:
        raise ValueError('Expected the audited 90-trajectory / 8815-pair benchmark')
    if [(data['split'] == s).sum() for s in ('train', 'val', 'test')] != [54, 18, 18]:
        raise ValueError('Expected the original trajectory-disjoint 54/18/18 split')
    ann_root = Path(cfg['ann_root'])
    ann_cfg = json.loads((ann_root / 'config.json').read_text())
    for key in ('dt', 'horizon', 'gamma', 'batch_size', 'learning_rate', 'updates_per_member'):
        if cfg[key] != ann_cfg[key]:
            raise ValueError(f'ANN reuse condition differs: {key}')
    if ann_cfg['hidden_layers'] != [64, 64, 64] or ann_cfg['activation'] != 'tanh':
        raise ValueError('Expected the original 64/64/64 tanh ANN')
    ann_manifest = json.loads((ann_root / 'manifest.json').read_text())
    pair_hash, dense_hash = sha256(cfg['training_pairs']), sha256(cfg['source'])
    if (pair_hash != ann_manifest['train_pair_sha256'] or
            dense_hash != ann_manifest['dense_sha256']):
        raise ValueError('Data hashes differ from the ANN experiment')
    all_records = json.loads((ann_root / 'members.json').read_text())
    if len({r['seed'] for r in all_records}) != len(all_records):
        raise ValueError('Duplicate ANN member seeds')
    by_seed = {r['seed']: r for r in all_records}
    transform = Transform('boxcox', cfg['gamma'])
    x, y = pairs['x'], pairs['y']
    a = np.c_[x[:, 0] / 1000, transform.forward(x[:, 1:10])]
    b = np.c_[y[:, 0] / 1000, transform.forward(y[:, 1:])]
    norm = Standardizer().fit(np.r_[a, b])
    inp = Standardizer().fit(np.c_[norm.forward(a), np.log(x[:, -1])])
    records = []
    for seed in cfg['seeds']:
        r = dict(by_seed[seed])
        if r['updates'] != cfg['updates_per_member'] or r['train_data_sha256'] != pair_hash:
            raise ValueError('ANN member has a different training budget or dataset')
        if sha256(r['path'] + '.keras') != r['sha256']:
            raise ValueError('ANN model hash mismatch')
        meta = json.loads(Path(r['path'] + '.json').read_text())
        if meta['spec']['kind'] != 'boxcox' or meta['spec']['gamma'] != cfg['gamma'] or meta['epsilon'] != 1e-8:
            raise ValueError('ANN transformation differs')
        for key, value in [('output_mean', norm.mean), ('output_scale', norm.scale),
                           ('input_mean', inp.mean), ('input_scale', inp.scale)]:
            np.testing.assert_array_equal(meta[key], value)
        r['preprocessing_sha256'] = sha256(r['path'] + '.json')
        records.append(r)
    sources = {str(p): sha256(p) for p in Path('kinetics').glob('*.py')}
    fine = {}
    for split in ('val', 'test'):
        path = ann_root / f'reference_ignition_{split}.csv'
        fine[split] = dict(path=str(path), sha256=sha256(path))
    fingerprint = dict(pair_sha256=pair_hash, dense_sha256=dense_hash, ann_members=records,
                       reference_ignition=fine, code_sha256=sources)
    return data, pairs, fingerprint


def prepare_run(cfg, fingerprint, resume=False):
    root = Path(cfg['output'])
    namespace = Path('results/ann_deeponet_mean10').resolve()
    if not root.resolve().is_relative_to(namespace) or root.resolve() == namespace:
        raise ValueError('Use a new run below results/ann_deeponet_mean10; preserve original results')
    snapshot = dict(config=cfg, fingerprint=fingerprint)
    if root.exists():
        if not resume:
            raise FileExistsError('Output exists; use --resume with identical inputs/code or a new run')
        if json.loads((root / 'run.json').read_text()) != snapshot:
            raise ValueError('Resume inputs, configuration or code changed')
    else:
        root.mkdir(parents=True)
        write_json(root / 'run.json', snapshot)
        write_json(root / 'environment.json', dict(
            python=platform.python_version(), libraries={n: version(n) for n in
                ('cantera', 'tensorflow_cpu', 'keras', 'numpy', 'scipy', 'pandas')},
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            git_status=subprocess.check_output(['git', 'status', '--short'], text=True),
            test_used_for_selection=False,
            methods=['ann10', 'deeponet10', 'mixed20'],
            architecture_note='Finite-dimensional temperature/pressure-query surrogate; not arbitrary-function operator learning'))
        (root / 'models').mkdir()
        write_json(root / 'deeponet_members.json', [])
    return root


def registered_members(root, cfg, fingerprint):
    records = json.loads((root / 'deeponet_members.json').read_text())
    if len({r['seed'] for r in records}) != len(records):
        raise ValueError('Duplicate DeepONet member')
    for r in records:
        if (r['seed'] not in cfg['seeds'] or r['updates'] != cfg['updates_per_member'] or
                r['train_data_sha256'] != fingerprint['pair_sha256'] or
                sha256(r['path'] + '.keras') != r['sha256'] or
                sha256(r['path'] + '.json') != r['preprocessing_sha256']):
            raise ValueError('Registered DeepONet provenance/hash mismatch')
    return records


def train(root, cfg, pairs, fingerprint, deadline):
    records = registered_members(root, cfg, fingerprint)
    for seed in cfg['seeds']:
        if seed in {r['seed'] for r in records}:
            continue
        path = root / 'models' / f'thermo_seed{seed}'
        if any(Path(str(path) + suffix).exists() for suffix in ('.keras', '.json')):
            raise FileExistsError('Unregistered partial model exists; do not overwrite it')
        check_deadline(deadline)
        print('TRAIN_MEMBER', seed, flush=True)
        p, seconds, history = train_member(dict(kind='boxcox', gamma=cfg['gamma']), seed,
            cfg['updates_per_member'], pairs['x'], pairs['y'], cfg, deadline)
        p.save(path)
        record = dict(seed=seed, path=str(path), updates=cfg['updates_per_member'],
                      train_seconds=seconds, parameters=p.model.count_params(),
                      train_data_sha256=fingerprint['pair_sha256'], sha256=sha256(str(path) + '.keras'),
                      preprocessing_sha256=sha256(str(path) + '.json'))
        pd.DataFrame(history).to_csv(root / 'models' / f'history_seed{seed}.csv', index=False)
        records.append(record)
        write_json(root / 'deeponet_members.json', records)
        del p
        gc.collect()


def physical_diagnostics(trace, truth, pressure, equilibrium, thermo, deadline):
    """Never normalize an invalid raw composition to manufacture valid diagnostics."""
    gas = mechanism()
    h = np.full(len(trace), np.nan)
    s = np.full(len(trace), np.nan)
    for j, state in enumerate(trace):
        if j % 100 == 0:
            check_deadline(deadline)
        y = state[1:]
        if (not np.isfinite(state).all() or (y < 0).any() or
                abs(y.sum() - 1) > 1e-8 or not 200 <= state[0] <= 6000):
            continue
        hs, _ = thermo.species_h_cp(np.array([state[0]]))
        h[j] = y @ hs[0]
        gas.TPY = state[0], pressure, y
        s[j] = gas.entropy_mass
    ds = np.diff(s)
    ds = ds[np.isfinite(ds)]
    dh = h - h[0]
    finite_h = dh[np.isfinite(dh)]
    delta_s = s - s[0]
    finite_s = delta_s[np.isfinite(delta_s)]
    complete = len(trace) == len(truth)
    return dict(thermo_invalid_points=int((~np.isfinite(s)).sum()),
        enthalpy_drift_max=float(np.max(np.abs(finite_h))) if len(finite_h) else np.nan,
        entropy_decreasing_fraction=float(np.mean(ds < -1e-6)) if len(ds) else np.nan,
        entropy_valid_step_count=len(ds), entropy_tolerance=1e-6,
        largest_entropy_step_drop=float(max(0., -ds.min())) if len(ds) else np.nan,
        minimum_entropy_delta=float(finite_s.min()) if len(finite_s) else np.nan,
        final_temperature_error=float(trace[-1, 0] - truth[-1, 0]) if complete else np.nan,
        final_equilibrium_temperature_gap=float(trace[-1, 0] - equilibrium[0]) if complete else np.nan,
        minimum_temperature=float(trace[:, 0].min()), maximum_temperature=float(trace[:, 0].max()))


def one_steps(predictor, states, pressures, matrix, chunk_size, deadline):
    x = np.concatenate([states[:, :-1], np.broadcast_to(pressures[:, None, None],
                       (*states[:, :-1].shape[:2], 1))], axis=-1).reshape(-1, 11)
    target = states[:, 1:].reshape(-1, 10)
    predicted = np.full_like(target, np.nan)
    valid_ann = valid_deeponet = total_candidates = 0
    for start in range(0, len(x), chunk_size):
        check_deadline(deadline)
        end = start + chunk_size
        predicted[start:end] = predictor.predict(x[start:end])
        valid_ann += int(predictor.last_diagnostics['valid_ann'].sum())
        valid_deeponet += int(predictor.last_diagnostics['valid_deeponet'].sum())
        total_candidates += len(x[start:end]) * len(predictor.members)
    result = one_step_metrics(target, predicted, x, matrix)
    result.update(pair_count=len(x), valid_ann=valid_ann, valid_deeponet=valid_deeponet,
                  invalid_candidate_fraction=1 - (valid_ann + valid_deeponet) / total_candidates)
    finite = np.isfinite(predicted).all(axis=1)
    species = []
    for k, name in enumerate(SPECIES):
        error = predicted[finite, k + 1] - target[finite, k + 1]
        species.append(dict(species=name, finite_rows=int(finite.sum()), total_rows=len(x),
                            rmse=float(np.sqrt(np.mean(error ** 2))) if len(error) else np.nan))
    return result, species


def evaluate(root, cfg, data, fingerprint, split, deadline):
    records = registered_members(root, cfg, fingerprint)
    if {r['seed'] for r in records} != set(cfg['seeds']):
        raise ValueError('All ten DeepONets must finish before comparison')
    check_deadline(deadline)
    load_started = time.perf_counter()
    ann = [Predictor.load(r['path']) for r in fingerprint['ann_members']]
    by_seed = {r['seed']: r for r in records}
    deep = [ThermoDeepONet.load(by_seed[s]['path']) for s in cfg['seeds']]
    require_shared_preprocessing(ann + deep)
    loading_seconds = time.perf_counter() - load_started
    ids = np.flatnonzero(data['split'] == split)
    states, pressures = data['states'][ids], data['pressures'][ids]
    fine_info = fingerprint['reference_ignition'][split]
    if sha256(fine_info['path']) != fine_info['sha256']:
        raise ValueError('Fine ignition reference changed')
    fine = pd.read_csv(fine_info['path']).set_index('trajectory')
    if set(fine.index) != set(ids) or not np.allclose(fine.reference_dt, cfg['dt'] / 10):
        raise ValueError('Fine ignition reference ICs or resolution differ')
    definitions = [('ann10', ann, ['ann'] * 10), ('deeponet10', deep, ['deeponet'] * 10),
                   ('mixed20', ann + deep, ['ann'] * 10 + ['deeponet'] * 10)]
    for name, members, families in definitions:
        for mode in cfg['modes']:
            destination = root / 'evaluations' / f'{name}_{mode}_{split}'
            if destination.exists():
                if (destination / 'complete.json').exists():
                    verified_evaluation(destination)
                    continue
                raise FileExistsError('Partial evaluation output exists; preserve it and use a new run')
            check_deadline(deadline)
            p = MixedMean(members, families, mode)
            started = time.perf_counter()
            p.predict(np.c_[states[:, 0], pressures])
            warmup_seconds = time.perf_counter() - started
            p.reset_timing()
            started = time.perf_counter()
            traces, events = rollout(p, states[:, 0], pressures, states.shape[1] - 1,
                                      cfg['dt'], ids, deadline)
            rollout_seconds = time.perf_counter() - started
            rollout_timing = dict(p.timing)
            diagnostics_started = time.perf_counter()
            rows, species_rows, diagnostics = [], [], []
            for i, truth, trace, pressure in zip(ids, states, traces, pressures):
                check_deadline(deadline)
                for horizon in (.002, cfg['horizon']):
                    row, species = trajectory_metrics(trace, truth, cfg['dt'], data['element_matrix'],
                                                       round(horizon / cfg['dt']))
                    tau = float(fine.loc[i, 'tau400'])
                    tau = tau if tau <= horizon else np.nan
                    pred_tau = row['predicted_tau400']
                    row.update(trajectory=int(i), horizon=horizon, reference_tau400_refined=tau,
                        tau400_relative_error_refined=abs(pred_tau - tau) / tau
                            if np.isfinite(tau) and np.isfinite(pred_tau) else np.nan)
                    row['reference_ignites'] = bool(np.isfinite(row['reference_tau400']))
                    row['prediction_ignites'] = bool(np.isfinite(pred_tau))
                    row['unresolved_before_ignition'] = bool(not row['complete'] and not row['prediction_ignites'])
                    row['ignition_classification_correct'] = bool(
                        row['reference_ignites'] == row['prediction_ignites']
                        and (row['prediction_ignites'] or row['complete']))
                    rows.append(row)
                    species_rows.extend(dict(trajectory=int(i), horizon=horizon,
                        **{**r, 'species': SPECIES[r['species']]}) for r in species)
                diagnostics.append(dict(trajectory=int(i), **physical_diagnostics(
                    trace, truth, pressure, data['equilibrium'][i], p.physics, deadline)))
            diagnostics_seconds = time.perf_counter() - diagnostics_started
            started = time.perf_counter()
            one_step, one_species = one_steps(p, states, pressures, data['element_matrix'],
                                              cfg['one_step_chunk_size'], deadline)
            one_step_seconds = time.perf_counter() - started
            check_deadline(deadline)
            destination.mkdir(parents=True)
            np.savez_compressed(destination / 'traces.npz',
                                **{f'trajectory_{int(i)}': t for i, t in zip(ids, traces)})
            pd.DataFrame(events).to_csv(destination / 'events.csv', index=False)
            pd.DataFrame(rows).to_csv(destination / 'trajectories.csv', index=False)
            pd.DataFrame(species_rows).to_csv(destination / 'species_profiles.csv', index=False)
            pd.DataFrame(diagnostics).to_csv(destination / 'physics.csv', index=False)
            pd.DataFrame(one_species).to_csv(destination / 'one_step_species.csv', index=False)
            write_json(destination / 'one_step.json', one_step)
            write_json(destination / 'timing.json', dict(warmup_seconds=warmup_seconds,
                       shared_20_member_loading_seconds=loading_seconds,
                       diagnostics_seconds=diagnostics_seconds,
                       rollout_seconds=rollout_seconds, one_step_seconds=one_step_seconds, **rollout_timing))
            outputs = {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}
            write_json(destination / 'complete.json', dict(method=name, mode=mode, split=split,
                       member_count=len(members), output_sha256=outputs))
            print('EVALUATED', name, mode, split, flush=True)
            del p
            gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ann_deeponet_mean10.json')
    parser.add_argument('--phase', choices=('prepare', 'train', 'evaluate', 'report'), default='prepare')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--budget-seconds', type=float)
    args = parser.parse_args(argv)
    if args.phase != 'prepare' and (args.budget_seconds is None or
            not np.isfinite(args.budget_seconds) or args.budget_seconds <= 0):
        parser.error('train/evaluate/report require an explicitly approved positive --budget-seconds')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
    if branch != 'bw':
        parser.error('Use the existing bw checkout; this runner never switches or resets branches')
    began = time.perf_counter()
    deadline = began + args.budget_seconds if args.budget_seconds else float('inf')
    cfg = json.loads(Path(args.config).read_text())
    data, pairs, fingerprint = load_inputs(cfg)
    root = prepare_run(cfg, fingerprint, args.resume)
    try:
        check_deadline(deadline)
        if args.phase == 'train':
            train(root, cfg, pairs, fingerprint, deadline)
        elif args.phase == 'evaluate':
            evaluate(root, cfg, data, fingerprint, args.split, deadline)
        elif args.phase == 'report':
            from .ann_deeponet_report import report
            report(root, data, deadline)
        print('PHASE_COMPLETE', args.phase, root, flush=True)
    finally:
        with (root / 'execution.jsonl').open('a') as f:
            f.write(json.dumps(dict(phase=args.phase, split=args.split,
                    elapsed_seconds=time.perf_counter() - began, budget_seconds=args.budget_seconds)) + '\n')


if __name__ == '__main__':
    main()
