"""Read-only checkpoint audit. Run sequentially; outputs must be new paths.

This checks initial-condition first steps, not dense one-step or rollout accuracy.
Use --mode medoid with TF_ENABLE_ONEDNN_OPTS=0 and 1 in separate processes to
diagnose the historical reload discrepancy. It does not alter tolerances.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def comparison(actual, expected, temperature_atol, species_rtol, species_atol):
    import numpy as np
    t = np.isclose(actual[:, 0], expected[:, 0], rtol=1e-9,
                   atol=temperature_atol, equal_nan=True)
    y = np.isclose(actual[:, 1:], expected[:, 1:], rtol=species_rtol,
                   atol=species_atol, equal_nan=True)
    return dict(temperature_pass=bool(t.all()), species_pass=bool(y.all()),
                max_T_difference_K=float(np.nanmax(np.abs(actual[:, 0]-expected[:, 0]))),
                max_Y_difference=float(np.nanmax(np.abs(actual[:, 1:]-expected[:, 1:]))),
                temperature_failed_rows=np.flatnonzero(~t).tolist(),
                species_failed_coordinates=np.argwhere(~y).tolist(),
                temperature_rtol=1e-9, temperature_atol=temperature_atol,
                species_rtol=species_rtol, species_atol=species_atol)


def medoid(output):
    # Intentionally import TensorFlow first, as the old installation helper did.
    # oneDNN must be selected before process startup, not after this import.
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    from kinetics.committee import Committee
    started = time.perf_counter()
    root = Path('results/committee100_long10ms')
    with np.load('results/long10ms_data/trajectories.npz') as data:
        ids = np.flatnonzero(data['split'] == 'test')
        x = np.c_[data['states'][ids, 0], data['pressures'][ids]]
    p = Committee.load(root / 'methods/medoid100.json')
    actual = p.predict(x)
    chosen = p.last_diagnostics['selected_seed'].copy()
    candidates = p.repair(p.proposals(x), x)
    with np.load(root / 'traces/medoid100_test.npz') as trace:
        expected = np.array([trace[f'trajectory_{int(i)}'][1] for i in ids])
    events = pd.read_csv(root / 'selection_medoid100_test.csv')
    expected_seeds = events[events.step == 1].sort_values('trajectory').selected_seed.to_numpy()
    report = dict(onednn=os.environ.get('TF_ENABLE_ONEDNN_OPTS'),
                  tensorflow=tf.__version__, intra_threads=tf.config.threading.get_intra_op_parallelism_threads(),
                  inter_threads=tf.config.threading.get_inter_op_parallelism_threads(),
                  helper_tolerances=comparison(actual, expected, 1e-6, 1e-8, 1e-10),
                  repository_verifier_tolerances=comparison(actual, expected, 1e-10, 1e-9, 1e-10),
                  selected_seeds_identical=bool(np.array_equal(chosen, expected_seeds)),
                  trajectory_ids=ids.tolist(), elapsed_seconds=time.perf_counter()-started,
                  scope='Historical medoid reload diagnostic only; no new comparison arm')
    np.savez_compressed(output / 'arrays.npz', ids=ids, x=x, actual=actual,
                        expected=expected, selected_seed=chosen, candidates=candidates)
    dump(output / 'diagnostic.json', report)
    print(json.dumps(report, indent=2), flush=True)


def audit(output):
    import numpy as np
    import pandas as pd
    import h5py
    from kinetics.ann_deeponet_study import load_inputs, registered_members
    from kinetics.deeponet import ThermoDeepONet
    from kinetics.study import Predictor, tf
    from kinetics.mixed_ensemble import MixedMean, require_shared_preprocessing
    from kinetics.data import mechanism
    started = time.perf_counter()
    cfg = json.loads(Path('configs/ann_deeponet_mean10.json').read_text())
    data, pairs, fingerprint = load_inputs(cfg)
    root = Path(cfg['output'])
    saved = json.loads((root / 'run.json').read_text())
    assert saved == dict(config=cfg, fingerprint=fingerprint), 'Saved provenance differs'
    deep_records = registered_members(root, cfg, fingerprint)
    assert [r['seed'] for r in deep_records] == cfg['seeds']
    all_ann = json.loads(Path(cfg['ann_root'], 'members.json').read_text())
    assert len(all_ann) == 100 and len({r['seed'] for r in all_ann}) == 100
    base_meta = json.loads(Path(all_ann[0]['path'] + '.json').read_text())
    file_rows = []
    for family, records in [('ann', all_ann), ('T_P', deep_records)]:
        for record in records:
            assert record['updates'] == 10000
            assert record['train_data_sha256'] == fingerprint['pair_sha256']
            path = Path(record['path'] + '.keras')
            assert digest(path) == record['sha256']
            meta_path = Path(record['path'] + '.json')
            meta = json.loads(meta_path.read_text())
            # ANN's spec also has a display-only 'name'; compare the actual
            # transform settings rather than requiring identical labels.
            assert meta['spec']['kind'] == base_meta['spec']['kind'] == 'boxcox'
            assert meta['spec']['gamma'] == base_meta['spec']['gamma'] == .1
            assert not meta['spec'].get('species', False)
            for key in ('epsilon', 'output_mean', 'output_scale', 'input_mean', 'input_scale'):
                assert meta[key] == base_meta[key], (meta_path, key)
            # ANN manifests lack preprocessing hashes; check those bytes against
            # the checkpoint's Git blobs as well as the train-only refit above.
            git_bytes = subprocess.check_output(['git', 'show', 'HEAD:' + str(meta_path)])
            assert hashlib.sha256(git_bytes).hexdigest() == digest(meta_path)
            with zipfile.ZipFile(path) as archive:
                with archive.open('model.weights.h5') as handle, h5py.File(handle) as weights:
                    optimizer_saved = 'optimizer' in weights
            file_rows.append(dict(family=family, seed=record['seed'], path=str(path),
                                  sha256=digest(path), preprocessing_sha256=digest(meta_path),
                                  optimizer_group_saved=optimizer_saved))
    pd.DataFrame(file_rows).to_csv(output / 'verified_models.csv', index=False)
    missing = [p for p in ('scripts/deeponet_followups.py',
        'results/deeponet_followups/run_002_sequential', 'results/deeponet_continuation/run_001',
        'scripts/p_only_long_training.py', 'configs/p_only_long_training.json',
        'results/p_only_long_training/run_001', 'results/time_profile_figures') if not Path(p).exists()]
    lock_mismatches = []
    for line in Path('requirements.lock.txt').read_text().splitlines():
        name, expected = line.split('==')
        actual = importlib.metadata.version(name)
        if actual != expected:
            lock_mismatches.append([name, expected, actual])
    assert not lock_mismatches
    load_start = time.perf_counter()
    ann = [Predictor.load(r['path']) for r in fingerprint['ann_members']]
    deep = [ThermoDeepONet.load(r['path']) for r in deep_records]
    require_shared_preprocessing(ann + deep)
    assert {p.model.count_params() for p in ann} == {9738}
    assert {p.model.count_params() for p in deep} == {9687}
    loading = time.perf_counter()-load_start
    ids = np.flatnonzero(data['split'] == 'test')
    x = np.c_[data['states'][ids, 0], data['pressures'][ids]]
    arrays = dict(ids=ids, x=x, cantera_next=data['states'][ids, 1])
    observations = []
    for name, members, families in [('ann10', ann, ['ann']*10),
            ('T_P10', deep, ['deeponet']*10),
            ('mixed20', ann+deep, ['ann']*10+['deeponet']*10)]:
        for mode in ('raw', 'repaired'):
            p = MixedMean(members, families, mode)
            warm_start = time.perf_counter()
            p.predict(x)
            warm = time.perf_counter()-warm_start
            p.reset_timing()
            actual = p.predict(x)
            arrays[f'{name}_{mode}'] = actual
            row = dict(team=name, mode=mode, initial_condition_count=len(ids),
                finite_rows=int(np.isfinite(actual).all(axis=1).sum()),
                valid_count=p.last_diagnostics['valid_count'].tolist(),
                valid_ann=p.last_diagnostics['valid_ann'].tolist(),
                valid_deeponet=p.last_diagnostics['valid_deeponet'].tolist(),
                warmup_seconds=warm, timing=p.timing.copy())
            if name == 'ann10' and mode == 'repaired':
                with np.load(Path(cfg['ann_root'], 'traces/mean10_test.npz')) as saved_trace:
                    expected = np.array([saved_trace[f'trajectory_{int(i)}'][1] for i in ids])
                row['historical_mean10_reload'] = comparison(actual, expected, 1e-10, 1e-9, 1e-10)
            observations.append(row)
            del p
            gc.collect()
    np.savez_compressed(output / 'initial_condition_first_steps.npz', **arrays)
    gas = mechanism()
    report = dict(git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], text=True),
        python=platform.python_version(), versions={name: importlib.metadata.version(name) for name in
            ('cantera', 'tensorflow_cpu', 'keras', 'numpy', 'scipy', 'pandas', 'matplotlib', 'pytest')},
        thread_settings={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'TF_ENABLE_ONEDNN_OPTS')},
        tensorflow_threads=dict(intra=tf.config.threading.get_intra_op_parallelism_threads(),
                               inter=tf.config.threading.get_inter_op_parallelism_threads()),
        gpu_devices=[p.name for p in tf.config.list_physical_devices('GPU')],
        data_shape=list(data['states'].shape), selected_pairs=len(pairs['x']),
        splits={s: int((data['split'] == s).sum()) for s in ('train', 'val', 'test')},
        dt=float(data['dt']), species=gas.species_names, reactions=gas.n_reactions,
        hashes={p: digest(p) for p in ('requirements.lock.txt', cfg['source'], cfg['training_pairs'])},
        checked_ann_models=len(all_ann), checked_T_P_models=len(deep_records),
        saved_run_fingerprint_identical=True, train_only_preprocessing_refit_identical=True,
        missing_followup_paths=missing, shared_loading_seconds=loading,
        elapsed_seconds=time.perf_counter()-started, first_steps=observations,
        scope='Initial 18 test states only; not dense one-step, not full 10 ms rollout',
        source_sha256=digest(__file__))
    dump(output / 'baseline.json', report)
    print('AUDIT_COMPLETE', output, 'seconds', report['elapsed_seconds'], flush=True)


def plots(output):
    import matplotlib.pyplot as plt
    from plot_ann_deeponet import STYLE, training
    with plt.rc_context(STYLE):
        training(Path('results/ann_deeponet_mean10/run_001'), output)
    dump(output / 'provenance.json', dict(source_sha256=digest(__file__),
         plotting_source_sha256=digest(Path(__file__).with_name('plot_ann_deeponet.py')),
         note='Observed mini-batch loss only; no interpolation or full-train/validation MSE claim',
         output_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file()}))
    print('HISTORY_PLOTS_COMPLETE', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('audit', 'medoid', 'plots'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    {'audit': audit, 'medoid': medoid, 'plots': plots}[args.mode](args.output)


if __name__ == '__main__':
    main()
