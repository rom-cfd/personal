"""Frozen-weight Wang ANN-hard comparison; no training or test-set tuning.

Run: python -m kinetics.wang_study
"""
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
    parser.add_argument('--output', type=Path, default=Path('results/wang_hard10ms'))
    args = parser.parse_args()
    root = args.output
    if root.exists():
        raise FileExistsError('Preserve existing results; use a new output path')
    root.mkdir()
    for directory in ('methods', 'traces'):
        (root / directory).mkdir()
    baseline = Path('results/committee100_long10ms')
    config = json.loads((baseline / 'config.json').read_text())
    (root / 'config.json').write_text(json.dumps(config, indent=2))
    with np.load(config['source']) as data:
        states, pressures = data['states'], data['pressures']
        split, dt, matrix = data['split'], float(data['dt']), data['element_matrix']
        species_names = data['species']
    records = json.loads((baseline / 'members.json').read_text())
    for record in records:
        path = Path(record['path'] + '.keras')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Model changed: {path}')
    frozen = dict(status='running', paper_doi='10.1016/j.combustflame.2025.114105',
                  formulation='Eqs. 5-9; E=A(current-proposal), apply correction in Y',
                  negative_policy='reject entire candidate, no clipping or QP fallback',
                  gamma=.1, dt=dt, horizon=.01, baseline=str(baseline),
                  definitions=['mean10', 'medoid10', 'mean100', 'medoid100'],
                  weights_reused=True, training=False, parameter_tuning=False,
                  chemical_ode_at_inference=False, max_seconds=1800,
                  model_hashes={str(r['seed']): r['sha256'] for r in records},
                  source_sha256=hashlib.sha256(Path(config['source']).read_bytes()).hexdigest())
    (root / 'manifest.json').write_text(json.dumps(frozen, indent=2))
    started = time.perf_counter()
    summaries, details, species, timings, diagnostics, probes = [], [], [], [], [], []
    for size in (10, 100):
        p = Committee.load(baseline / 'methods' / f'medoid{size}.json')
        from .wang_physics import WangThermo
        p.physics_backend = 'wang_boxcox'
        p.vector_physics = WangThermo()
        for partition in ('val', 'test'):
            ids = np.flatnonzero(split == partition)
            fine = pd.read_csv(baseline / f'reference_ignition_{partition}.csv').set_index('trajectory')
            for prefix, rule in [('mean', 'mean_after_repair'), ('medoid', 'medoid')]:
                name = f'{prefix}{size}'
                p.rule = rule
                p.save(root / 'methods' / f'{name}.json')
                x0 = np.c_[states[ids, 0], pressures[ids]]
                p.predict(x0)
                p.timing = {k: 0. for k in p.timing}
                p.vector_physics.counts = {k: 0 for k in p.vector_physics.counts}
                begin = time.perf_counter()
                def progress(step, active):
                    print('PROGRESS', name, partition, step, active, flush=True)
                    if time.perf_counter() - started > 1800:
                        raise TimeoutError('CPU 30-minute cap reached')
                _, rows, sp, events, traces = run_rollouts(
                    p, states[ids], pressures[ids], ids, matrix, dt, progress)
                timings.append(dict(method=name, split=partition,
                                    seconds=time.perf_counter()-begin, **p.timing))
                diagnostics.append(dict(method=name, split=partition, **p.vector_physics.counts))
                np.savez_compressed(root / 'traces' / f'{name}_{partition}.npz',
                                    **{f'trajectory_{int(i)}': t for i, t in zip(ids, traces)})
                pd.DataFrame(events).to_csv(root / f'selection_{name}_{partition}.csv', index=False)
                for row in rows:
                    tau = float(fine.loc[row['trajectory'], 'tau400'])
                    row['reference_tau400_refined'] = tau
                    row['tau400_relative_error_refined'] = (
                        abs(row['predicted_tau400']-tau)/tau
                        if np.isfinite(tau) and row['prediction_ignites'] else np.nan)
                    # Incomplete/unignited predictions remain unresolved, not TN.
                summary = horizon_summary(rows)
                summary['median_ignition_error_refined'] = float(pd.DataFrame(rows).tau400_relative_error_refined.median())
                common = dict(method=name, split=partition, horizon=.01)
                summaries.append(dict(**common, **summary))
                details.extend(dict(**common, **row) for row in rows)
                species.extend(dict(**common, **{**row, 'species': str(species_names[row['species']])}) for row in sp)
                indices = np.unique(np.r_[np.arange(8), np.linspace(0, states.shape[1]-2, 24).astype(int)])
                px = np.c_[states[ids][:, indices].reshape(-1, 10), np.repeat(pressures[ids], len(indices))]
                py = states[ids][:, indices+1].reshape(-1, 10)
                pred = p.predict(px)
                probes.append(dict(**common, **evaluate(py, pred, px, matrix)))
                for filename, values in [('summary', summaries), ('trajectories', details),
                        ('species_profiles', species), ('timing', timings),
                        ('correction_diagnostics', diagnostics), ('one_step_probes', probes)]:
                    pd.DataFrame(values).to_csv(root / f'{filename}.csv', index=False)
                print('RESULT', name, partition, summary, diagnostics[-1], flush=True)
        del p
    old = pd.read_csv(baseline / 'summary.csv')
    old = old[np.isclose(old.horizon, .01)].copy()
    old['correction'] = 'previous_Y_weighted_nonnegative_QP'
    new = pd.DataFrame(summaries)
    new['correction'] = 'Wang_G_linearized'
    pd.concat([old, new], ignore_index=True).to_csv(root / 'comparison.csv', index=False)
    frozen.update(status='completed', execution_seconds=time.perf_counter()-started)
    (root / 'manifest.json').write_text(json.dumps(frozen, indent=2))
    print('COMPLETED', frozen['execution_seconds'], flush=True)


if __name__ == '__main__':
    main()
