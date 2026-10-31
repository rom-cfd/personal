"""Read completed new-run evaluations and emit separate, non-overwriting reports."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import SPECIES
from .ann_deeponet_study import check_deadline, write_json, verified_evaluation


def completed_evaluations(root):
    evaluations = []
    for marker in sorted((Path(root) / 'evaluations').glob('*/complete.json')):
        info = verified_evaluation(marker.parent)
        evaluations.append((marker.parent, info))
    return evaluations


def report(root, data, deadline=float('inf')):
    root = Path(root)
    evaluations = completed_evaluations(root)
    if not evaluations:
        raise ValueError('No completed evaluations; there are no performance results to report')
    destination = root / 'reports' / f'report_{time.time_ns()}'
    destination.mkdir(parents=True)
    rows = []
    for directory, info in evaluations:
        check_deadline(deadline)
        details = pd.read_csv(directory / 'trajectories.csv')
        physics = pd.read_csv(directory / 'physics.csv')
        events = pd.read_csv(directory / 'events.csv')
        timings = json.loads((directory / 'timing.json').read_text())
        for horizon, frame in details.groupby('horizon'):
            paired = frame.tau400_relative_error_refined.dropna()
            actual, predicted = frame.reference_ignites, frame.prediction_ignites
            rows.append(dict(method=info['method'], mode=info['mode'], split=info['split'],
                horizon=horizon, trajectories=len(frame), completion_fraction=float(frame.complete.mean()),
                physical_completion_fraction=float(frame.physically_admissible_complete.mean()),
                temperature_rmse_completed_only=float(frame.full_temperature_rmse.mean()),
                species_rmse_completed_only=float(frame.full_species_rmse.mean()),
                ignition_classification=float(frame.ignition_classification_correct.mean()),
                ignition_pairs=len(paired), ignition_relative_error_median=float(paired.median()),
                ignition_relative_error_p95=float(paired.quantile(.95)),
                true_positive=int((actual & predicted).sum()),
                false_negative=int((actual & ~predicted & frame.complete).sum()),
                false_positive=int((~actual & predicted).sum()),
                true_negative=int((~actual & ~predicted & frame.complete).sum()),
                unresolved=int(frame.unresolved_before_ignition.sum()),
                mass_drift_max=float(frame.mass_sum_max.max()), elemental_drift_max=float(frame.elemental_max.max()),
                full_trace_enthalpy_drift_max=float(physics.enthalpy_drift_max.max()),
                full_trace_entropy_decreasing_fraction=float(physics.entropy_decreasing_fraction.mean()),
                full_trace_invalid_candidate_fraction=float(1 - events.valid_count.sum() /
                                                 (len(events) * info['member_count'])),
                full_trace_valid_ann_mean=float(events.valid_ann.mean()),
                full_trace_valid_deeponet_mean=float(events.valid_deeponet.mean()),
                **timings))
    summary = pd.DataFrame(rows)
    summary.to_csv(destination / 'summary.csv', index=False)
    # Every temperature/species panel is one initial condition, with all methods
    # sharing only the reference. No trace is padded or extrapolated on failure.
    for split in sorted({i['split'] for _, i in evaluations}):
        ids = np.flatnonzero(data['split'] == split)
        for mode in ('raw', 'repaired'):
            available = [(d, i) for d, i in evaluations if i['split'] == split and i['mode'] == mode]
            if not available:
                continue
            loaded = []
            for d, info in available:
                with np.load(d / 'traces.npz') as f:
                    loaded.append((info['method'], {k: f[k] for k in f.files}))
            for component, label in enumerate(['T'] + SPECIES):
                check_deadline(deadline)
                columns = min(3, len(ids))
                count_rows = (len(ids) + columns - 1) // columns
                fig, axes = plt.subplots(count_rows, columns,
                                         figsize=(4.3 * columns, 2.7 * count_rows), squeeze=False)
                for axis, i in zip(axes.flat, ids):
                    truth = data['states'][i]
                    dt = float(data['dt'])
                    axis.plot(np.arange(len(truth)) * dt * 1000, truth[:, component], 'k--', label='Cantera')
                    for name, traces in loaded:
                        trace = traces[f'trajectory_{int(i)}']
                        axis.plot(np.arange(len(trace)) * dt * 1000, trace[:, component], label=name)
                    axis.set_title(f'IC {int(i)}')
                    axis.set_xlabel('Time [ms]')
                    axis.set_ylabel('T [K]' if component == 0 else f'Y_{label}')
                for axis in list(axes.flat)[len(ids):]:
                    axis.set_visible(False)
                axes.flat[0].legend(fontsize=7)
                fig.suptitle(f'{split} / {mode} / {label}; truncated lines are failed runs')
                fig.tight_layout()
                fig.savefig(destination / f'{split}_{mode}_{label}.png', dpi=120)
                plt.close(fig)
    deep = json.loads((root / 'deeponet_members.json').read_text())
    run = json.loads((root / 'run.json').read_text())
    ann = run['fingerprint']['ann_members']
    write_json(destination / 'training_cost.json', dict(
        ann_reused=True, ann_current_training_seconds=0,
        ann_historical_allocated_seconds=sum(r['group_seconds'] / r['group_size'] for r in ann),
        ann_cost_note='Allocated share of past group wall times, not a current matched-hardware measurement',
        deeponet_training_seconds=sum(r['train_seconds'] for r in deep),
        represented_updates_per_family=10 * run['config']['updates_per_member'],
        ann_parameters_per_member=9738, deeponet_parameters=[r['parameters'] for r in deep]))
    lines = ['# ANN10 / T/P-DeepONet10 / mixed20 comparison', '',
             'Only completed evaluations are included. Missing methods/splits are unrun, not passed.',
             'Raw and repaired are separate self-fed trajectories. Full-horizon errors exclude incomplete traces;',
             'read completion, physical validity and coverage alongside errors. Conservation imposed by repair',
             'is not evidence it was learned. This temperature/pressure-query model has finite-dimensional inputs,',
             'not arbitrary-function operator-learning validation.', '',
             'Columns prefixed full_trace and rollout timings refer to the full available rollout,',
             'including in the 2ms error-summary rows; they are not 2ms-only diagnostics.', '',
             'Mixed20 uses the same trained members, all at its own shared state each step. It costs more',
             'inference work than a 10-member team. Valid members have equal weight; invalid filtering',
             'can alter the effective ANN/DeepONet ratio. Events contain both family counts.', '',
             'See summary.csv, training_cost.json, and source evaluation directories for complete metrics.',
             'Temperature and all nine species figures show every IC and full available horizon.', '']
    for row in summary.itertuples():
        lines.append(f'- {row.split}/{row.mode}/{row.method}, {row.horizon * 1000:g} ms: '
                     f'completion={row.completion_fraction:.1%}, '
                     f'T RMSE (completed only)={row.temperature_rmse_completed_only:.6g} K, '
                     f'ignition classification={row.ignition_classification:.1%}, '
                     f'paired delays={row.ignition_pairs}.')
    (destination / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    check_deadline(deadline)
    write_json(destination / 'complete.json', dict(evaluations=[i for _, i in evaluations]))
    print('REPORT_COMPLETE', destination, flush=True)
    return destination
