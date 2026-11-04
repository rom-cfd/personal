"""Existing raw/repaired inference; validation choices frozen before test."""
import gc
import json
from pathlib import Path

import numpy as np
from .gamma_sweep_training import gamma_tag
from .gamma_sweep_study import load_groups
from .soft_constraint_evaluation import TEAMS
from .soft_constraint_study import event
from .p_only_training import sha256, write_json
from .deeponet_long_evaluation import evaluate_arm as ordinary_arm
from .tp_phi_evaluation import evaluate_arm as phi_arm
from scripts.nrmse_util_reference import _nrmse_1d


def trace_sources(root, cfg, snapshot, split, mode):
    result = {}
    for gamma in cfg['gammas']:
        for team, _ in TEAMS:
            if gamma == .1 and split == 'test':
                info = snapshot['baseline_test_traces'][f'{team}_{mode}']
                path, expected = Path(info['path']), info['sha256']
            else:
                markers = list((root / gamma_tag(gamma) / 'evaluations' / f'{team}_{mode}_{split}').glob('*/complete.json'))
                if len(markers) != 1:
                    raise ValueError('Require exactly one frozen evaluation per gamma/arm')
                path = markers[0].parent / 'traces.npz'
                expected = json.loads(markers[0].read_text())['output_sha256']['traces.npz']
            if sha256(path) != expected:
                raise ValueError('Saved trace changed')
            result[f'{gamma_tag(gamma)}_{team}'] = dict(path=str(path), sha256=expected)
    return result


def load_profiles(root, cfg, snapshot, data, split, mode):
    ids = np.flatnonzero(data['split'] == split).astype(int).tolist()
    sources = trace_sources(root, cfg, snapshot, split, mode); traces = {}
    for method, info in sources.items():
        with np.load(info['path']) as f:
            traces[method] = {i: f[f'trajectory_{i}'] for i in ids}
        for i, trace in traces[method].items():
            if trace.ndim != 2 or trace.shape[1] != 10 or not 1 <= len(trace) <= len(data['states'][i]) or not np.isfinite(trace).all():
                raise ValueError('Invalid trace')
            np.testing.assert_array_equal(trace[0], data['states'][i, 0])
    return traces, sources


def validation_choices(data, traces, gammas):
    """Completion first; compare tied maximum-coverage variants on identical ICs."""
    ids = np.flatnonzero(data['split'] == 'val').astype(int).tolist(); choices = {}
    for team, _ in TEAMS:
        completed = {g: {i for i in ids if len(traces[f'{gamma_tag(g)}_{team}'][i]) == len(data['states'][i])} for g in gammas}
        maximum = max(map(len, completed.values()))
        eligible = [g for g in gammas if len(completed[g]) == maximum]
        common = sorted(set.intersection(*(completed[g] for g in eligible)))
        scores = {str(g): float(np.median([_nrmse_1d(data['states'][i, :, 0], traces[f'{gamma_tag(g)}_{team}'][i][:, 0]) for i in common])) for g in eligible} if common else {}
        chosen = min(eligible, key=lambda g: (scores[str(g)], g)) if scores else None
        choices[team] = dict(chosen_gamma=chosen, maximum_completed_ICs=maximum,
            completion_counts={str(g): len(completed[g]) for g in gammas},
            eligible_gammas=eligible, common_ICs=common, common_IC_count=len(common), temperature_NRMSE=scores)
    return dict(split='val', mode='repaired', test_used=False,
        rule='Maximize complete10ms IC count; minimize median temperature NRMSE on common ICs among ties; exact score ties choose lower gamma',
        choices=choices)


def evaluate(root, cfg, snapshot, data, deadline):
    if len(json.loads((root / 'final_members.json').read_text())) != 240:
        raise ValueError('Freeze200 new and40 control endpoints before evaluation')
    for split in cfg['evaluation_splits']:
        if split == 'test' and not (root / 'validation_selection.json').exists():
            raise ValueError('Freeze validation choices before test')
        for gamma in cfg['gammas']:
            if gamma == .1 and split == 'test':
                for mode in cfg['modes']:
                    # Hash verification only; original traces remain at their source paths.
                    for team, _ in TEAMS:
                        info = snapshot['baseline_test_traces'][f'{team}_{mode}']
                        if sha256(info['path']) != info['sha256']:
                            raise ValueError('Control trace changed')
                        event(root, phase='evaluation_finished', gamma=gamma, method=team, split=split, mode=mode, cached=True)
                continue
            groups = load_groups(root, snapshot, gamma)
            for team, family in TEAMS:
                models = groups[family]; labels = ['ann'] * 10 if family == 'ann' else ['deeponet'] * 10
                if team.startswith('ann_'):
                    models = groups['ann'] + models; labels = ['ann'] * 10 + labels
                runner = phi_arm if family == 'tp_phi' else ordinary_arm
                for mode in cfg['modes']:
                    runner(root / gamma_tag(gamma), team, models, labels, mode, split, data,
                           snapshot['base_config'], snapshot['fingerprint'], deadline)
                    event(root, phase='evaluation_finished', gamma=gamma, method=team, split=split, mode=mode, cached=False)
            del groups, models; gc.collect()
        if split == 'val':
            traces, _ = load_profiles(root, cfg, snapshot, data, 'val', 'repaired')
            selected = validation_choices(data, traces, cfg['gammas'])
            path = root / 'validation_selection.json'
            if path.exists():
                if json.loads(path.read_text()) != selected:
                    raise ValueError('Frozen validation choice changed')
            else:
                write_json(path, selected)
            event(root, phase='validation_selection_frozen', choices=selected['choices'])
            del traces
