"""Evaluate frozen matched10k weight groups with the original raw/repaired rules."""
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from .ann_deeponet_study import check_deadline, verified_evaluation
from .data import samples
from .deeponet_long_evaluation import evaluate_arm as ordinary_arm
from .tp_phi_evaluation import evaluate_arm as phi_arm
from .soft_constraint_evaluation import load_groups, TEAMS
from .soft_constraint_training import SoftObjective
from .soft_constraint_study import inputs_for, event
from .penalty_sweep_study import weight_root
from .p_only_training import sha256, write_json
from .study import tf


def member_diagnostics(child, weight, snapshot, groups, data, pairs, deadline):
    destination = child / 'member_diagnostics'
    if (destination / 'complete.json').exists():
        verified_evaluation(destination); return
    destination.mkdir(parents=True, exist_ok=False); rows = []
    for family, models in groups.items():
        objectives = [SoftObjective(p, snapshot['element_matrix'], weight) for p in models]
        width = 12 if family == 'tp_phi' else 11
        @tf.function(input_signature=[tf.TensorSpec([None, width], tf.float32)])
        def forward(features):
            return tf.stack([p.model(features, training=False) for p in models])
        for split in ('train', 'val'):
            if split == 'train':
                x, y = inputs_for(family, pairs, data), pairs['y']
            else:
                x, y, ids = samples(data, 'val')
                if family == 'tp_phi':
                    x = np.c_[x[:, :10], np.repeat(data['initials'][ids, 2], data['states'].shape[1] - 1), x[:, -1]]
            p = models[0]; current = p.norm.forward(p.encode_state(x[:, :10]))
            target = (p.norm.forward(p.encode_state(y)) - current).astype('float32')
            features = p.features(x).astype('float32')
            totals = np.zeros((len(models), 3)); invalid = np.zeros(len(models), int)
            for start in range(0, len(x), 512):
                check_deadline(deadline); end = min(start + 512, len(x)); outputs = forward(features[start:end])
                for k, (delta, objective) in enumerate(zip(outputs, objectives)):
                    _, values = objective.components(delta, tf.constant(target[start:end]),
                        tf.constant(current[start:end], tf.float64), tf.constant(y[start:end, 1:], tf.float64))
                    totals[k] += values.numpy() * (end - start)
                    z = p.norm.inverse(current[start:end] + delta.numpy().astype('float64'))[:, 1:]
                    invalid[k] += int(np.any(z < 0, axis=1).sum())
            for p, (total, mse, element), bad in zip(models, totals / len(x), invalid):
                rows.append(dict(lambda_value=weight, family=family, seed=p.seed_label, split=split, pairs=len(x),
                    objective=float(total), data_MSE=float(mse), physical_element_L1=float(element),
                    weighted_element_L1=float(weight * element), invalid_inverse_row_fraction=float(bad / len(x))))
        event(child, phase='member_diagnostics_finished', lambda_value=weight, family=family)
    pd.DataFrame(rows).to_csv(destination / 'loss_components.csv', index=False)
    write_json(destination / 'complete.json', dict(output_sha256={p.name: sha256(p) for p in destination.iterdir()}))


def evaluate(root, cfg, snapshot, data, pairs, deadline):
    if len(json.loads((root / 'final_members.json').read_text())) != 120:
        raise ValueError('Freeze all120 endpoints before reading test outcomes')
    for weight in cfg['lambdas']:
        child = weight_root(root, weight); before = time.perf_counter()
        groups = load_groups(child, snapshot)
        if not (child / 'model_loading.json').exists():
            write_json(child / 'model_loading.json', dict(seconds=time.perf_counter() - before, members=40))
        member_diagnostics(child, weight, snapshot, groups, data, pairs, deadline)
        for team, family in TEAMS:
            models = groups[family]; labels = ['ann'] * 10 if family == 'ann' else ['deeponet'] * 10
            if team.startswith('ann_'):
                models = groups['ann'] + models; labels = ['ann'] * 10 + labels
            runner = phi_arm if family == 'tp_phi' else ordinary_arm
            for mode in cfg['modes']:
                runner(child, team, models, labels, mode, 'test', data,
                       snapshot['base_config'], snapshot['fingerprint'], deadline)
                event(root, phase='evaluation_finished', lambda_value=weight, method=team, mode=mode)
        del groups, models; gc.collect()
