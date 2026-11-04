"""Frozen120 long endpoints; original physics and same-state ensemble rules."""
import gc
import json
import time
from .element_long_evaluation import load_groups
from .penalty_sweep_evaluation import member_diagnostics
from .penalty_sweep_study import weight_root
from .soft_constraint_evaluation import TEAMS
from .deeponet_long_evaluation import evaluate_arm as ordinary_arm
from .tp_phi_evaluation import evaluate_arm as phi_arm
from .p_only_training import write_json
from .soft_constraint_study import event


def evaluate(root, cfg, snapshot, data, pairs, deadline):
    records = json.loads((root / 'final_members.json').read_text())
    if len(records) != 120:
        raise ValueError('Freeze all120 endpoints before test evaluation')
    for w in cfg['lambdas']:
        child = weight_root(root, w); started = time.perf_counter()
        groups = load_groups(child, snapshot)
        if not (child / 'model_loading.json').exists():
            write_json(child / 'model_loading.json', dict(seconds=time.perf_counter() - started, members=40))
        member_diagnostics(child, w, snapshot, groups, data, pairs, deadline)
        for team, family in TEAMS:
            models = groups[family]; labels = ['ann'] * 10 if family == 'ann' else ['deeponet'] * 10
            if team.startswith('ann_'):
                models = groups['ann'] + models; labels = ['ann'] * 10 + labels
            runner = phi_arm if family == 'tp_phi' else ordinary_arm
            for mode in cfg['modes']:
                runner(child, team, models, labels, mode, 'test', data,
                       snapshot['base_config'], snapshot['fingerprint'], deadline)
                event(root, phase='evaluation_finished', lambda_value=w, method=team, mode=mode)
        del groups, models; gc.collect()
