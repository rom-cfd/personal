"""Prepare or explicitly budget sequential P-only 10 x 10k baseline training.

The default only audits data and saves an UNTRAINED initial model. No test
evaluation or long training is performed. Existing result paths are preserved.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from importlib.metadata import version

import numpy as np
from .ann_deeponet_study import load_inputs
from .p_only_deeponet import PressureDeepONet
from .p_only_training import PressureTrainer, sha256, write_json
from .study import tf


def validate_config(cfg):
    expected = dict(input_partition='current_T_Y9__logP', branch_layers=[64,64],
        trunk_layers=[32], latent_width=7, batch_size=512, learning_rate=.001,
        updates_per_member=10000, checkpoint_every=1000, full_train_every=1000,
        seeds=[11,22,33,44,55,66,77,88,99,110])
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'Fixed P-only baseline setting differs: {key}')
    root = Path(cfg['output']).resolve()
    namespace = Path('results/p_only_rebuild').resolve()
    if not root.is_relative_to(namespace) or root == namespace:
        raise ValueError('Use a new run below results/p_only_rebuild')
    if os.environ.get('TF_ENABLE_ONEDNN_OPTS') != '0':
        raise ValueError('Set TF_ENABLE_ONEDNN_OPTS=0 before starting Python')


def prepare(cfg, resume=False):
    validate_config(cfg)
    # The legacy audit checks common data/ANN conditions. Its input-partition
    # guard describes its own T/P architecture, not the actual P-only model.
    audit_cfg = {**cfg, 'input_partition': 'current_Y9__current_T_logP'}
    _, pairs, fingerprint = load_inputs(audit_cfg)
    snapshot = dict(config=cfg, fingerprint=fingerprint)
    root = Path(cfg['output'])
    if root.exists():
        if not resume:
            raise FileExistsError('Run exists; preserve it and use --resume or a new output')
        if json.loads((root / 'run.json').read_text()) != snapshot:
            raise ValueError('Run data/config/source provenance differs')
    else:
        root.mkdir(parents=True, exist_ok=False)
        p = PressureDeepONet(dict(kind='boxcox', gamma=.1), pairs['x'], pairs['y'], cfg, cfg['seeds'][0])
        assert p.model.count_params() == 9719
        base = json.loads(Path(fingerprint['ann_members'][0]['path']+'.json').read_text())
        for key, value in [('output_mean',p.norm.mean),('output_scale',p.norm.scale),
                           ('input_mean',p.input_norm.mean),('input_scale',p.input_norm.scale)]:
            np.testing.assert_array_equal(value, base[key])
        p.save(root / 'initial_untrained_seed11')
        write_json(root / 'run.json', snapshot)
        write_json(root / 'prepared.json', dict(status='prepared_untrained', parameters=9719,
            updates=0, planned_member_updates=100000, seeds=cfg['seeds'],
            current_model_sha256=sha256(root/'initial_untrained_seed11.keras'),
            preprocessing_sha256=sha256(root/'initial_untrained_seed11.json'),
            test_used_for_selection=False, git_head=subprocess.check_output(
                ['git','rev-parse','HEAD'],text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],text=True),
            python=platform.python_version(), versions={n:version(n) for n in
                ('tensorflow_cpu','keras','numpy','cantera')},
            threads=dict(intra=tf.config.threading.get_intra_op_parallelism_threads(),
                         inter=tf.config.threading.get_inter_op_parallelism_threads()),
            note='Finite-state DeepONet-style transition model; untrained initialization only'))
    return root, pairs


def train(root, pairs, cfg, budget_seconds):
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError('Positive finite explicitly approved training budget required')
    budget_path = root / 'training_budget.json'
    if budget_path.exists():
        budget = json.loads(budget_path.read_text())
        if budget['seconds'] != budget_seconds:
            raise ValueError('Resume cannot change the original training budget')
    else:
        budget = dict(seconds=budget_seconds, started_unix=time.time())
        budget['expires_unix'] = budget['started_unix'] + budget_seconds
        write_json(budget_path, budget)
    # A fixed absolute expiry conservatively counts interruption/idle time too.
    # Resuming never starts a fresh budget window.
    deadline = time.perf_counter() + max(0., budget['expires_unix']-time.time())
    invocation = time.perf_counter()
    reason = 'time_limit'
    try:
        for seed in cfg['seeds']:
            record_path = root / 'members' / f'seed{seed}.json'
            if record_path.exists():
                record = json.loads(record_path.read_text())
                checkpoint = Path(record['checkpoint'])
                if (record['seed'] != seed or record['updates'] != cfg['updates_per_member']
                        or record['parameters'] != 9719 or record['path'] != str(checkpoint/'model')
                        or record['sha256'] != sha256(checkpoint/'model.keras')
                        or record['preprocessing_sha256'] != sha256(checkpoint/'model.json')
                        or record['train_data_sha256'] != sha256(cfg['training_pairs'])):
                    raise ValueError('Invalid completed member record')
                verified = PressureTrainer.restore(checkpoint, pairs['x'], pairs['y'], cfg, seed)
                if verified.update != cfg['updates_per_member']:
                    raise ValueError('Completed member checkpoint has wrong update')
                del verified
                gc.collect()
                continue
            if time.perf_counter() >= deadline:
                break
            member_root = root / 'checkpoints' / f'seed{seed}'
            checkpoints = sorted(p.parent for p in member_root.glob('*/complete.json'))
            trainer = (PressureTrainer.restore(checkpoints[-1], pairs['x'], pairs['y'], cfg, seed)
                       if checkpoints else PressureTrainer(pairs['x'],pairs['y'],cfg,seed))
            if not trainer.full_train_history and time.perf_counter() < deadline:
                trainer.full_train_loss(deadline)
            while trainer.update < cfg['updates_per_member']:
                target = min(cfg['updates_per_member'],
                    (trainer.update//cfg['checkpoint_every']+1)*cfg['checkpoint_every'])
                stop = trainer.advance(target, deadline)
                if trainer.update % cfg['full_train_every'] == 0 and stop != 'time_limit':
                    trainer.full_train_loss(deadline)
                checkpoint = member_root / f'update_{trainer.update:06d}_{time.time_ns()}'
                trainer.save_checkpoint(checkpoint, stop)
                print('P_ONLY_PROGRESS',seed,trainer.update,stop,flush=True)
                if stop == 'time_limit':
                    break
            if trainer.update == cfg['updates_per_member']:
                # A checkpoint can already be complete when recovering a crash
                # between saving it and registering the finished member.
                checkpoints = sorted(p.parent for p in member_root.glob('*/complete.json'))
                checkpoint = checkpoints[-1]
                record_path.parent.mkdir(parents=True,exist_ok=True)
                write_json(record_path, dict(seed=seed, updates=trainer.update, parameters=9719,
                    checkpoint=str(checkpoint), path=str(checkpoint/'model'),
                    sha256=sha256(checkpoint/'model.keras'),
                    preprocessing_sha256=sha256(checkpoint/'model.json'),
                    train_data_sha256=sha256(cfg['training_pairs']), reason='fixed_10000_updates'))
            del trainer
            gc.collect()
            if time.perf_counter() >= deadline:
                break
        else:
            reason = 'all_members_completed'
    except Exception:
        reason = 'error'
        raise
    finally:
        with (root / 'execution.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(phase='train', reason=reason,
                elapsed_seconds=time.perf_counter()-invocation, budget_seconds=budget_seconds,
                budget_expires_unix=budget['expires_unix']))+'\n')
    print('P_ONLY_TRAIN_STOP',reason,flush=True)
    return reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/p_only_rebuild.json')
    parser.add_argument('--phase',choices=('prepare','train'),default='prepare')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--budget-seconds',type=float)
    args = parser.parse_args()
    if args.phase == 'train' and (args.budget_seconds is None
            or not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0):
        parser.error('Training needs a positive finite --budget-seconds agreed with the user')
    cfg = json.loads(Path(args.config).read_text())
    root,pairs = prepare(cfg,args.resume)
    if args.phase == 'train':
        train(root,pairs,cfg,args.budget_seconds)
    else:
        print('P_ONLY_PREPARED_UNTRAINED',root,flush=True)


if __name__ == '__main__':
    main()
