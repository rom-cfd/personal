"""Six-hour serial continuation of lambda1/100/1000,120 existing10k models."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from .ann_deeponet_study import load_inputs
from .element_long_training import joint_monitor, joint_check
from .penalty_sweep_study import weight_root
from .p_only_training import sha256, write_json
from .soft_constraint_study import inputs_for, event
from .soft_constraint_training import SoftTrainer, FAMILIES, soft_signature
from .study import tf

POLICY = dict(minimum_updates=50000, check_every=5000, window_updates=20000,
              observe_every=1000, relative_thresholds=[.01, .01, .03], sustain_updates=10000)


def prepare(cfg, resume=False):
    if subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != 'bw':
        raise ValueError('Use the existing bw checkout')
    if os.environ.get('TF_ENABLE_ONEDNN_OPTS') != '0':
        raise ValueError('Set oneDNN0 before TensorFlow import')
    if (cfg['lambdas'] != [1., 100., 1000.] or cfg['total_budget_seconds'] != 21600
            or cfg['evaluation_reserve_seconds'] != 3600 or cfg['plateau'] != POLICY
            or cfg['components_required'] != ['total', 'data', 'element']
            or cfg['test_used_for_selection'] or cfg['modes'] != ['raw', 'repaired']
            or cfg['evaluation_splits'] != ['test']):
        raise ValueError('Keep the user-authorized six-hour120-member joint-plateau plan')
    subprocess.run(['git', 'merge-base', '--is-ancestor', cfg['required_pre_experiment_commit'], 'HEAD'], check=True)
    origin = Path(cfg['origin']); old = json.loads((origin / 'run.json').read_text())
    data, pairs, fingerprint = load_inputs(old['base_config'])
    for key in ('pair_sha256', 'dense_sha256', 'ann_members', 'reference_ignition'):
        if fingerprint[key] != old['fingerprint'][key]:
            raise ValueError(f'Origin data/control changed: {key}')
    records = json.loads((origin / 'final_members.json').read_text())
    keys = {(r['mu'], r['family'], r['seed']) for r in records}
    expected = {(w, f, s) for w in cfg['lambdas'] for f in FAMILIES for s in old['base_config']['seeds']}
    if len(records) != 120 or keys != expected:
        raise ValueError('Need all120 distinct fixed10k origins')
    signatures = {(w, f): soft_signature(inputs_for(f, pairs, data), pairs['y'], old['model_configs'][f],
                  f, old['element_matrix'], w) for w in cfg['lambdas'] for f in FAMILIES}
    for r in records:
        cp = Path(r['checkpoint']); marker = json.loads((cp / 'complete.json').read_text())
        for name, expected_hash in marker['output_sha256'].items():
            if sha256(cp / name) != expected_hash:
                raise ValueError('Origin checkpoint changed')
        state = json.loads((cp / 'state.json').read_text())
        if (r['updates'] != 10000 or state['update'] != 10000 or state['optimizer_iteration'] != 10000
                or state['signature'] != signatures[r['mu'], r['family']]):
            raise ValueError('Origin update/Adam/data/runtime signature differs')
    root = Path(cfg['output']); namespace = Path('results/weight_long_study').resolve()
    if root.resolve() == namespace or not root.resolve().is_relative_to(namespace):
        raise ValueError('Use a new output below results/weight_long_study')
    origin_files = [origin / n for n in ('run.json', 'final_members.json', 'FINAL_ARTIFACTS.json',
                   'postrun_verification.json', 'all_artifacts_sha256.json')]
    sources = sorted(Path('kinetics').glob('weight_long_*.py')) + sorted(Path('scripts').glob('*weight_long*.py'))
    sources += [Path('kinetics/element_long_training.py'), Path('kinetics/deeponet_long_training.py'),
                Path('kinetics/penalty_sweep_evaluation.py'), Path('scripts/report_kinetics_nrmse.py'),
                Path('scripts/nrmse_util_reference.py'), Path('references/user_nrmse/util.py')]
    snapshot = dict(config=cfg, origin_snapshot=old, origin_members=records,
        base_config=old['base_config'], model_configs=old['model_configs'], element_matrix=old['element_matrix'],
        baseline_members=old['baseline_members'], fingerprint=fingerprint,
        origin_sha256={str(p): sha256(p) for p in origin_files},
        source_sha256={str(p): sha256(p) for p in sources},
        authorization='User requested commit/push then120-model long continuation; maximum6h including evaluation/plots',
        state_policy='Preserve weights, Adam, absolute LR, RNG streams, preprocessing, lambda and histories')
    if root.exists():
        if not resume:
            raise FileExistsError('Use a new output or identical --resume')
        if json.loads((root / 'run.json').read_text()) != snapshot:
            raise ValueError('Long resume provenance changed')
    else:
        previous = {str(p): sha256(p) for p in Path('results').rglob('*') if p.is_file()}
        root.mkdir(parents=True, exist_ok=False); write_json(root / 'run.json', snapshot)
        write_json(root / 'original_results_sha256.json', previous)
        write_json(root / 'environment.json', dict(python=platform.python_version(),
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            git_status=subprocess.check_output(['git', 'status', '--short'], text=True),
            threads=dict(intra=tf.config.threading.get_intra_op_parallelism_threads(),
                         inter=tf.config.threading.get_inter_op_parallelism_threads()),
            thread_environment={k: os.environ.get(k) for k in ('TF_ENABLE_ONEDNN_OPTS', 'OMP_NUM_THREADS',
                                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')}))
        for w in cfg['lambdas']:
            child = weight_root(root, w); child.mkdir()
            write_json(child / 'run.json', dict(long_run_sha256=sha256(root / 'run.json'), lambda_value=w))
    return root, data, pairs, snapshot


def deadlines(root, cfg):
    path = Path(root) / 'budget.json'
    if path.exists():
        budget = json.loads(path.read_text())
    else:
        start = time.time()
        budget = dict(started_unix=start, seconds=21600, expires_unix=start + 21600,
                      training_expires_unix=start + 21600 - cfg['evaluation_reserve_seconds'])
        write_json(path, budget)
    if budget['seconds'] != 21600:
        raise ValueError('Resume cannot renew the approved clock')
    now, clock = time.time(), time.perf_counter()
    return clock + max(0, budget['training_expires_unix'] - now), clock + max(0, budget['expires_unix'] - now)


def latest(root, weight, family, seed):
    paths = sorted((weight_root(root, weight) / 'members' / family / f'seed{seed}').glob('finish_*.json'))
    return json.loads(paths[-1].read_text()) if paths else None


def start_checkpoint(root, weight, family, seed, origin):
    # A crash may precede the terminal record: keep the most recent complete
    # observed checkpoint, not just the most recent finished invocation.
    parent = weight_root(root, weight) / 'checkpoints' / family / f'seed{seed}'
    complete = sorted(parent.glob('*/complete.json'))
    return complete[-1].parent if complete else Path(origin['checkpoint'])


def train_member(root, cfg, snapshot, pairs, data, weight, family, seed, deadline, maximum_update=None):
    started = time.perf_counter(); child = weight_root(root, weight)
    x = inputs_for(family, pairs, data); y = pairs['y']
    origin = next(r for r in snapshot['origin_members'] if (r['mu'], r['family'], r['seed']) == (weight, family, seed))
    cp = start_checkpoint(root, weight, family, seed, origin)
    trainer = SoftTrainer.restore(cp, x, y, snapshot['model_configs'][family], seed, family,
                                  np.array(snapshot['element_matrix']), weight)
    run_hash = sha256(root / 'run.json')
    if trainer.plateau_monitor.get('enabled'):
        if trainer.plateau_monitor.get('long_run_sha256') != run_hash:
            raise ValueError('Joint-monitor lineage changed')
    else:
        if cp != Path(origin['checkpoint']):
            raise ValueError('A continued checkpoint lost its monitor')
        trainer.plateau_monitor = joint_monitor(cfg['plateau'], run_hash, trainer.update, trainer.elapsed_seconds)
    resume_start = trainer.update; reason = 'time_limit'
    directory = child / 'checkpoints' / family / f'seed{seed}'
    event(root, phase='member_started', lambda_value=weight, family=family, seed=seed,
          update=trainer.update, restored_from=str(cp))
    if trainer.plateau_monitor['plateau']:
        reason = 'joint_plateau'
    else:
        while time.perf_counter() < deadline:
            if maximum_update is not None and trainer.update >= maximum_update:
                reason = 'test_update_limit'; break
            target = (trainer.update // cfg['observe_every'] + 1) * cfg['observe_every']
            if maximum_update is not None:
                target = min(target, maximum_update)
            if trainer.advance(target, deadline) == 'time_limit':
                break
            loss = trainer.full_train_loss(deadline)
            plateau = loss is not None and joint_check(trainer.full_train_history, trainer.update, trainer.plateau_monitor)
            if trainer.update % cfg['checkpoint_every'] == 0 or plateau:
                trainer.save_checkpoint(directory / f'update_{trainer.update:07d}_{time.time_ns()}',
                                        'joint_plateau' if plateau else 'observed_checkpoint')
                event(root, phase='train_progress', lambda_value=weight, family=family, seed=seed,
                      update=trainer.update, loss=loss, joint_plateau=bool(plateau),
                      active_seconds=time.perf_counter() - started)
            if plateau:
                reason = 'joint_plateau'; break
    cp = trainer.save_checkpoint(directory / f'update_{trainer.update:07d}_{time.time_ns()}', reason)
    record = dict(mu=weight, family=family, seed=seed, updates=trainer.update, reason=reason,
        checkpoint=str(cp), path=str(cp / 'model'), sha256=sha256(cp / 'model.keras'),
        preprocessing_sha256=sha256(cp / 'model.json'), parameters=trainer.predictor.model.count_params(),
        optimizer_iteration=int(trainer.optimizer.iterations.numpy()), resume_start_update=resume_start,
        added_updates=trainer.update - resume_start, total_added_updates=trainer.update - origin['updates'],
        original_checkpoint=origin['checkpoint'], active_seconds=time.perf_counter() - started,
        carried_seconds=trainer.elapsed_seconds, timing=trainer.timing,
        full_train=trainer.full_train_history[-1], monitor=trainer.plateau_monitor, test_used_for_selection=False)
    parent = child / 'members' / family / f'seed{seed}'; parent.mkdir(parents=True, exist_ok=True)
    write_json(parent / f'finish_{trainer.update:07d}_{time.time_ns()}.json', record)
    event(root, phase='member_finished', lambda_value=weight, family=family, seed=seed,
          update=trainer.update, reason=reason, added_updates=record['added_updates'],
          active_seconds=record['active_seconds'])
    del trainer; gc.collect(); return record


def train(root, cfg, snapshot, pairs, data, deadline):
    if (root / 'final_members.json').exists():
        return
    tasks = [(w, f, s) for w in cfg['lambdas'] for f in FAMILIES for s in snapshot['base_config']['seeds']]
    # Finished first-pass members retain their allocation after a resume.
    pending = [t for t in tasks if latest(root, *t) is None]
    for i, task in enumerate(pending):
        remaining = deadline - time.perf_counter() - 10
        if remaining <= 0:
            break
        train_member(root, cfg, snapshot, pairs, data, *task,
                     time.perf_counter() + remaining / (len(pending) - i))
    if any(latest(root, *t) is None for t in tasks):
        raise TimeoutError('Not all120 models received continuation time; preserve the fixed clock and partial checkpoints')
    while deadline - time.perf_counter() > 30:
        capped = [t for t in tasks if latest(root, *t)['reason'] == 'time_limit']
        if not capped:
            break
        for i, task in enumerate(capped):
            remaining = deadline - time.perf_counter() - 10
            if remaining <= 0:
                break
            train_member(root, cfg, snapshot, pairs, data, *task,
                         time.perf_counter() + remaining / (len(capped) - i))
    records = [latest(root, *t) for t in tasks]
    for w in cfg['lambdas']:
        write_json(weight_root(root, w) / 'final_members.json', [r for r in records if r['mu'] == w])
    write_json(root / 'final_members.json', records)
    event(root, phase='training_finished', members=120,
          joint_plateau_members=sum(r['reason'] == 'joint_plateau' for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/weight_long_study.json')
    parser.add_argument('--phase', choices=('prepare', 'run'), default='prepare')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); cfg = json.loads(Path(args.config).read_text())
    root, data, pairs, snapshot = prepare(cfg, args.resume)
    if args.phase == 'prepare':
        print('WEIGHT_LONG_PREPARED', root, flush=True); return
    train_deadline, total_deadline = deadlines(root, cfg)
    before = time.perf_counter(); train(root, cfg, snapshot, pairs, data, train_deadline)
    event(root, phase='training_wall_finished', seconds=time.perf_counter() - before)
    from .weight_long_evaluation import evaluate
    before = time.perf_counter(); evaluate(root, cfg, snapshot, data, pairs, total_deadline - 1500)
    event(root, phase='evaluation_wall_finished', seconds=time.perf_counter() - before)
    from scripts.report_weight_long import report
    report(root, cfg, snapshot, data, total_deadline); event(root, phase='pipeline_finished')


if __name__ == '__main__':
    main()
