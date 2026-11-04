"""Pure-MSE Box-Cox sweep: up to10k updates and150 seconds per new model."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from importlib.metadata import version

import numpy as np
from .ann_deeponet_study import load_inputs, check_deadline
from .gamma_sweep_training import GammaTrainer, gamma_tag, weights_hash
from .soft_constraint_training import FAMILIES, LOADERS
from .soft_constraint_study import model_config, inputs_for, event
from .p_only_training import sha256, write_json
from .mixed_ensemble import require_shared_preprocessing
from .study import tf


def validate_config(cfg):
    if (cfg['gammas'] != [.05, .1, .2, .3, .5, 1.] or cfg['reference_gamma'] != .1
            or cfg['families'] != list(FAMILIES) or cfg['updates_per_member'] != 10000
            or cfg['loss'] != 'transformed_residual_MSE_only' or cfg['test_used_for_selection']
            or cfg['evaluation_splits'] != ['val', 'test'] or cfg['modes'] != ['raw', 'repaired']
            or cfg['selection'] != 'validation_repaired_completion_then_temperature_NRMSE_on_common_ICs'):
        raise ValueError('Preserve the fixed six-gamma pure-MSE matched10k design')
    if not 0 < cfg['training_budget_seconds'] < cfg['total_budget_seconds'] - cfg['report_reserve_seconds']:
        raise ValueError('Reserve evaluation and reporting time')
    if cfg['member_budget_seconds'] != 150 or cfg['training_budget_seconds'] < 200 * cfg['member_budget_seconds']:
        raise ValueError('Keep the previous150-second allowance for each of200 new members')
    if cfg['evaluation_budget_seconds'] + cfg['training_budget_seconds'] + cfg['report_reserve_seconds'] != cfg['total_budget_seconds']:
        raise ValueError('Phase guards must add up to the derived outer watchdog limit')


def prepare(cfg, resume=False):
    validate_config(cfg)
    if subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != 'bw':
        raise ValueError('Use the existing bw checkout')
    if os.environ.get('TF_ENABLE_ONEDNN_OPTS') != '0':
        raise ValueError('Set oneDNN0 before TensorFlow import')
    root = Path(cfg['output']); namespace = Path('results/gamma_sweep').resolve()
    if root.resolve() == namespace or not root.resolve().is_relative_to(namespace):
        raise ValueError('New output below results/gamma_sweep required')
    base = json.loads(Path(cfg['base_config']).read_text())
    data, pairs, fingerprint = load_inputs(base)
    origin = Path(cfg['baseline_origin']); baseline = json.loads((origin / 'run.json').read_text())
    if baseline['base_config'] != base or any(baseline['fingerprint'][k] != fingerprint[k]
            for k in ('pair_sha256', 'dense_sha256', 'ann_members', 'reference_ignition')):
        raise ValueError('Baseline data or architecture configuration changed')
    for family, records in baseline['baseline_members'].items():
        if [r['seed'] for r in records] != base['seeds']:
            raise ValueError('Baseline member order changed')
        for r in records:
            if r['updates'] != 10000 or sha256(r['path'] + '.keras') != r['sha256']:
                raise ValueError('Baseline model changed')
            info = json.loads(Path(r['path'] + '.json').read_text())
            if info['spec']['kind'] != 'boxcox' or info['spec']['gamma'] != .1:
                raise ValueError('Require unconstrained gamma=.1 models')
    cached = json.loads((origin / 'baseline_traces.json').read_text())
    for info in cached.values():
        if sha256(info['path']) != info['sha256']:
            raise ValueError('Frozen control trace changed')
    source = sorted(Path('kinetics').glob('*.py')) + sorted(Path('scripts').glob('*.py'))
    source += [Path('references/user_nrmse/util.py'), Path(cfg['base_config'])]
    snapshot = dict(config=cfg, base_config=base, fingerprint=fingerprint,
        model_configs={f: model_config(base, f) for f in FAMILIES},
        baseline_members=baseline['baseline_members'], baseline_test_traces=cached,
        source_sha256={str(p): sha256(p) for p in source},
        origin_sha256={str(p): sha256(p) for p in (origin / 'run.json', origin / 'baseline_traces.json')},
        new_members=200, cached_members=40, authorized_updates=2000000,
        training='Fresh pure MSE; stop at10000 updates or150 seconds per model; same seeds/minibatch streams across new gammas',
        preprocessing='Only gamma changes; train-only state/input standardizers refit for each gamma',
        control_timing='Existing gamma=.1 models and test traces reused; historical batched ANN time is not a fresh serial speed baseline',
        selection='Freeze validation-only completion/temperature choice before any new test evaluation',
        test_note='Previously inspected test ICs; exploratory sensitivity comparison, not a new independent holdout')
    if root.exists():
        if not resume:
            raise FileExistsError('Use a new output or an identical resume')
        if json.loads((root / 'run.json').read_text()) != snapshot:
            raise ValueError('Gamma sweep resume provenance changed')
    else:
        previous = {str(p): sha256(p) for p in Path('results').rglob('*') if p.is_file()}
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / 'run.json', snapshot)
        write_json(root / 'original_results_sha256.json', previous)
        write_json(root / 'environment.json', dict(python=platform.python_version(),
            versions={n: version(n) for n in ('tensorflow_cpu', 'keras', 'numpy', 'cantera', 'scipy', 'matplotlib')},
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            git_status=subprocess.check_output(['git', 'status', '--short'], text=True),
            tensorflow_threads=dict(intra=tf.config.threading.get_intra_op_parallelism_threads(),
                                    inter=tf.config.threading.get_inter_op_parallelism_threads()),
            thread_environment={k: os.environ.get(k) for k in ('TF_ENABLE_ONEDNN_OPTS', 'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')}))
        for gamma in cfg['gammas']:
            child = root / gamma_tag(gamma); child.mkdir()
            if gamma == .1:
                records = [dict(r, family=f, gamma=.1, reused=True) for f, members in baseline['baseline_members'].items() for r in members]
                write_json(child / 'final_members.json', records)
    return root, data, pairs, snapshot


def deadlines(root, cfg):
    budget = json.loads((root / 'budget.json').read_text())
    if budget['seconds'] != cfg['total_budget_seconds']:
        raise ValueError('Cannot change the original clock')
    now, clock = time.time(), time.perf_counter()
    return (clock + max(0, budget['training_expires_unix'] - now),
            clock + max(0, budget['expires_unix'] - now))


def member_deadline(destination, cfg):
    """Persist the first start; resuming a member never grants another150 seconds."""
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'budget.json'
    if path.exists():
        budget = json.loads(path.read_text())
        if budget['seconds'] != cfg['member_budget_seconds']:
            raise ValueError('Cannot change a member time allowance')
    else:
        now = time.time()
        budget = dict(started_unix=now, seconds=cfg['member_budget_seconds'], expires_unix=now+cfg['member_budget_seconds'])
        write_json(path, budget)
    return time.perf_counter()+max(0, budget['expires_unix']-time.time()), budget


def train(root, cfg, snapshot, pairs, data, deadline):
    if (root / 'final_members.json').exists():
        return
    all_records = []
    for gamma in cfg['gammas']:
        child = root / gamma_tag(gamma)
        if (child / 'final_members.json').exists():
            all_records.extend(json.loads((child / 'final_members.json').read_text())); continue
        records = []
        for family in FAMILIES:
            x, y = inputs_for(family, pairs, data), pairs['y']
            config = {**snapshot['model_configs'][family], 'gamma': gamma}
            for seed in snapshot['base_config']['seeds']:
                check_deadline(deadline)
                destination = child / 'members' / family / f'seed{seed}'
                terminal = sorted(destination.glob('finish_*.json'))
                if terminal:
                    r = json.loads(terminal[-1].read_text())
                    if (r['updates'] <= 10000 and r['reason'] in ('fixed_update_budget', 'member_time_limit')):
                        records.append(r); continue
                started = time.perf_counter()
                stop_at, member_budget = member_deadline(destination, cfg)
                stop_at = min(stop_at, deadline)
                event(root, phase='member_started', gamma=gamma, family=family, seed=seed)
                parent = child / 'checkpoints' / family / f'seed{seed}'
                checkpoints = sorted(parent.glob('*/complete.json'))
                trainer = GammaTrainer.restore(checkpoints[-1].parent, x, y, config, seed, family, gamma) if checkpoints else GammaTrainer(x, y, config, seed, family, gamma)
                destination.mkdir(parents=True, exist_ok=True)
                init = destination / 'initialization.json'
                if not init.exists():
                    if trainer.update:
                        raise ValueError('Missing initialization provenance on resume')
                    write_json(init, dict(initial_weights_sha256=weights_hash(trainer.predictor),
                        initial_numpy_rng=trainer.rng.bit_generator.state, gamma=gamma, family=family, seed=seed))
                if not trainer.full_train_history:
                    trainer.full_train_loss(stop_at)
                while trainer.update < 10000 and time.perf_counter() < stop_at:
                    target = min((trainer.update // cfg['observe_every'] + 1) * cfg['observe_every'], 10000)
                    if trainer.advance(target, stop_at) == 'time_limit':
                        break
                    loss = trainer.full_train_loss(stop_at)
                    if trainer.update % cfg['checkpoint_every'] == 0:
                        trainer.save_checkpoint(parent / f'update_{trainer.update:06d}_{time.time_ns()}', 'observed_checkpoint')
                        event(root, phase='train_progress', gamma=gamma, family=family, seed=seed,
                              update=trainer.update, full_train_MSE=loss)
                reason = 'fixed_update_budget' if trainer.update == 10000 else 'member_time_limit'
                cp = trainer.save_checkpoint(parent / f'update_{trainer.update:06d}_{time.time_ns()}', reason)
                r = dict(gamma=gamma, family=family, seed=seed, updates=trainer.update, reason=reason,
                    checkpoint=str(cp), path=str(cp / 'model'), sha256=sha256(cp / 'model.keras'),
                    preprocessing_sha256=sha256(cp / 'model.json'), parameters=trainer.predictor.model.count_params(),
                    optimizer_iteration=int(trainer.optimizer.iterations.numpy()),
                    active_seconds=time.perf_counter() - started, cumulative_member_seconds=trainer.elapsed_seconds,
                    member_budget=member_budget, partial_endpoint=trainer.update < 10000,
                    timing=trainer.plateau_monitor['timing'], reused=False, soft_constraint=False)
                write_json(destination / f'finish_{trainer.update:06d}_{time.time_ns()}.json', r)
                event(root, phase='member_finished', gamma=gamma, family=family, seed=seed, update=trainer.update,
                      reason=reason, seconds=r['active_seconds'])
                del trainer; gc.collect()
                records.append(r)
        write_json(child / 'final_members.json', records); all_records.extend(records)
        event(root, phase='gamma_training_finished', gamma=gamma, members=40)
    write_json(root / 'final_members.json', all_records)


def load_groups(root, snapshot, gamma):
    records = json.loads((root / gamma_tag(gamma) / 'final_members.json').read_text())
    if len(records) != 40 or any(not 0 <= r['updates'] <= 10000 for r in records):
        raise ValueError('Require forty frozen endpoints with their actual update counts')
    tf.keras.backend.clear_session(); groups = {}
    for family in FAMILIES:
        lookup = {r['seed']: r for r in records if r['family'] == family}; group = []
        for seed in snapshot['base_config']['seeds']:
            r = lookup[seed]
            if sha256(r['path'] + '.keras') != r['sha256']:
                raise ValueError('Endpoint model changed')
            p = LOADERS[family].load(r['path']); p.seed_label = seed
            if p.transform.gamma != gamma:
                raise ValueError('Wrong gamma endpoint')
            group.append(p)
        groups[family] = group
    require_shared_preprocessing([p for group in groups.values() for p in group])
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/gamma_sweep.json')
    parser.add_argument('--phase', choices=('prepare', 'run'), default='prepare')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); cfg = json.loads(Path(args.config).read_text())
    root, data, pairs, snapshot = prepare(cfg, args.resume)
    if args.phase == 'prepare':
        print('GAMMA_SWEEP_PREPARED', root, flush=True); return
    train_deadline, total_deadline = deadlines(root, cfg)
    before = time.perf_counter(); train(root, cfg, snapshot, pairs, data, train_deadline)
    event(root, phase='training_wall_finished', seconds=time.perf_counter() - before)
    from .gamma_sweep_evaluation import evaluate
    evaluation_deadline = min(total_deadline-cfg['report_reserve_seconds'],time.perf_counter()+cfg['evaluation_budget_seconds'])
    before = time.perf_counter(); evaluate(root, cfg, snapshot, data, evaluation_deadline)
    event(root, phase='evaluation_wall_finished', seconds=time.perf_counter() - before)
    from scripts.report_gamma_sweep import report
    report(root, cfg, snapshot, data, min(total_deadline,time.perf_counter()+cfg['report_reserve_seconds']))
    event(root, phase='pipeline_finished')


if __name__ == '__main__':
    main()
