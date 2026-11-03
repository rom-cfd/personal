"""Matched-10k then exact continuation, all members sequential, six-hour cap."""
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
from .ann_deeponet_study import load_inputs
from .p_only_training import sha256, write_json
from .tp_phi_training import PhiTrainer, PhiSerialSteps, new_monitor, check_plateau
from .study import tf


def event(root, **value):
    value['utc_unix'] = time.time()
    with (root/'execution.jsonl').open('a') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n')
    print(json.dumps(value, allow_nan=False), flush=True)


def prepare(cfg, resume=False):
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip() != 'bw':
        raise ValueError('Use the existing bw checkout')
    if os.environ.get('TF_ENABLE_ONEDNN_OPTS') != '0':
        raise ValueError('Set oneDNN=0 before importing TensorFlow')
    backup = cfg['pre_experiment_backup_commit']
    if subprocess.check_output(['git','rev-parse','origin/bw'],text=True).strip() != backup:
        raise ValueError('Verify the pre-experiment backup on origin/bw first')
    if (cfg['total_budget_seconds'] != 21600 or cfg['evaluation_reserve_seconds'] != 2400
            or cfg['test_used_for_selection'] or cfg['matched_updates'] != 10000):
        raise ValueError('Expected the authorized matched10k / six-hour experiment')
    expected = dict(minimum_updates=50000, check_every=5000, window_updates=20000,
        observe_every=1000, relative_thresholds=[.01,.01,.03], sustain_updates=10000)
    if cfg['plateau'] != expected:
        raise ValueError('Preserve the previously declared train-only plateau policy')
    root = Path(cfg['output'])
    namespace = Path('results/tp_phi_study').resolve()
    if root.resolve() == namespace or not root.resolve().is_relative_to(namespace):
        raise ValueError('New T/P/phi result namespace required')
    base = json.loads(Path(cfg['base_config']).read_text())
    data, pairs, fingerprint = load_inputs(base)
    # Initial metadata is a known IC descriptor. No future T/Y or labels enter phi.
    phis = data['initials'][:,2]
    if not np.isfinite(phis).all() or np.any(phis <= 0):
        raise ValueError('Invalid initial equivalence ratios')
    model_cfg = {**base, **cfg['architecture']}
    augmented = dict(pairs)
    augmented['x'] = np.c_[pairs['x'][:,:10], phis[pairs['parent'][:,0]], pairs['x'][:,-1]]
    snapshot = dict(config=cfg, base_config=base, model_config=model_cfg, fingerprint=fingerprint,
        phi_source='initials[parent_trajectory,2], constant per trajectory',
        physical_input='T,Y9,phi_initial,P', target='next T,Y9, fixed dt=2us',
        pre_experiment_backup_commit=backup)
    if root.exists():
        if not resume: raise FileExistsError('Use a new output or identical --resume')
        if json.loads((root/'run.json').read_text()) != snapshot:
            raise ValueError('Resume source/config/data changed')
    else:
        previous = {str(p):sha256(p) for p in Path('results').rglob('*') if p.is_file()}
        root.mkdir(parents=True, exist_ok=False)
        write_json(root/'run.json', snapshot)
        write_json(root/'original_results_sha256.json', previous)
        write_json(root/'environment.json', dict(python=platform.python_version(),
            versions={n:version(n) for n in ('tensorflow_cpu','keras','numpy','scipy','cantera','matplotlib')},
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],text=True),
            tensorflow_threads=dict(intra=tf.config.threading.get_intra_op_parallelism_threads(),
                inter=tf.config.threading.get_inter_op_parallelism_threads()),
            cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
            thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS','TF_ENABLE_ONEDNN_OPTS','CUDA_VISIBLE_DEVICES')},
            authorization='User requested T/P/phi trunk, matched10k and plateau continuation, max six hours'))
    return root, data, augmented, snapshot


def deadlines(root, cfg):
    path = root/'budget.json'
    if path.exists():
        budget = json.loads(path.read_text())
        if budget['seconds'] != cfg['total_budget_seconds']:
            raise ValueError('Resume cannot reset the clock')
    else:
        budget = dict(seconds=cfg['total_budget_seconds'], started_unix=time.time())
        budget['expires_unix'] = budget['started_unix'] + budget['seconds']
        budget['training_expires_unix'] = budget['expires_unix'] - cfg['evaluation_reserve_seconds']
        write_json(path, budget)
    now, clock = time.time(), time.perf_counter()
    return (clock+max(0.,budget['training_expires_unix']-now),
            clock+max(0.,budget['expires_unix']-now))


def latest(root, stage, seed):
    paths = sorted((root/'members'/stage/f'seed{seed}').glob('finish_*.json'))
    return json.loads(paths[-1].read_text()) if paths else None


def train_member(root, cfg, snapshot, pairs, seed, stage, deadline):
    started = time.perf_counter()
    model_cfg = snapshot['model_config']
    prior = latest(root,stage,seed)
    if prior:
        trainer = PhiTrainer.restore(prior['checkpoint'], pairs['x'], pairs['y'], model_cfg, seed)
    elif stage == 'long':
        origin = latest(root,'matched',seed)
        if not origin or origin['updates'] != cfg['matched_updates']:
            raise ValueError('Preserve all matched10k checkpoints before continuation')
        trainer = PhiTrainer.restore(origin['checkpoint'],pairs['x'],pairs['y'],model_cfg,seed)
        trainer.plateau_monitor = new_monitor(cfg['plateau'], trainer.update, trainer.elapsed_seconds)
    else:
        trainer = PhiTrainer(pairs['x'],pairs['y'],model_cfg,seed)
        trainer.plateau_monitor = new_monitor(cfg['plateau'])
    origin_update = trainer.update
    engine = PhiSerialSteps(trainer)
    if not trainer.full_train_history: engine.full_train_loss(deadline)
    reason = 'time_limit'
    # No extra update ceiling: only the fixed10k endpoint, plateau, or wall cap.
    while time.perf_counter() < deadline:
        if stage == 'matched' and trainer.update >= cfg['matched_updates']:
            reason = 'matched_update_budget'; break
        target = (trainer.update//1000+1)*1000
        if stage == 'matched': target = min(target,cfg['matched_updates'])
        if engine.advance(target,deadline) == 'time_limit': break
        loss = engine.full_train_loss(deadline)
        plateau = (stage == 'long' and loss is not None and
            check_plateau(trainer.full_train_history,trainer.update,trainer.plateau_monitor))
        if trainer.update % cfg['checkpoint_every'] == 0 or plateau:
            before = time.perf_counter()
            cp = root/'checkpoints'/f'seed{seed}'/f'update_{trainer.update:07d}_{time.time_ns()}'
            trainer.save_checkpoint(cp,'plateau' if plateau else 'observed_checkpoint')
            engine.timing['checkpoint_seconds'] += time.perf_counter()-before
            event(root,phase='train_progress',stage=stage,seed=seed,update=trainer.update,
                full_train_loss=loss,plateau=bool(plateau),member_active_seconds=time.perf_counter()-started)
        if plateau: reason = 'plateau'; break
        if stage == 'matched' and trainer.update == cfg['matched_updates']:
            reason = 'matched_update_budget'; break
    cp = root/'checkpoints'/f'seed{seed}'/f'update_{trainer.update:07d}_{time.time_ns()}'
    trainer.save_checkpoint(cp,reason)
    history = trainer.full_train_history
    record = dict(family='tp_phi',stage=stage,seed=seed,updates=trainer.update,reason=reason,
        checkpoint=str(cp),path=str(cp/'model'),sha256=sha256(cp/'model.keras'),
        preprocessing_sha256=sha256(cp/'model.json'),parameters=trainer.predictor.model.count_params(),
        optimizer_iteration=int(trainer.optimizer.iterations.numpy()),
        last_full_train_loss=history[-1]['full_train_loss'] if history else None,
        last_full_train_loss_update=history[-1]['update'] if history else None,
        invocation_origin_update=origin_update,additional_updates=trainer.update-origin_update,
        timing=engine.timing,active_seconds=time.perf_counter()-started,
        carried_elapsed_seconds=trainer.elapsed_seconds,
        plateau_checks=trainer.plateau_monitor['checks'],test_used_for_selection=False)
    parent = root/'members'/stage/f'seed{seed}';parent.mkdir(parents=True,exist_ok=True)
    write_json(parent/f'finish_{trainer.update:07d}_{time.time_ns()}.json',record)
    event(root,phase='member_finished',stage=stage,seed=seed,update=trainer.update,
        reason=reason,active_seconds=record['active_seconds'],additional_updates=record['additional_updates'])
    del trainer, engine
    gc.collect()
    return record


def train(root,cfg,snapshot,pairs,deadline):
    seeds = snapshot['base_config']['seeds']
    if not (root/'matched_members.json').exists():
        for seed in seeds:
            record = latest(root,'matched',seed)
            if record and record['updates'] == cfg['matched_updates']: continue
            if time.perf_counter() >= deadline: raise TimeoutError('Matched training incomplete')
            train_member(root,cfg,snapshot,pairs,seed,'matched',deadline)
        records = [latest(root,'matched',s) for s in seeds]
        if any(r['updates'] != cfg['matched_updates'] for r in records):
            raise TimeoutError('Matched training incomplete')
        write_json(root/'matched_members.json',records)
        event(root,phase='matched_training_finished',members=len(records),updates_each=10000)
    if (root/'final_members.json').exists(): return
    for index,seed in enumerate(seeds):
        record = latest(root,'long',seed)
        if record and record['reason'] == 'plateau': continue
        remaining = deadline-time.perf_counter()-10
        if remaining <= 0: break
        train_member(root,cfg,snapshot,pairs,seed,'long',time.perf_counter()+remaining/(len(seeds)-index))
    while deadline-time.perf_counter() > 30:
        pending = [s for s in seeds if latest(root,'long',s) and latest(root,'long',s)['reason']=='time_limit']
        if not pending: break
        for index,seed in enumerate(pending):
            remaining = deadline-time.perf_counter()-10
            if remaining <= 0: break
            train_member(root,cfg,snapshot,pairs,seed,'long',time.perf_counter()+remaining/(len(pending)-index))
    records = [latest(root,'long',s) for s in seeds]
    if any(r is None for r in records): raise TimeoutError('Not all long members ran')
    write_json(root/'final_members.json',records)
    event(root,phase='training_finished',members=len(records),
        plateau_members=sum(r['reason']=='plateau' for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/tp_phi_study.json')
    parser.add_argument('--phase',choices=('prepare','run'),default='prepare')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root,data,pairs,snapshot = prepare(cfg,args.resume)
    if args.phase == 'prepare': print('PHI_PREPARED',root,flush=True); return
    train_deadline,total_deadline = deadlines(root,cfg)
    before = time.perf_counter()
    train(root,cfg,snapshot,pairs,train_deadline)
    event(root,phase='training_wall_finished',seconds=time.perf_counter()-before)
    from .tp_phi_evaluation import evaluate
    before = time.perf_counter()
    try: evaluate(root,cfg,snapshot,pairs,data,total_deadline)
    except TimeoutError: event(root,phase='evaluation_time_limit')
    event(root,phase='evaluation_wall_finished',seconds=time.perf_counter()-before)
    from .tp_phi_report import report
    report(root,cfg,snapshot,data,total_deadline)
    event(root,phase='pipeline_finished')


if __name__ == '__main__':
    main()
