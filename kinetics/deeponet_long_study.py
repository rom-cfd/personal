"""Six-hour sequential P-only / T/P plateau study and fixed comparison pipeline."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from importlib.metadata import version

from .ann_deeponet_study import load_inputs
from .deeponet_long_training import SerialSteps,restore_trainer,fresh_tp,new_monitor,check_plateau
from .p_only_training import sha256,write_json
from .study import tf


def event(root,**value):
    value['utc_unix']=time.time()
    with (root/'execution.jsonl').open('a') as handle:
        handle.write(json.dumps(value,allow_nan=False)+'\n')
    print(json.dumps(value,allow_nan=False),flush=True)


def prepare(cfg,resume=False):
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='bw':
        raise ValueError('Use the existing bw checkout')
    if os.environ.get('TF_ENABLE_ONEDNN_OPTS')!='0':
        raise ValueError('Set TF_ENABLE_ONEDNN_OPTS=0 before importing TensorFlow')
    root=Path(cfg['output']);namespace=Path('results/deeponet_long_study').resolve()
    if not root.resolve().is_relative_to(namespace) or root.resolve()==namespace:
        raise ValueError('New long-study output namespace required')
    if (cfg['total_budget_seconds']!=21600 or cfg['evaluation_reserve_seconds']!=2400
            or cfg['maximum_updates']!=1000000 or cfg['family_order']!=['p_only','tp']
            or cfg['test_used_for_selection']):
        raise ValueError('The agreed sequential six-hour plan differs')
    policy=cfg['plateau']
    if policy!=dict(minimum_updates=50000,check_every=5000,window_updates=20000,
                   observe_every=1000,relative_thresholds=[.01,.01,.03],sustain_updates=10000):
        raise ValueError('The declared train-only plateau policy differs')
    base=json.loads(Path(cfg['base_config']).read_text())
    data,pairs,fingerprint=load_inputs(base)
    origins={}
    for seed in base['seeds']:
        record_path=Path(cfg['p_only_root'])/'members'/f'seed{seed}.json'
        record=json.loads(record_path.read_text())
        if (record['seed']!=seed or record['updates']!=10000
                or sha256(record['path']+'.keras')!=record['sha256']
                or sha256(record['path']+'.json')!=record['preprocessing_sha256']
                or record['train_data_sha256']!=fingerprint['pair_sha256']):
            raise ValueError('P-only origin hash/provenance mismatch')
        origins[str(seed)]=record
    original_tp=json.loads(Path(base['output'],'deeponet_members.json').read_text())
    for r in original_tp:
        if sha256(r['path']+'.keras')!=r['sha256'] or sha256(r['path']+'.json')!=r['preprocessing_sha256']:
            raise ValueError('Original T/P hash mismatch')
    snapshot=dict(config=cfg,base_config=base,fingerprint=fingerprint,p_only_origins=origins,
        original_tp=original_tp,p_only_config=json.loads(Path(cfg['p_only_root'],'run.json').read_text())['config'])
    if root.exists():
        if not resume:
            raise FileExistsError('Output exists; use identical --resume or a new run')
        if json.loads((root/'run.json').read_text())!=snapshot:
            raise ValueError('Long-study configuration/data/source/origin changed')
    else:
        previous={str(p):sha256(p) for p in Path('results').rglob('*') if p.is_file()}
        root.mkdir(parents=True,exist_ok=False)
        write_json(root/'run.json',snapshot)
        write_json(root/'original_results_sha256.json',previous)
        write_json(root/'environment.json',dict(python=platform.python_version(),
            versions={n:version(n) for n in ('tensorflow_cpu','keras','numpy','scipy','cantera','matplotlib')},
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],text=True),
            threads=dict(intra=tf.config.threading.get_intra_op_parallelism_threads(),
                         inter=tf.config.threading.get_inter_op_parallelism_threads()),
            cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
            memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
            thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS','TF_ENABLE_ONEDNN_OPTS')},
            authorization='User explicitly requested sequential long training and evaluation; maximum six hours'))
    return root,data,pairs,snapshot


def budget_deadlines(root,cfg):
    path=root/'budget.json'
    if path.exists():
        budget=json.loads(path.read_text())
        if budget['seconds']!=cfg['total_budget_seconds']:
            raise ValueError('Resume cannot restart/change the agreed budget')
    else:
        budget=dict(seconds=cfg['total_budget_seconds'],started_unix=time.time())
        budget['expires_unix']=budget['started_unix']+budget['seconds']
        budget['training_expires_unix']=budget['expires_unix']-cfg['evaluation_reserve_seconds']
        write_json(path,budget)
    now=time.time();clock=time.perf_counter()
    return (clock+max(0.,budget['training_expires_unix']-now),
            clock+max(0.,budget['expires_unix']-now))


def latest_record(root,family,seed):
    paths=sorted((root/'members'/family/f'seed{seed}').glob('finish_*.json'))
    return json.loads(paths[-1].read_text()) if paths else None


def train_member(root,cfg,snapshot,pairs,family,seed,deadline):
    started=time.perf_counter()
    model_cfg=snapshot['p_only_config'] if family=='p_only' else snapshot['base_config']
    directory=root/'checkpoints'/family/f'seed{seed}'
    paths=sorted(p.parent for p in directory.glob('*/complete.json'))
    if paths:
        trainer=restore_trainer(paths[-1],pairs['x'],pairs['y'],model_cfg,seed,family)
        mode='continuation_within_long_run'
    elif family=='p_only':
        origin=snapshot['p_only_origins'][str(seed)]
        trainer=restore_trainer(origin['checkpoint'],pairs['x'],pairs['y'],model_cfg,seed,family)
        trainer.plateau_monitor=new_monitor(cfg['plateau'],trainer.update,trainer.elapsed_seconds)
        mode='exact_adam_rng_continuation_from_10000'
    else:
        trainer=fresh_tp(pairs['x'],pairs['y'],model_cfg,seed)
        trainer.plateau_monitor=new_monitor(cfg['plateau'])
        mode='new_reproduction_from_initialization_old_adam_unavailable'
    engine=SerialSteps(trainer)
    if not trainer.full_train_history:
        engine.full_train_loss(deadline)
    reason='time_limit'
    checkpoint=paths[-1] if paths else None
    while time.perf_counter()<deadline and trainer.update<cfg['maximum_updates']:
        target=min((trainer.update//1000+1)*1000,cfg['maximum_updates'])
        if engine.advance(target,deadline)=='time_limit':
            break
        loss=engine.full_train_loss(deadline)
        plateau=loss is not None and check_plateau(trainer.full_train_history,trainer.update,trainer.plateau_monitor)
        if trainer.update%cfg['checkpoint_every']==0 or plateau:
            save_started=time.perf_counter()
            checkpoint=directory/f'update_{trainer.update:07d}_{time.time_ns()}'
            trainer.save_checkpoint(checkpoint,'plateau' if plateau else 'observed_checkpoint')
            engine.timing['checkpoint_seconds']+=time.perf_counter()-save_started
            event(root,phase='train_progress',family=family,seed=seed,update=trainer.update,
                full_train_loss=loss,plateau=plateau,member_active_seconds=time.perf_counter()-started)
        if plateau:
            reason='plateau';break
    else:
        if trainer.update>=cfg['maximum_updates']:
            reason='update_limit'
    # Always preserve an explicit terminal checkpoint, including an uneven
    # final update or deadline immediately after initialization.
    checkpoint=directory/f'update_{trainer.update:07d}_{time.time_ns()}'
    trainer.save_checkpoint(checkpoint,reason)
    full=trainer.full_train_history
    record=dict(family=family,seed=seed,updates=trainer.update,reason=reason,mode=mode,
        checkpoint=str(checkpoint),path=str(checkpoint/'model'),
        sha256=sha256(checkpoint/'model.keras'),preprocessing_sha256=sha256(checkpoint/'model.json'),
        parameters=trainer.predictor.model.count_params(),optimizer_iteration=int(trainer.optimizer.iterations.numpy()),
        last_full_train_loss=full[-1]['full_train_loss'] if full else None,
        last_full_train_loss_update=full[-1]['update'] if full else None,
        origin_update=trainer.plateau_monitor['origin_update'],
        origin_seconds=trainer.plateau_monitor['origin_seconds'],timing=engine.timing,
        active_seconds=time.perf_counter()-started,
        plateau_checks=trainer.plateau_monitor['checks'],test_used_for_selection=False)
    records=root/'members'/family/f'seed{seed}';records.mkdir(parents=True,exist_ok=True)
    write_json(records/f'finish_{trainer.update:07d}_{time.time_ns()}.json',record)
    event(root,phase='member_finished',family=family,seed=seed,update=trainer.update,reason=reason,
          active_seconds=record['active_seconds'])
    del trainer,engine
    gc.collect()
    return record


def train(root,cfg,snapshot,pairs,deadline):
    final=root/'final_members.json'
    if final.exists():
        return json.loads(final.read_text())
    tasks=[(f,s) for f in cfg['family_order'] for s in snapshot['base_config']['seeds']]
    # First pass is in the requested family/member order. Early plateau frees
    # time for remaining members. A second pass can use spare time on capped
    # members without resetting any optimizer/RNG or the global budget.
    for index,(family,seed) in enumerate(tasks):
        record=latest_record(root,family,seed)
        if record and record['reason'] in ('plateau','update_limit'):
            continue
        remaining=deadline-time.perf_counter()
        if remaining<=0:break
        member_deadline=time.perf_counter()+remaining/(len(tasks)-index)
        train_member(root,cfg,snapshot,pairs,family,seed,member_deadline)
    while deadline-time.perf_counter()>30:
        pending=[(f,s) for f,s in tasks if latest_record(root,f,s)
                 and latest_record(root,f,s)['reason']=='time_limit']
        if not pending:break
        for i,(family,seed) in enumerate(pending):
            remaining=deadline-time.perf_counter()
            if remaining<=0:break
            train_member(root,cfg,snapshot,pairs,family,seed,
                         time.perf_counter()+remaining/(len(pending)-i))
    records=[latest_record(root,f,s) for f,s in tasks]
    write_json(final,records)
    event(root,phase='training_finished',members_recorded=sum(r is not None for r in records),
          plateau_members=sum(bool(r and r['reason']=='plateau') for r in records))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/deeponet_long_study.json')
    parser.add_argument('--phase',choices=('prepare','train','evaluate','report','run'),default='prepare')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    root,data,pairs,snapshot=prepare(cfg,args.resume)
    if args.phase=='prepare':
        print('LONG_PREPARED',root,flush=True);return
    training_deadline,total_deadline=budget_deadlines(root,cfg)
    if args.phase in ('train','run'):
        train(root,cfg,snapshot,pairs,training_deadline)
    if args.phase in ('evaluate','run'):
        from .deeponet_long_evaluation import evaluate
        try:
            evaluate(root,cfg,snapshot,data,total_deadline)
        except TimeoutError:
            event(root,phase='evaluation_time_limit')
    if args.phase in ('report','run'):
        from .deeponet_long_report import report
        report(root,cfg,snapshot,data,total_deadline)


if __name__=='__main__':
    main()
