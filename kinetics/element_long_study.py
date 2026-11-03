"""Six-hour serial continuation of all40 soft-mu10 models and paired reports."""
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
from .p_only_training import sha256,write_json
from .soft_constraint_training import SoftTrainer,FAMILIES
from .soft_constraint_study import inputs_for
from .element_long_training import joint_monitor,joint_check
from .study import tf


def event(root,**values):
    values['utc_unix']=time.time()
    with (root/'execution.jsonl').open('a') as f:f.write(json.dumps(values,allow_nan=False)+'\n')
    print(json.dumps(values,allow_nan=False),flush=True)


def prepare(cfg,resume=False):
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='bw':raise ValueError('Use bw')
    if os.environ.get('TF_ENABLE_ONEDNN_OPTS')!='0':raise ValueError('Set oneDNN0 before TF import')
    if cfg['total_budget_seconds']!=21600 or cfg['evaluation_reserve_seconds']!=2400 or cfg['mu']!=10:
        raise ValueError('Preserve authorized six-hour mu10 plan')
    if cfg['components_required']!=['total','data','element'] or cfg['test_used_for_selection']:
        raise ValueError('Joint train-only component stopping required')
    policy=dict(minimum_updates=50000,check_every=5000,window_updates=20000,
        observe_every=1000,relative_thresholds=[.01,.01,.03],sustain_updates=10000)
    if cfg['plateau']!=policy:raise ValueError('Declared plateau policy changed')
    origin=Path(cfg['origin']);old=json.loads((origin/'run.json').read_text())
    data,pairs,fingerprint=load_inputs(old['base_config'])
    records=json.loads((origin/'final_members.json').read_text())
    if len(records)!=40 or any(r['updates']!=10000 or r['mu']!=10 for r in records):raise ValueError('Need40 mu10 origins')
    for r in records:
        cp=Path(r['checkpoint']);marker=json.loads((cp/'complete.json').read_text())
        for name,h in marker['output_sha256'].items():
            if sha256(cp/name)!=h:raise ValueError('Origin checkpoint changed')
        state=json.loads((cp/'state.json').read_text())
        if state['update']!=10000 or state['optimizer_iteration']!=10000:raise ValueError('Origin Adam mismatch')
    root=Path(cfg['output']);namespace=Path('results/element_soft_long').resolve()
    if root.resolve()==namespace or not root.resolve().is_relative_to(namespace):raise ValueError('New output required')
    origin_sources={str(p):sha256(p) for p in [origin/'run.json',origin/'final_members.json',
        origin/'baseline_traces.json',origin/'member_diagnostics/complete.json',
        origin/'member_diagnostics/loss_components.csv']}
    sources=list(Path('kinetics').glob('element_long_*.py'))+list(Path('scripts').glob('*element_long*.py'))
    snapshot=dict(config=cfg,origin_snapshot=old,origin_members=records,origin_sha256=origin_sources,
        base_config=old['base_config'],model_configs=old['model_configs'],
        element_matrix=old['element_matrix'],baseline_members=old['baseline_members'],
        fingerprint=fingerprint,source_sha256={str(p):sha256(p) for p in sources},
        authorization='User authorized continuation of all40 soft models, maximum6h including evaluation/plots')
    if root.exists():
        if not resume:raise FileExistsError('Use new output or identical resume')
        if json.loads((root/'run.json').read_text())!=snapshot:raise ValueError('Long resume provenance changed')
    else:
        previous={str(p):sha256(p) for p in Path('results').rglob('*') if p.is_file()}
        root.mkdir(parents=True,exist_ok=False);write_json(root/'run.json',snapshot)
        write_json(root/'original_results_sha256.json',previous)
        write_json(root/'environment.json',dict(python=platform.python_version(),
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],text=True),
            threads=dict(intra=tf.config.threading.get_intra_op_parallelism_threads(),
                         inter=tf.config.threading.get_inter_op_parallelism_threads()),
            thread_environment={k:os.environ.get(k) for k in ('TF_ENABLE_ONEDNN_OPTS','OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')}))
    return root,data,pairs,snapshot


def deadlines(root,cfg):
    path=root/'budget.json'
    if path.exists():budget=json.loads(path.read_text())
    else:
        budget=dict(started_unix=time.time(),seconds=21600)
        budget['expires_unix']=budget['started_unix']+budget['seconds']
        budget['training_expires_unix']=budget['expires_unix']-cfg['evaluation_reserve_seconds']
        write_json(path,budget)
    if budget['seconds']!=21600:raise ValueError('Resume cannot reset/change clock')
    now,clock=time.time(),time.perf_counter()
    return clock+max(0,budget['training_expires_unix']-now),clock+max(0,budget['expires_unix']-now)


def latest(root,family,seed):
    paths=sorted((root/'members'/family/f'seed{seed}').glob('finish_*.json'))
    return json.loads(paths[-1].read_text()) if paths else None


def train_member(root,cfg,snapshot,pairs,data,family,seed,deadline,maximum_update=None):
    started=time.perf_counter();x=inputs_for(family,pairs,data);y=pairs['y']
    prior=latest(root,family,seed)
    origin=next(r for r in snapshot['origin_members'] if r['family']==family and r['seed']==seed)
    cp=Path(prior['checkpoint'] if prior else origin['checkpoint'])
    trainer=SoftTrainer.restore(cp,x,y,snapshot['model_configs'][family],seed,family,
        np.array(snapshot['element_matrix']),10.)
    run_hash=sha256(root/'run.json')
    if prior:
        if trainer.plateau_monitor.get('long_run_sha256')!=run_hash:raise ValueError('Long monitor provenance mismatch')
    else:trainer.plateau_monitor=joint_monitor(cfg['plateau'],run_hash,trainer.update,trainer.elapsed_seconds)
    origin_update=trainer.update;reason='time_limit'
    directory=root/'checkpoints'/family/f'seed{seed}'
    while time.perf_counter()<deadline:
        if maximum_update is not None and trainer.update>=maximum_update:reason='test_update_limit';break
        target=(trainer.update//cfg['observe_every']+1)*cfg['observe_every']
        if maximum_update is not None:target=min(target,maximum_update)
        if trainer.advance(target,deadline)=='time_limit':break
        loss=trainer.full_train_loss(deadline)
        plateau=loss is not None and joint_check(trainer.full_train_history,trainer.update,trainer.plateau_monitor)
        if trainer.update%cfg['checkpoint_every']==0 or plateau:
            trainer.save_checkpoint(directory/f'update_{trainer.update:07d}_{time.time_ns()}',
                'joint_plateau' if plateau else 'observed_checkpoint')
            event(root,phase='train_progress',family=family,seed=seed,update=trainer.update,
                loss=loss,joint_plateau=bool(plateau),active_seconds=time.perf_counter()-started)
        if plateau:reason='joint_plateau';break
    cp=trainer.save_checkpoint(directory/f'update_{trainer.update:07d}_{time.time_ns()}',reason)
    record=dict(family=family,seed=seed,updates=trainer.update,reason=reason,mu=10,
        checkpoint=str(cp),path=str(cp/'model'),sha256=sha256(cp/'model.keras'),
        preprocessing_sha256=sha256(cp/'model.json'),parameters=trainer.predictor.model.count_params(),
        optimizer_iteration=int(trainer.optimizer.iterations.numpy()),origin_update=origin_update,
        added_updates=trainer.update-origin_update,active_seconds=time.perf_counter()-started,
        carried_seconds=trainer.elapsed_seconds,timing=trainer.timing,
        full_train=trainer.full_train_history[-1],monitor=trainer.plateau_monitor,test_used_for_selection=False)
    parent=root/'members'/family/f'seed{seed}';parent.mkdir(parents=True,exist_ok=True)
    write_json(parent/f'finish_{trainer.update:07d}_{time.time_ns()}.json',record)
    event(root,phase='member_finished',family=family,seed=seed,update=trainer.update,
        reason=reason,active_seconds=record['active_seconds'],added_updates=record['added_updates'])
    del trainer;gc.collect();return record


def train(root,cfg,snapshot,pairs,data,deadline):
    if (root/'final_members.json').exists():return
    tasks=[(f,s) for f in FAMILIES for s in snapshot['base_config']['seeds']]
    for i,(family,seed) in enumerate(tasks):
        r=latest(root,family,seed)
        if r and r['reason']=='joint_plateau':continue
        remaining=deadline-time.perf_counter()-10
        if remaining<=0:break
        train_member(root,cfg,snapshot,pairs,data,family,seed,time.perf_counter()+remaining/(len(tasks)-i))
    while deadline-time.perf_counter()>30:
        pending=[(f,s) for f,s in tasks if latest(root,f,s) and latest(root,f,s)['reason']=='time_limit']
        if not pending:break
        for i,(f,s) in enumerate(pending):
            remaining=deadline-time.perf_counter()-10
            if remaining<=0:break
            train_member(root,cfg,snapshot,pairs,data,f,s,time.perf_counter()+remaining/(len(pending)-i))
    records=[latest(root,f,s) for f,s in tasks]
    if any(r is None for r in records):raise TimeoutError('Not all40 models received continuation time')
    write_json(root/'final_members.json',records)
    event(root,phase='training_finished',members=40,joint_plateau_members=sum(r['reason']=='joint_plateau' for r in records))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/element_soft_long.json')
    parser.add_argument('--phase',choices=('prepare','run'),default='prepare');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());root,data,pairs,snapshot=prepare(cfg,args.resume)
    if args.phase=='prepare':print('ELEMENT_LONG_PREPARED',root,flush=True);return
    training_deadline,total_deadline=deadlines(root,cfg)
    before=time.perf_counter();train(root,cfg,snapshot,pairs,data,training_deadline)
    event(root,phase='training_wall_finished',seconds=time.perf_counter()-before)
    from .element_long_evaluation import evaluate
    before=time.perf_counter();evaluate(root,cfg,snapshot,data,pairs,total_deadline-600)
    event(root,phase='evaluation_wall_finished',seconds=time.perf_counter()-before)
    from scripts.report_element_long_comparison import report
    report(root,cfg,snapshot,data,total_deadline);event(root,phase='pipeline_finished')


if __name__=='__main__':main()
