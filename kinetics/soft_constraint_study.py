"""User-requested40x10000 soft-constraint run; preserve all prior studies."""
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
from .data import mechanism,SPECIES
from .p_only_training import sha256,write_json
from .soft_constraint_training import SoftTrainer,FAMILIES
from .study import tf


def event(root,**values):
    values['utc_unix']=time.time()
    with (root/'execution.jsonl').open('a') as f:f.write(json.dumps(values,allow_nan=False)+'\n')
    print(json.dumps(values,allow_nan=False),flush=True)


def model_config(base,family):
    cfg={**base,'hidden_layers':[64,64,64],'activation':'tanh'}
    cfg['input_partition']={'ann':'T_Y9_logP','p_only':'current_T_Y9__logP',
        'tp':'current_Y9__current_T_logP','tp_phi':'current_Y9__current_T_logP_initial_phi'}[family]
    return cfg


def inputs_for(family,pairs,data):
    if family!='tp_phi':return pairs['x']
    return np.c_[pairs['x'][:,:10],data['initials'][pairs['parent'][:,0],2],pairs['x'][:,-1]]


def prepare(cfg,resume):
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='bw':
        raise ValueError('Use the existing bw checkout')
    if os.environ.get('TF_ENABLE_ONEDNN_OPTS')!='0':raise ValueError('Set oneDNN0 before import')
    if (cfg['families']!=list(FAMILIES) or cfg['updates_per_member']!=10000 or cfg['mu']!=10
            or cfg['test_used_for_selection'] or cfg['modes']!=['raw','repaired']):
        raise ValueError('Keep the user-requested mu10,40x10k fixed experiment')
    root=Path(cfg['output']);namespace=Path('results/soft_constraint_10k').resolve()
    if root.resolve()==namespace or not root.resolve().is_relative_to(namespace):raise ValueError('New output namespace required')
    base=json.loads(Path(cfg['base_config']).read_text())
    data,pairs,fingerprint=load_inputs(base)
    gas=mechanism();elements=[str(e) for e in data['elements']]
    matrix=np.array([[gas.n_atoms(s,e)*gas.atomic_weight(e)/gas.molecular_weights[k]
        for k,s in enumerate(SPECIES)] for e in elements])
    if elements!=['H','O','N']:raise ValueError('Expected H/O/N element order')
    np.testing.assert_array_equal(matrix,data['element_matrix'])
    baseline_grid=Path(cfg['baseline_grid']);grid=json.loads((baseline_grid/'run.json').read_text())
    if grid['source_sha256'][base['source']]!=fingerprint['dense_sha256']:raise ValueError('Different baseline data')
    baseline_members={'ann':grid['members']['ann'],'p_only':grid['members']['p_short'],
        'tp':grid['members']['tp_short'],
        'tp_phi':json.loads(Path('results/tp_phi_study/run_001/matched_members.json').read_text())}
    for family,records in baseline_members.items():
        if len(records)!=10 or [r['seed'] for r in records]!=base['seeds']:raise ValueError('Baseline member order mismatch')
        for r in records:
            if r['updates']!=10000 or sha256(r['path']+'.keras')!=r['sha256']:
                raise ValueError('Incorrect baseline checkpoint')
    source_files=[Path(cfg['reference_step1']),Path(cfg['reference_step2']),
                  Path('scripts/nrmse_util_reference.py'),Path('references/user_nrmse/util.py')]
    source_files+=sorted(Path('scripts').glob('*soft_constraint*.py'))
    snapshot=dict(config=cfg,base_config=base,fingerprint=fingerprint,element_matrix=matrix.tolist(),
        elements=elements,baseline_members=baseline_members,
        model_configs={f:model_config(base,f) for f in FAMILIES},
        source_sha256={str(p):sha256(p) for p in source_files},
        authorization='Current user: all NN families freshly trained10000 updates each; mu10',
        preprocessing='Existing data, train-only Box-Cox gamma.1 and standardizers unchanged')
    if root.exists():
        if not resume:raise FileExistsError('New output required unless identical resume')
        if json.loads((root/'run.json').read_text())!=snapshot:raise ValueError('Resume provenance changed')
    else:
        previous={str(p):sha256(p) for p in Path('results').rglob('*') if p.is_file()}
        root.mkdir(parents=True,exist_ok=False);write_json(root/'run.json',snapshot)
        write_json(root/'original_results_sha256.json',previous)
        write_json(root/'environment.json',dict(python=platform.python_version(),
            versions={n:version(n) for n in ('tensorflow_cpu','keras','numpy','cantera','scipy','matplotlib')},
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],text=True),
            tensorflow_threads=dict(intra=tf.config.threading.get_intra_op_parallelism_threads(),
                inter=tf.config.threading.get_inter_op_parallelism_threads()),
            thread_environment={k:os.environ.get(k) for k in ('TF_ENABLE_ONEDNN_OPTS','OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')},
            cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip()))
    return root,data,pairs,snapshot


def deadlines(root,cfg):
    path=root/'budget.json'
    if path.exists():budget=json.loads(path.read_text())
    else:
        budget=dict(started_unix=time.time(),authorized_updates=400000,
            training_safety_seconds=cfg['training_wall_safety_seconds'],
            pipeline_safety_seconds=cfg['pipeline_wall_safety_seconds'])
        budget['training_expires_unix']=budget['started_unix']+budget['training_safety_seconds']
        budget['expires_unix']=budget['started_unix']+budget['pipeline_safety_seconds']
        write_json(path,budget)
    now,clock=time.time(),time.perf_counter()
    return clock+max(0,budget['training_expires_unix']-now),clock+max(0,budget['expires_unix']-now)


def train_member(root,cfg,snapshot,pairs,data,family,seed,deadline):
    started=time.perf_counter();x=inputs_for(family,pairs,data);y=pairs['y']
    config=snapshot['model_configs'][family];matrix=np.array(snapshot['element_matrix'])
    parent=root/'checkpoints'/family/f'seed{seed}';checkpoints=sorted(parent.glob('*/complete.json'))
    if checkpoints:
        trainer=SoftTrainer.restore(checkpoints[-1].parent,x,y,config,seed,family,matrix,cfg['mu'])
    else:trainer=SoftTrainer(x,y,config,seed,family,matrix,cfg['mu'])
    if not trainer.full_train_history:trainer.full_train_loss(deadline)
    reason='time_limit'
    while trainer.update<cfg['updates_per_member'] and time.perf_counter()<deadline:
        target=min((trainer.update//cfg['observe_every']+1)*cfg['observe_every'],cfg['updates_per_member'])
        if trainer.advance(target,deadline)=='time_limit':break
        loss=trainer.full_train_loss(deadline)
        if trainer.update%cfg['checkpoint_every']==0:
            trainer.save_checkpoint(parent/f'update_{trainer.update:06d}_{time.time_ns()}','observed_checkpoint')
            event(root,phase='train_progress',family=family,seed=seed,update=trainer.update,
                loss=loss,active_seconds=time.perf_counter()-started)
    if trainer.update==cfg['updates_per_member']:reason='fixed_update_budget'
    cp=trainer.save_checkpoint(parent/f'update_{trainer.update:06d}_{time.time_ns()}',reason)
    record=dict(family=family,seed=seed,updates=trainer.update,reason=reason,mu=cfg['mu'],
        checkpoint=str(cp),path=str(cp/'model'),sha256=sha256(cp/'model.keras'),
        preprocessing_sha256=sha256(cp/'model.json'),parameters=trainer.predictor.model.count_params(),
        optimizer_iteration=int(trainer.optimizer.iterations.numpy()),timing=trainer.timing,
        active_seconds=time.perf_counter()-started,
        full_train=trainer.full_train_history[-1] if trainer.full_train_history else None,
        carried_seconds=trainer.elapsed_seconds,test_used_for_selection=False)
    member=root/'members'/family;member.mkdir(parents=True,exist_ok=True)
    write_json(member/f'seed{seed}_{trainer.update:06d}_{time.time_ns()}.json',record)
    event(root,phase='member_finished',family=family,seed=seed,update=trainer.update,
        reason=reason,active_seconds=record['active_seconds'])
    del trainer;gc.collect()
    return record


def train(root,cfg,snapshot,pairs,data,deadline):
    if (root/'final_members.json').exists():return
    records=[]
    for family in FAMILIES:
        for seed in snapshot['base_config']['seeds']:
            prior=sorted((root/'members'/family).glob(f'seed{seed}_*.json'))
            record=json.loads(prior[-1].read_text()) if prior else None
            if not record or record['updates']<10000:
                if time.perf_counter()>=deadline:raise TimeoutError('Training safety limit; checkpoints preserved')
                record=train_member(root,cfg,snapshot,pairs,data,family,seed,deadline)
            if record['updates']!=10000:raise TimeoutError('Incomplete member; do not mark as complete')
            records.append(record)
    write_json(root/'final_members.json',records)
    event(root,phase='training_finished',members=40,updates=sum(r['updates'] for r in records))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/soft_constraint_10k.json')
    parser.add_argument('--phase',choices=('prepare','run'),default='prepare');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    root,data,pairs,snapshot=prepare(cfg,args.resume)
    if args.phase=='prepare':print('SOFT_PREPARED',root,flush=True);return
    training_deadline,total_deadline=deadlines(root,cfg)
    before=time.perf_counter();train(root,cfg,snapshot,pairs,data,training_deadline)
    event(root,phase='training_wall_finished',seconds=time.perf_counter()-before)
    from .soft_constraint_evaluation import evaluate
    before=time.perf_counter();evaluate(root,cfg,snapshot,data,pairs,total_deadline-600)
    event(root,phase='evaluation_wall_finished',seconds=time.perf_counter()-before)
    from scripts.report_soft_constraint_comparison import report
    report(root,cfg,snapshot,data,total_deadline)
    event(root,phase='pipeline_finished')


if __name__=='__main__':main()
