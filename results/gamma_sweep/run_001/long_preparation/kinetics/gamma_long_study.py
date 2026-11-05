"""Sequential pure-MSE continuation, cumulative150 seconds for each of240 models."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from .ann_deeponet_study import check_deadline
from .gamma_sweep_training import GammaTrainer,gamma_tag,weights_hash
from .gamma_long_training import MemberAllowance,restore_origin,restore_long,start_long_monitor
from .deeponet_long_training import check_plateau
from .soft_constraint_training import FAMILIES
from .soft_constraint_study import event,inputs_for
from .p_only_training import sha256,write_json
from .study import tf


def validate_config(cfg):
    if (cfg['gammas']!=[.05,.1,.2,.3,.5,1.] or cfg['member_budget_seconds']!=150
        or cfg['loss']!='transformed_residual_MSE_only' or cfg['test_used_for_selection']
        or cfg['evaluation_splits']!=['val','test'] or cfg['modes']!=['raw','repaired']):
        raise ValueError('Keep the six-gamma pure-MSE150-second plan')
    if cfg['training_budget_seconds']+cfg['evaluation_budget_seconds']+cfg['report_reserve_seconds']!=cfg['total_budget_seconds']:
        raise ValueError('Derived phase guards differ')
    if cfg['plateau']!=dict(minimum_updates=50000,check_every=5000,window_updates=20000,
                            observe_every=1000,relative_thresholds=[.01,.01,.03],sustain_updates=10000):
        raise ValueError('Preserve the original scalar MSE plateau policy')


def prepare(cfg,resume=False):
    validate_config(cfg)
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='bw':raise ValueError('Use bw')
    if os.environ.get('TF_ENABLE_ONEDNN_OPTS')!='0':raise ValueError('Set oneDNN0 before import')
    origin=Path(cfg['origin']);final=json.loads((origin/'FINAL_ARTIFACTS.json').read_text())
    attempt=sorted((origin/'process_launches').glob('attempt_*'))[-1]
    if json.loads((attempt/'exit.json').read_text())['returncode']!=0:raise ValueError('Finish10k evaluation first')
    if sha256(Path(final['report'])/'complete.json')!=final['report_complete_sha256']:raise ValueError('10k report changed')
    old=json.loads((origin/'run.json').read_text());records=json.loads((origin/'final_members.json').read_text())
    if len(records)!=240 or any(r['updates']!=10000 for r in records):raise ValueError('Need240 completed10k origins')
    inherited=[];rebuild=[]
    for r in records:
        if sha256(r['path']+'.keras')!=r['sha256']:raise ValueError('Short model changed')
        if r['gamma']==.1 and r['family'] in ('ann','tp'):
            if 'checkpoint' in r:raise ValueError('Reinspect newly available optimizer state')
            rebuild.append(r);continue
        if not (Path(r['checkpoint'])/'adam.npz').exists():raise ValueError('Missing expected Adam state')
        kind='legacy_exact_checkpoint' if r['gamma']==.1 else 'gamma_sweep_checkpoint'
        inherited.append(dict(r,origin_kind=kind))
    if len(inherited)!=220 or len(rebuild)!=20:raise ValueError('Expected220 exact states and20 missing-state rebuilds')
    with np.load(old['base_config']['source']) as f:data={k:f[k] for k in f.files}
    with np.load(old['base_config']['training_pairs']) as f:pairs={k:f[k] for k in f.files}
    source=sorted(Path('kinetics').glob('*.py'))+sorted(Path('scripts').glob('*.py'))+[Path('references/user_nrmse/util.py')]
    snapshot=dict(config=cfg,short_snapshot=old,base_config=old['base_config'],fingerprint=old['fingerprint'],
        model_configs=old['model_configs'],inherited_short_members=inherited,rebuild_requests=rebuild,
        source_sha256={str(p):sha256(p) for p in source},
        origin_sha256={str(origin/n):sha256(origin/n) for n in ('run.json','final_members.json','FINAL_ARTIFACTS.json','validation_selection.json')},
        origin_report=final['report'],authorization=cfg['authorization'],
        missing_state_policy='Rebuild gamma.1 ANN/T-P only; bootstrap counts within150s. Others retain exact Adam/RNG.',
        plateau_policy='Full-train scalar MSE only; no soft physical loss or validation stopping')
    root=Path(cfg['output']);namespace=Path('results/gamma_long').resolve()
    if root.resolve()==namespace or not root.resolve().is_relative_to(namespace):raise ValueError('New gamma_long output required')
    if root.exists():
        if not resume:raise FileExistsError('New output or identical resume required')
        if json.loads((root/'run.json').read_text())!=snapshot:raise ValueError('Long resume provenance changed')
    else:
        previous={str(p):sha256(p) for p in Path('results').rglob('*') if p.is_file()}
        root.mkdir(parents=True,exist_ok=False);write_json(root/'run.json',snapshot)
        write_json(root/'original_results_sha256.json',previous)
        write_json(root/'environment.json',dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],text=True),
            thread_environment={k:os.environ.get(k) for k in ('TF_ENABLE_ONEDNN_OPTS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')}))
    return root,data,pairs,snapshot


def allowance_for(root,record,cfg):
    return MemberAllowance(root/'member_allowances'/gamma_tag(record['gamma'])/record['family']/f"seed{record['seed']}.json",cfg['member_budget_seconds'])


def bootstrap(root,cfg,snapshot,pairs,data,deadline):
    path=root/'paired_short_members.json'
    if path.exists():return json.loads(path.read_text())
    result=list(snapshot['inherited_short_members'])
    for old in snapshot['rebuild_requests']:
        check_deadline(deadline);family,seed=old['family'],old['seed']
        folder=root/'bootstrap_members'/family/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'short.json').exists():result.append(json.loads((folder/'short.json').read_text()));continue
        allowance=allowance_for(root,old,cfg);stop=allowance.begin('rebuild_short10k',deadline)
        if stop is None:raise TimeoutError('Missing-state bootstrap exhausted its member allowance')
        x,y=inputs_for(family,pairs,data),pairs['y'];config={**snapshot['model_configs'][family],'gamma':.1}
        parent=root/'bootstrap_checkpoints'/family/f'seed{seed}';checkpoints=sorted(parent.glob('*/complete.json'))
        try:
            trainer=GammaTrainer.restore(checkpoints[-1].parent,x,y,config,seed,family,.1) if checkpoints else GammaTrainer(x,y,config,seed,family,.1)
            if trainer.update==0:
                expected=json.loads((Path(cfg['origin'])/'gamma_0.05/members'/family/f'seed{seed}'/'initialization.json').read_text())
                if weights_hash(trainer.predictor)!=expected['initial_weights_sha256']:raise ValueError('Rebuilt origin initialization differs')
            if not trainer.full_train_history:trainer.full_train_loss(stop)
            while trainer.update<10000 and time.perf_counter()<stop:
                target=min(trainer.update+1000,10000)
                if trainer.advance(target,stop)=='time_limit':break
                trainer.full_train_loss(stop)
            cp=trainer.save_checkpoint(parent/f'update_{trainer.update:07d}_{time.time_ns()}','rebuilt_short_10000' if trainer.update==10000 else 'bootstrap_time_limit')
            if trainer.update!=10000:raise TimeoutError('Cannot claim a matched10k rebuilt origin')
            r=dict(gamma=.1,family=family,seed=seed,updates=10000,origin_kind='rebuilt_short_checkpoint',
                checkpoint=str(cp),path=str(cp/'model'),sha256=sha256(cp/'model.keras'),
                preprocessing_sha256=sha256(cp/'model.json'),parameters=trainer.predictor.model.count_params(),
                historical_short_path=old['path'],historical_short_sha256=old['sha256'])
            write_json(folder/'short.json',r);result.append(r);del trainer
        finally:allowance.close()
        event(root,phase='bootstrap_member_finished',family=family,seed=seed,consumed_seconds=allowance.state['consumed_seconds'])
        gc.collect()
    result.sort(key=lambda r:(cfg['gammas'].index(r['gamma']),FAMILIES.index(r['family']),r['seed']))
    write_json(path,result);event(root,phase='paired_short_origins_frozen',members=len(result),rebuilt=20)
    return result


def train(root,cfg,snapshot,short_records,pairs,data,deadline):
    if (root/'final_members.json').exists():return
    records=[];run_sha=sha256(root/'run.json')
    for origin in short_records:
        check_deadline(deadline);gamma,family,seed=origin['gamma'],origin['family'],origin['seed']
        destination=root/'members'/gamma_tag(gamma)/family/f'seed{seed}'
        terminal=sorted(destination.glob('finish_*.json'))
        if terminal:records.append(json.loads(terminal[-1].read_text()));continue
        destination.mkdir(parents=True,exist_ok=True);allowance=allowance_for(root,origin,cfg)
        parent=root/'checkpoints'/gamma_tag(gamma)/family/f'seed{seed}';checkpoints=sorted(parent.glob('*/complete.json'))
        stop=allowance.begin('long_continuation',deadline)
        if stop is None:
            if checkpoints:
                cp=checkpoints[-1].parent;state=json.loads((cp/'state.json').read_text())
                marker=json.loads((cp/'complete.json').read_text())
                for name,expected in marker['output_sha256'].items():
                    if sha256(cp/name)!=expected:raise ValueError('Recovered checkpoint changed')
                monitor=state['plateau_monitor']
                if monitor.get('long_run_sha256')!=run_sha:raise ValueError('Recovered monitor lineage changed')
                r=dict(gamma=gamma,family=family,seed=seed,updates=state['update'],
                    reason='plateau' if monitor.get('plateau') else 'member_time_limit',added_updates=state['update']-10000,
                    checkpoint=str(cp),path=str(cp/'model'),sha256=sha256(cp/'model.keras'),
                    preprocessing_sha256=sha256(cp/'model.json'),optimizer_iteration=state['optimizer_iteration'],
                    parameters=origin['parameters'],origin_checkpoint=origin['checkpoint'],origin_kind=origin['origin_kind'],
                    borrowed_short_endpoint=False,monitor=monitor,allowance=allowance.state)
            else:
                r=dict(origin,origin_checkpoint=origin['checkpoint'],reason='member_time_limit',added_updates=0,
                    borrowed_short_endpoint=True,monitor=None,allowance=allowance.state)
            write_json(destination/f'finish_{r["updates"]:07d}_{time.time_ns()}.json',r);records.append(r)
            event(root,phase='member_finished',gamma=gamma,family=family,seed=seed,update=r['updates'],reason=r['reason'],consumed_seconds=allowance.state['consumed_seconds'])
            continue
        event(root,phase='member_started',gamma=gamma,family=family,seed=seed,remaining_seconds=allowance.remaining)
        started=time.perf_counter();x,y=inputs_for(family,pairs,data),pairs['y']
        try:
            tf.keras.backend.clear_session()
            if checkpoints:trainer=restore_long(checkpoints[-1].parent,x,y,origin,run_sha)
            else:
                trainer=restore_origin(origin,x,y);start_long_monitor(trainer,cfg['plateau'],run_sha,origin)
            reason='plateau' if trainer.plateau_monitor.get('plateau') else 'member_time_limit'
            while not trainer.plateau_monitor.get('plateau') and time.perf_counter()<stop:
                target=(trainer.update//cfg['observe_every']+1)*cfg['observe_every']
                if trainer.advance(target,stop)=='time_limit':break
                loss=trainer.full_train_loss(stop)
                if loss is None:break
                if check_plateau(trainer.full_train_history,trainer.update,trainer.plateau_monitor):reason='plateau';break
                if trainer.update%cfg['checkpoint_every']==0:
                    trainer.save_checkpoint(parent/f'update_{trainer.update:07d}_{time.time_ns()}','observed_checkpoint')
                    event(root,phase='train_progress',gamma=gamma,family=family,seed=seed,update=trainer.update,full_train_MSE=loss)
            cp=trainer.save_checkpoint(parent/f'update_{trainer.update:07d}_{time.time_ns()}',reason)
            r=dict(gamma=gamma,family=family,seed=seed,updates=trainer.update,reason=reason,added_updates=trainer.update-10000,
                checkpoint=str(cp),path=str(cp/'model'),sha256=sha256(cp/'model.keras'),preprocessing_sha256=sha256(cp/'model.json'),
                optimizer_iteration=int(trainer.optimizer.iterations.numpy()),parameters=trainer.predictor.model.count_params(),
                origin_checkpoint=origin['checkpoint'],origin_kind=origin['origin_kind'],
                active_visit_seconds=time.perf_counter()-started,monitor=trainer.plateau_monitor,
                last_full_train=trainer.full_train_history[-1] if trainer.full_train_history else None,
                borrowed_short_endpoint=False)
            del trainer
        finally:allowance.close()
        r['allowance']=allowance.state;write_json(destination/f'finish_{r["updates"]:07d}_{time.time_ns()}.json',r)
        records.append(r);event(root,phase='member_finished',gamma=gamma,family=family,seed=seed,update=r['updates'],
            reason=r['reason'],consumed_seconds=allowance.state['consumed_seconds'])
        gc.collect()
    write_json(root/'final_members.json',records)
    event(root,phase='long_endpoints_frozen',members=len(records),plateau=sum(r['reason']=='plateau' for r in records))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/gamma_long.json')
    parser.add_argument('--phase',choices=('prepare','run'),default='prepare');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());root,data,pairs,snapshot=prepare(cfg,args.resume)
    if args.phase=='prepare':print('GAMMA_LONG_PREPARED',root,flush=True);return
    budget=json.loads((root/'budget.json').read_text());now=time.time();clock=time.perf_counter()
    train_deadline=clock+max(0,budget['training_expires_unix']-now);total_deadline=clock+max(0,budget['expires_unix']-now)
    before=time.perf_counter();short=bootstrap(root,cfg,snapshot,pairs,data,train_deadline)
    from .gamma_long_evaluation import bootstrap_evaluations,evaluate
    bootstrap_evaluations(root,cfg,snapshot,short,data,train_deadline)
    train(root,cfg,snapshot,short,pairs,data,train_deadline)
    event(root,phase='training_wall_finished',seconds=time.perf_counter()-before)
    before=time.perf_counter();evaluate(root,cfg,snapshot,data,min(total_deadline-cfg['report_reserve_seconds'],time.perf_counter()+cfg['evaluation_budget_seconds']))
    event(root,phase='evaluation_wall_finished',seconds=time.perf_counter()-before)
    from scripts.report_gamma_long import report
    report(root,cfg,snapshot,data,min(total_deadline,time.perf_counter()+cfg['report_reserve_seconds']))
    event(root,phase='pipeline_finished')


if __name__=='__main__':main()
