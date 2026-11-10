"""Controlled continuation of the completed React-DeepONet reconstruction.

The old run is read-only. New held-out data stay closed until validation-only
selection is frozen. Physical correction is an inference-only comparison.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/react-refinement-mpl')
import argparse,csv,json,time,shutil,fcntl,traceback,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import cantera as ct
from .react_deeponet_experiment import FifthRootScale,write,csv_write,sha,scores,valid_state,ORIGINALS
from .react_deeponet_v2_model import DetailedReactDeepONet
from .react_deeponet_reconstruction import ReconstructionNet
from .react_deeponet_rebuild import REP,REST,rollout,mathematical_representative

AUDIT=Path('/workspace/react_refinement_followup/run_001')
ACTIVE=np.arange(9)


def fresh_cases(cfg,previous):
    used={(round(c['T0'],6),round(c['phi'],6)) for c in previous};cases=[]
    for i,(t,p) in enumerate(cfg['new_validation']):
        pair=(round(t,6),round(p,6));assert pair not in used;used.add(pair)
        cases.append(dict(case_id=10000+i,split='val',domain='extrapolation',T0=float(t),phi=float(p)))
    rng=np.random.default_rng(cfg['fresh_seed']);n=cfg['holdout_per_group'];side=int(math.ceil(math.sqrt(n)))
    for group in ['interpolation','extrapolation']:
        for i in range(n):
            for _ in range(100):
                if group=='interpolation':
                    t=1000+200*((i//side)+rng.random())/side;p=.75+.4*((i%side)+rng.random())/side
                elif i%4==0:t=rng.uniform(980,1000);p=rng.uniform(.6,1.3)
                elif i%4==1:t=rng.uniform(1200,1240);p=rng.uniform(.6,1.3)
                elif i%4==2:t=rng.uniform(1000,1200);p=rng.uniform(.6,.75)
                else:t=rng.uniform(1000,1200);p=rng.uniform(1.15,1.3)
                pair=(round(float(t),6),round(float(p),6))
                if pair not in used:break
            else:raise RuntimeError('Unable to create disjoint IC')
            used.add(pair);cases.append(dict(case_id=(20000 if group=='interpolation' else 20100)+i,split='test_'+group,domain=group,T0=pair[0],phi=pair[1]))
    return cases


def reference(case,mechanism):
    gas=ct.Solution(str(mechanism));gas.TP=case['T0'],ct.one_atm;gas.set_equivalence_ratio(case['phi'],'H2','O2:1,N2:3.76')
    r=ct.IdealGasConstPressureReactor(gas,clone=False);net=ct.ReactorNet([r]);net.rtol=1e-10;net.atol=1e-20;net.max_time_step=1e-7
    a=np.empty((10001,11));a[0]=np.r_[gas.T,gas.Y]
    for j in range(1,10001):net.advance(j*1e-7);a[j]=np.r_[gas.T,gas.Y]
    if a[:,1:].min() < -1e-14:raise ValueError('Reference negative beyond roundoff')
    a[:,1:]=np.maximum(a[:,1:],0)
    return a


def prepare(root,config):
    if root.exists():raise FileExistsError('Use a new experiment root')
    started=time.perf_counter();cfg=json.loads(Path(config).read_text());source=Path(cfg['source_root'])
    assert json.loads((source/'complete.json').read_text())['status']=='passed'
    if cfg.get('software_fixture'):assert str(root.resolve()).startswith('/tmp/')
    root.mkdir(parents=True)
    for folder in ['data','inputs','models','evaluations','report','preflight','source_snapshot/kinetics']:(root/folder).mkdir(parents=True)
    write(root/'config.json',cfg);files=['react_refinement.py','react_refinement_report.py','react_deeponet_rebuild.py','react_deeponet_rebuild_report.py','react_deeponet_v2_model.py','react_deeponet_diagnostics.py','react_deeponet_experiment.py','react_deeponet_reconstruction.py','physics.py','data.py'];manifest={}
    for name in files:
        p=Path('kinetics')/name;shutil.copyfile(p,root/'source_snapshot/kinetics'/name);manifest['kinetics/'+name]=sha(p)
    (root/'source_snapshot/kinetics/__init__.py').write_text('"""Frozen refinement experiment."""\n')
    (root/'source_snapshot/run.py').write_text('from kinetics.react_refinement import main\nmain()\n')
    shutil.copyfile('scripts/supervise_react_refinement.py',root/'source_snapshot/supervise.py')
    for name in ['kinetics/__init__.py','run.py','supervise.py']:manifest[name]=sha(root/'source_snapshot'/name)
    write(root/'source_manifest.json',manifest)
    source_verification=json.loads((source/'verification.json').read_text());input_hashes={}
    for name,filename in dict(**cfg['baselines'],reconstruction='reconstruction_selected.pt').items():
        p=source/'models'/filename;assert sha(p)==source_verification['output_sha256']['models/'+filename]
        dest=root/'inputs'/f'{name}.pt';shutil.copyfile(p,dest);input_hashes[str(dest.relative_to(root))]=sha(dest)
    for name in ['preprocessing.json','mechanism.yaml']:
        shutil.copyfile(source/name,root/name);input_hashes[name]=sha(root/name)
    previous=json.loads((source/'cases.json').read_text());train_cases=[dict(c,domain='training') for c in previous if c['split']=='train'];old_val=[dict(c,domain='interpolation') for c in previous if c['split']=='val'];new=fresh_cases(cfg,previous)
    with np.load(source/'data/val.npz') as f:val=f['states']
    new_val=[c for c in new if c['split']=='val']
    if cfg.get('software_fixture'):
        train_cases=train_cases[:2];old_val=old_val[:1];val=val[:1];new_val=new_val[:1]
        a=np.load(source/'data/train.npz')['states'][:2];np.savez_compressed(root/'data/train.npz',states=a)
    else:shutil.copyfile(source/'data/train.npz',root/'data/train.npz')
    combined_val=np.concatenate([val,np.stack([reference(c,root/'mechanism.yaml') for c in new_val])])
    np.savez_compressed(root/'data/val.npz',states=combined_val)
    cases=train_cases+old_val+new_val
    for split in ['test_interpolation','test_extrapolation']:
        group=[c for c in new if c['split']==split];array=np.stack([reference(c,root/'mechanism.yaml') for c in group]);np.savez_compressed(root/'data'/f'{split}.npz',states=array);cases.extend(group)
    assert len({(c['T0'],c['phi']) for c in cases})==len(cases)
    write(root/'cases.json',cases);input_hashes['cases.json']=sha(root/'cases.json')
    for p in (root/'data').iterdir():input_hashes[str(p.relative_to(root))]=sha(p)
    for name in ['complete.json','verification.json','source_manifest.json']:
        shutil.copyfile(source/name,root/'preflight'/('prior_'+name))
    historical=[c for c in previous if c['split'] in ['paper_test','fresh_test']]
    write(root/'historical_cases.json',historical);input_hashes['historical_cases.json']=sha(root/'historical_cases.json')
    historic_inputs={str(source/'data'/f'{s}.npz'):sha(source/'data'/f'{s}.npz') for s in ['paper_test','fresh_test']}
    write(root/'historical_input_manifest.json',historic_inputs);input_hashes['historical_input_manifest.json']=sha(root/'historical_input_manifest.json')
    write(root/'dataset.json',dict(variables=json.loads((source/'dataset.json').read_text())['variables'],counts={s:sum(c['split']==s for c in cases) for s in ['train','val','test_interpolation','test_extrapolation']},train_normalization_unchanged=True,holdout_not_used_for_selection=True,historical_test_already_inspected=True,mechanism_limit='Public Burke H2/air10,25 Cantera entries; exact author20-reaction file unavailable'))
    write(root/'input_manifest.json',input_hashes)
    prior=[]
    for p in [Path('results/react_refinement_transition/run_001/complete.json'),AUDIT/'preflight.json']:
        if p.exists():prior.append(dict(path=str(p),sha256=sha(p),elapsed_seconds=json.loads(p.read_text())['elapsed_seconds']))
    if (AUDIT/'preflight.json').exists():shutil.copyfile(AUDIT/'preflight.json',root/'preflight/software_checks.json')
    write(root/'prepared.json',dict(status='passed',config_sha256=sha(root/'config.json'),source_manifest_sha256=sha(root/'source_manifest.json'),input_manifest_sha256=sha(root/'input_manifest.json'),dataset_sha256=sha(root/'dataset.json'),prior_complete_sha256=sha(source/'complete.json'),prior_compute=prior,prior_seconds=sum(v['elapsed_seconds'] for v in prior),prepare_seconds=time.perf_counter()-started,originals=ORIGINALS))
    write(root/'status.json',dict(phase='prepared',training_started=False));print('PREPARED',str(root),flush=True)


def sample_targets(states,phis,batch,maximum_offset,generator,steps=5):
    i=torch.randint(len(states),(batch,),generator=generator);origin=torch.randint(states.shape[1]-maximum_offset*steps,(batch,),generator=generator);offset=torch.randint(0,maximum_offset+1,(batch,),generator=generator)
    targets=torch.stack([states[i,origin+(k+1)*offset] for k in range(steps)])
    return states[i,origin],offset[:,None].float()*1e-7,phis[i,None],targets


def objective(model,state,tau,phi,targets,family):
    predictions=[]
    for _ in range(len(targets)):state=model(state,tau,phi);predictions.append(state)
    if family=='endpoint':return (predictions[-1]-targets[-1]).square().mean()
    if family=='all_steps':return (torch.stack(predictions)-targets).square().mean()
    raise ValueError(family)


def profile_scores(truth,pred,active=ACTIVE):
    complete=len(pred)==len(truth);full=scores(truth,pred) if complete else None;early=scores(truth[:1001],pred[:1001]) if len(pred)>=1001 else None
    return dict(complete=complete,points=len(pred),full=full,early=early,pass_full=bool(full is not None and np.max(full[active])<=.1),pass_both=bool(full is not None and early is not None and np.max(full[active])<=.1 and np.max(early[active])<=.1))


def ranking(domains,diagnostic_errors):
    both=sum(d['pass_both'] for d in domains.values());full=sum(d['pass_full'] for d in domains.values());complete=sum(d['complete'] for d in domains.values())
    balanced=min(d['pass_both']/d['total'] for d in domains.values());error=float(np.median(diagnostic_errors))
    return (-balanced,-both,-full,-complete,error)


def validation_metrics(model,reconstruction,val,cases,scale,mechanism):
    model.eval();domains={d:dict(total=0,complete=0,pass_full=0,pass_both=0) for d in sorted({c['domain'] for c in cases})};errors=[]
    for truth,case in zip(val,cases):
        pred,info=rollout(model,reconstruction,truth[0],case['phi'],scale,'raw_full',mechanism);metric=profile_scores(truth,pred)
        d=domains[case['domain']];d['total']+=1
        for k in ['complete','pass_full','pass_both']:d[k]+=int(metric[k])
        diag,_=mathematical_representative(model,scale.subset(REP),truth[0,REP],case['phi'])
        value=float(np.max(scores(truth[:,REP],diag))) if len(diag)==len(truth) else 1e6;errors.append(value if np.isfinite(value) else 1e6)
    rank=ranking(domains,errors);model.train()
    return rank,dict(domains=domains,complete=sum(d['complete'] for d in domains.values()),pass_full=sum(d['pass_full'] for d in domains.values()),pass_both=sum(d['pass_both'] for d in domains.values()),total=len(cases),balanced_both=-rank[0],median_diagnostic_worst5=rank[-1])


def train_candidate(root,cfg,spec,train,phis,val,val_cases,scale,reconstruction,target,deadline,phase):
    folder=root/'models'/spec['name'];folder.mkdir(exist_ok=True);last=folder/'last.pt';resuming=last.exists();saved=torch.load(last if resuming else root/'inputs/last_pretrain.pt',weights_only=False)
    torch.manual_seed(cfg['seed']);model=DetailedReactDeepONet();model.load_state_dict(saved['model']);opt=torch.optim.Adamax(model.parameters(),lr=1e-5)
    if resuming or not spec['reset_optimizer']:opt.load_state_dict(saved['optimizer'])
    generator=torch.Generator();generator.set_state(saved['generator_state']);n=saved['new_update'] if resuming else 0;active=saved['active_seconds'] if resuming else 0.
    compiled=torch.compile(model,fullgraph=True,dynamic=False) if cfg['compile_operator'] else model
    history=list(csv.DictReader((folder/'training.csv').open())) if resuming else [];validations=[]
    if resuming:
        selection=json.loads((folder/'selection.json').read_text());best_rank=tuple(selection['best_rank']);best_update=selection['selected_new_update'];validations=selection['validations']
    else:
        best_rank,metric=validation_metrics(model,reconstruction,val,val_cases,scale,root/'mechanism.yaml');best_update=0;validations.append(dict(new_update=0,rank=best_rank,**metric));shutil.copyfile(root/'inputs/last_pretrain.pt',folder/'selected.pt')
    started=time.perf_counter();reason='probe_updates_completed' if phase=='probing' else 'requested_updates_completed';latest=validations[-1]
    def checkpoint():
        nonlocal best_rank,best_update,latest
        rank,metric=validation_metrics(model,reconstruction,val,val_cases,scale,root/'mechanism.yaml');latest=dict(new_update=n,rank=rank,**metric);validations.append(latest)
        payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),generator_state=generator.get_state(),torch_rng=torch.get_rng_state(),new_update=n,base_global_update=cfg['base_global_update'],active_seconds=active,spec=spec)
        torch.save(payload,last)
        if rank<best_rank:best_rank=rank;best_update=n;torch.save(payload,folder/'selected.pt')
        write(folder/'selection.json',dict(spec=spec,best_rank=best_rank,selected_new_update=best_update,last_new_update=n,latest=latest,validations=validations,active_seconds=active,last_lr=opt.param_groups[0]['lr'],stop_reason=reason))
        if history:csv_write(folder/'training.csv',history)
    while n<target:
        if time.perf_counter()>=deadline:reason='overall_time_cap';break
        n+=1;lr=cfg['base_lr']*cfg['decay_rate']**((cfg['base_global_update']+n-1)/cfg['decay_steps'])*spec['lr_multiplier'];opt.param_groups[0]['lr']=lr
        tick=time.perf_counter();x,t,p,y=sample_targets(train,phis,cfg['batch_size'],cfg['maximum_offset'],generator,cfg['K']);opt.zero_grad(set_to_none=True);loss=objective(compiled,x,t,p,y,spec['family'])
        if not torch.isfinite(loss):
            active+=time.perf_counter()-tick;n-=1;reason='nonfinite_loss';write(folder/'numerical_failure.json',dict(reason=reason,attempted_new_update=n+1));break
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            active+=time.perf_counter()-tick;n-=1;reason='nonfinite_gradient';write(folder/'numerical_failure.json',dict(reason=reason,attempted_new_update=n+1));break
        opt.step();active+=time.perf_counter()-tick
        if n==1 or n%cfg['history_every']==0:history.append(dict(candidate=spec['name'],phase=phase,new_update=n,global_update=cfg['base_global_update']+n,loss=float(loss.detach()),lr=lr,active_seconds=active))
        if n==1 or n%cfg['validation_every']==0 or n==target:checkpoint()
        if n==1 or n%100==0:write(root/'status.json',dict(phase=phase,candidate=spec['name'],new_update=n,target=target,active_seconds=active,latest_validation=latest))
    if not last.exists() or json.loads((folder/'selection.json').read_text())['last_new_update']!=n:checkpoint()
    selection=json.loads((folder/'selection.json').read_text());selection.update(stop_reason=reason,stage_wall_seconds=time.perf_counter()-started);write(folder/'selection.json',selection)
    if not (folder/'training.csv').exists():(folder/'training.csv').write_text('candidate,phase,new_update,global_update,loss,lr,active_seconds\n')
    return selection


def load_operator(path):
    model=DetailedReactDeepONet();model.load_state_dict(torch.load(path,weights_only=False)['model']);model.eval();return model


def arc_time_pool(encoded,size):
    a=np.asarray(encoded);weights=np.abs(np.diff(a[:,:,:9],axis=1)).sum(axis=2,dtype=np.float64)+1e-12;pool=[]
    for w in weights:
        cdf=np.r_[0.,np.cumsum(w)];indices=np.searchsorted(cdf,np.linspace(0,1,size)*cdf[-1]).clip(0,len(cdf)-1);indices[0]=0;indices[-1]=len(cdf)-1;pool.append(indices)
    return torch.from_numpy(np.asarray(pool,dtype=np.int64))


def sample_reconstruction(states,pool,n,generator,balanced):
    i=torch.randint(len(states),(n,),generator=generator);t=torch.randint(states.shape[1],(n,),generator=generator);q=torch.randint(pool.shape[1],(n,),generator=generator);mix=torch.rand(n,generator=generator)
    if balanced:t=torch.where(mix<.5,pool[i,q],t)
    return states[i,t]


def reconstruction_metric(model,encoded_val,scale,truth):
    with torch.inference_mode():z=model(encoded_val[:,:,REP].reshape(-1,5));loss=float((z-encoded_val[:,:,REST].reshape(-1,6)).square().mean());pred=scale.subset(REST).inverse(z.numpy()).reshape(len(encoded_val),10001,6)
    both=0;full=0;physical=0;values=[]
    for p,r in zip(pred,truth[:,:,REST]):
        a=scores(r,p)[:4];b=scores(r[:1001],p[:1001])[:4];ok=bool(np.isfinite(p).all() and p.min()>=0)
        both+=int(ok and np.max(a)<=.1 and np.max(b)<=.1);full+=int(ok and np.max(a)<=.1);physical+=int(ok);v=float(np.max(a));values.append(v if np.isfinite(v) else 1e6)
    return (-both,-full,-physical,float(np.median(values)),loss),dict(teacher_forced_both=both,teacher_forced_full=full,teacher_forced_positive=physical,total=len(pred),median_worst4=float(np.median(values)),normalized_MSE=loss)


def train_reconstruction_controls(root,cfg,train,val,scale,deadline):
    x=torch.from_numpy(scale.forward(train));v=torch.from_numpy(scale.forward(val));pool=arc_time_pool(x.numpy(),cfg['reconstruction_time_pool_size']);results={}
    allocation=max(0.,(deadline-time.perf_counter())/2)
    for name,balanced in [('reconstruction_uniform',False),('reconstruction_arc',True)]:
        folder=root/'models'/name;folder.mkdir();saved=torch.load(root/'inputs/reconstruction.pt',weights_only=False);model=ReconstructionNet();model.load_state_dict(saved['model']);opt=torch.optim.Adam(model.parameters(),lr=cfg['reconstruction_lr']);opt.load_state_dict(saved['optimizer']);generator=torch.Generator();generator.set_state(saved['generator_state'])
        rank,metric=reconstruction_metric(model,v,scale,val);best=rank;best_metric=metric;best_update=0;active=0.;history=[];shutil.copyfile(root/'inputs/reconstruction.pt',folder/'selected.pt');end=min(deadline,time.perf_counter()+allocation);reason='requested_updates_completed';n=0
        for j in range(1,cfg['reconstruction_additional_updates']+1):
            if time.perf_counter()>end:reason='overall_time_cap';break
            lr=cfg['reconstruction_lr']*.9**((j-1)/5000);opt.param_groups[0]['lr']=lr;tick=time.perf_counter();batch=sample_reconstruction(x,pool,cfg['reconstruction_batch'],generator,balanced);opt.zero_grad(set_to_none=True);loss=(model(batch[:,REP])-batch[:,REST]).square().mean()
            if not torch.isfinite(loss):reason='nonfinite_loss';break
            loss.backward();opt.step();active+=time.perf_counter()-tick;n=j
            if j==1 or j%100==0:history.append(dict(candidate=name,new_update=j,loss=float(loss.detach()),lr=lr,active_seconds=active))
            if j==1 or j%1000==0 or j==cfg['reconstruction_additional_updates']:
                rank,metric=reconstruction_metric(model,v,scale,val)
                if rank<best:best=rank;best_metric=metric;best_update=j;torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator_state=generator.get_state(),new_update=j),folder/'selected.pt')
                write(root/'status.json',dict(phase='reconstruction_controls',candidate=name,new_update=j,target=cfg['reconstruction_additional_updates'],validation=metric))
                csv_write(folder/'training.csv',history)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator_state=generator.get_state(),new_update=n),folder/'last.pt')
        results[name]=dict(path=str((folder/'selected.pt').relative_to(root)),selected_new_update=best_update,last_new_update=n,rank=best,stop_reason=reason,active_seconds=active,selected_validation=best_metric,final_validation=metric,last_lr=opt.param_groups[0]['lr'])
        write(root/'reconstruction_controls.json',results)
    return results


def evaluate(root,cfg,selection,cases,scale,reconstruction,deadline):
    variables=json.loads((root/'dataset.json').read_text())['variables'];rows=[];metrics=[];rec_rows=[]
    selected_sha=sha(root/'selection_frozen.json');write(root/'test_access_started.json',dict(at=datetime.now(timezone.utc).isoformat(),selection_sha256=selected_sha))
    datasets=[]
    for split in ['test_interpolation','test_extrapolation']:
        a=np.load(root/'data'/f'{split}.npz')['states'];datasets.append((split,[c for c in cases if c['split']==split],a))
    if cfg['historical_evaluation']:
        previous=Path(cfg['source_root']);historical=json.loads((root/'historical_cases.json').read_text())
        for split in ['paper_test','fresh_test']:
            a=np.load(previous/'data'/f'{split}.npz')['states'];group=[dict(c,split='historical_'+split,domain='extrapolation' if split=='paper_test' else 'interpolation') for c in historical if c['split']==split];datasets.append(('historical_'+split,group,a))
    for split,group,states in datasets:
        primary=split.startswith('test_')
        if primary:
            for truth,case in zip(states,group):
                with torch.inference_mode():a=scale.subset(REST).inverse(reconstruction(torch.from_numpy(scale.subset(REP).forward(truth[:,REP]))).numpy())
                e=scores(truth[:,REST],a)
                for j,k in enumerate(REST):rec_rows.append(dict(case_id=case['case_id'],split=split,variable=variables[k],NRMSE=float(e[j]) if np.isfinite(e[j]) else None,diagnostic='Given true representatives; not an autoregressive NN rollout'))
        for name,definition in selection['models'].items():
            if not primary and definition['role']=='final':continue
            model=load_operator(root/definition['path']);target=root/'evaluations'/split/name;target.mkdir(parents=True)
            active_reconstruction=reconstruction
            if definition.get('reconstruction_path'):
                active_reconstruction=ReconstructionNet();active_reconstruction.load_state_dict(torch.load(root/definition['reconstruction_path'],weights_only=False)['model']);active_reconstruction.eval()
            for i,(truth,case) in enumerate(zip(states,group)):
                if time.perf_counter()>deadline:raise TimeoutError('Evaluation guard reached; no completion marker')
                traces=dict(reference=truth);modes=['raw_full','representative_unconstrained']+(['repaired_full'] if primary and definition['role']!='final' else [])
                for mode in modes:
                    rep=mode=='representative_unconstrained';columns=REP if rep else np.arange(11);active=np.arange(5) if rep else ACTIVE
                    if rep:
                        pred,physical=mathematical_representative(model,scale.subset(REP),truth[0,REP],case['phi']);info=dict(complete=len(pred)==len(truth),points=len(pred),reason='unconstrained_curve_diagnostic')
                    else:
                        pred,info=rollout(model,active_reconstruction,truth[0],case['phi'],scale,mode,root/'mechanism.yaml');physical=info['complete']
                    traces[mode]=pred;ref=truth[:,columns];sc=profile_scores(ref,pred,active);full,early=sc['full'],sc['early'];trace_path=str((target/f'IC{case["case_id"]}.npz').relative_to(root))
                    rows.append(dict(**case,model=name,role=definition['role'],mode=mode,**info,physical_valid=physical,joint='joint5' if rep else 'joint9',curve_accuracy_pass=sc['pass_full'],joint_pass=bool(sc['pass_full'] and physical),joint_full_and_early=bool(sc['pass_both'] and physical),worst_NRMSE=float(np.max(full[active])) if full is not None else None,early_worst_NRMSE=float(np.max(early[active])) if early is not None else None,negative_points=int(np.sum(np.any(pred[:,1:]<0,axis=1))),minimum_species=float(pred[:,1:].min()),traces_file=trace_path))
                    for j,k in enumerate(columns):metrics.append(dict(case_id=case['case_id'],split=split,model=name,role=definition['role'],mode=mode,variable=variables[k],complete=sc['complete'],NRMSE=float(full[j]) if full is not None and np.isfinite(full[j]) else None,early_0p1ms_NRMSE=float(early[j]) if early is not None and np.isfinite(early[j]) else None,reference_RMS=float(np.sqrt(np.mean(ref[:,j]**2)))))
                np.savez_compressed(target/f'IC{case["case_id"]}.npz',**traces)
                write(root/'status.json',dict(phase='evaluating',split=split,model=name,completed_cases=i+1,total_cases=len(group)))
    csv_write(root/'evaluations/rollouts.csv',rows);csv_write(root/'evaluations/NRMSE.csv',metrics);csv_write(root/'evaluations/reconstruction_given_truth.csv',rec_rows)


def verify(root):
    inputs=json.loads((root/'input_manifest.json').read_text());sources=json.loads((root/'source_manifest.json').read_text())
    assert all(sha(root/p)==h for p,h in inputs.items())
    assert all(sha(p)==h for p,h in json.loads((root/'historical_input_manifest.json').read_text()).items())
    assert all(sha(root/'source_snapshot'/p)==h for p,h in sources.items())
    assert all(sha(p)==h for p,h in ORIGINALS.items())
    selection=json.loads((root/'selection_frozen.json').read_text());access=json.loads((root/'test_access_started.json').read_text())
    assert access['selection_sha256']==sha(root/'selection_frozen.json') and access['at']>=selection['frozen_at']
    assert all(sha(root/d['path'])==d['sha256'] for d in selection['models'].values())
    assert all(sha(root/d['reconstruction_path'])==d['reconstruction_sha256'] for d in selection['models'].values() if d.get('reconstruction_path'))
    cases=json.loads((root/'cases.json').read_text());assert len({(c['T0'],c['phi']) for c in cases})==len(cases)
    rows=list(csv.DictReader((root/'evaluations/rollouts.csv').open()));metrics=list(csv.DictReader((root/'evaluations/NRMSE.csv').open()));variables=json.loads((root/'dataset.json').read_text())['variables'];full_checks=0;early_checks=0
    grouped={}
    for r in metrics:grouped.setdefault((r['split'],r['model'],r['case_id'],r['mode']),[]).append(r)
    for r in rows:
        with np.load(root/r['traces_file']) as a:
            p=a[r['mode']];columns=REP if r['mode']=='representative_unconstrained' else np.arange(11);ref=a['reference'][:,columns];active=np.arange(5) if len(columns)==5 else ACTIVE;sc=profile_scores(ref,p,active)
        physical=bool(sc['complete'] and np.isfinite(p).all() and np.all((p[:,0]>=200)&(p[:,0]<=6000)) and p[:,1:].min()>=0)
        assert (r['complete']=='True')==sc['complete'] and (r['physical_valid']=='True')==physical
        assert (r['curve_accuracy_pass']=='True')==sc['pass_full']
        assert (r['joint_pass']=='True')==(sc['pass_full'] and physical)
        assert (r['joint_full_and_early']=='True')==(sc['pass_both'] and physical)
        names=[variables[k] for k in columns]
        for m in grouped[r['split'],r['model'],r['case_id'],r['mode']]:
            j=names.index(m['variable'])
            for key,values in [('NRMSE',sc['full']),('early_0p1ms_NRMSE',sc['early'])]:
                if values is None or not np.isfinite(values[j]):assert m[key]==''
                else:
                    assert np.isclose(float(m[key]),values[j],rtol=1e-11,atol=1e-13)
                    if key=='NRMSE':full_checks+=1
                    else:early_checks+=1
    frozen=[]
    for p in root.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.tmp','.lock','.pyc']:continue
        rel=p.relative_to(root)
        if rel.parts[0]=='report' or str(rel) in ['status.json','worker_started.json','verification.json','complete.json','exit.json']:continue
        frozen.append(p)
    write(root/'verification.json',dict(status='passed',full_NRMSE_recomputations=full_checks,early_NRMSE_recomputations=early_checks,no_full_score_for_incomplete=True,eligibility_recomputed=True,original_data_unchanged=True,source_snapshot_unchanged=True,disjoint_ICs=True,selection_frozen_before_new_tests=True,output_sha256={str(p.relative_to(root)):sha(p) for p in frozen}))


def execute(root,require_publication=True):
    lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'worker_started.json').exists():raise RuntimeError('No duplicate run or renewed budget')
    cfg=json.loads((root/'config.json').read_text());prepared=json.loads((root/'prepared.json').read_text())
    assert prepared['status']=='passed' and prepared['config_sha256']==sha(root/'config.json') and prepared['source_manifest_sha256']==sha(root/'source_manifest.json') and prepared['input_manifest_sha256']==sha(root/'input_manifest.json') and prepared['dataset_sha256']==sha(root/'dataset.json')
    assert all(sha(root/p)==h for p,h in json.loads((root/'input_manifest.json').read_text()).items())
    assert all(sha(root/'source_snapshot'/p)==h for p,h in json.loads((root/'source_manifest.json').read_text()).items())
    assert all(sha(p)==h for p,h in ORIGINALS.items())
    if require_publication:
        pub=json.loads((AUDIT/'publication.json').read_text());assert pub['status']=='passed' and pub['remote_verified'] and pub['branch']=='bw' and pub['source_manifest_sha256']==sha(root/'source_manifest.json') and pub['prepared_sha256']==sha(root/'prepared.json')
    torch.set_num_threads(cfg['threads']);torch.set_num_interop_threads(1);started=time.perf_counter();deadline=started+cfg['budget_seconds']-prepared['prior_seconds']-prepared['prepare_seconds'];train_deadline=deadline-cfg['reserve_evaluation_seconds']
    write(root/'worker_started.json',dict(pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),remaining_seconds=deadline-started))
    try:
        all_cases=json.loads((root/'cases.json').read_text());train_cases=[c for c in all_cases if c['split']=='train'];val_cases=[c for c in all_cases if c['split']=='val'];train=np.load(root/'data/train.npz')['states'];val=np.load(root/'data/val.npz')['states'];prep=json.loads((root/'preprocessing.json').read_text());scale=FifthRootScale(prep['minimum'],prep['maximum'])
        encoded=torch.from_numpy(scale.subset(REP).forward(train[:,:,REP]));phis=torch.tensor([c['phi'] for c in train_cases],dtype=torch.float32)
        reconstruction=ReconstructionNet();reconstruction.load_state_dict(torch.load(root/'inputs/reconstruction.pt',weights_only=False)['model']);reconstruction.eval()
        probe_results={}
        for spec in cfg['candidates']:
            if time.perf_counter()>train_deadline:raise TimeoutError('Budget exhausted during probe stage')
            probe_results[spec['name']]=train_candidate(root,cfg,spec,encoded,phis,val,val_cases,scale,reconstruction,cfg['probe_updates'],train_deadline,'probing')
            write(root/'probe_results.json',probe_results)
        finalists=[]
        for family in ['endpoint','all_steps']:
            choices=[s for s in cfg['candidates'] if s['family']==family and not probe_results[s['name']]['stop_reason'].startswith('nonfinite_')]
            if not choices:raise RuntimeError(f'All candidates failed numerically in {family}; no automatic retraining')
            chosen=min(choices,key=lambda s:(tuple(probe_results[s['name']]['latest']['rank']),cfg['probe_tie_preference'].index(s['name'])))
            finalists.append(chosen)
        write(root/'promotion_frozen.json',dict(status='passed',chosen=finalists,uses_validation_only=True,test_used=False,probe_updates=cfg['probe_updates'],exact_probe_states_will_continue=True))
        allocation=max(0.,(train_deadline-time.perf_counter()-cfg['reconstruction_reserve_seconds'])/len(finalists));final_results={}
        for spec in finalists:
            end=min(train_deadline,time.perf_counter()+allocation)
            final_results[spec['family']]=train_candidate(root,cfg,spec,encoded,phis,val,val_cases,scale,reconstruction,cfg['target_additional_updates'],end,'continuing')
            write(root/'final_training.json',final_results)
        reconstruction_results=train_reconstruction_controls(root,cfg,train,val,scale,min(train_deadline,time.perf_counter()+cfg['reconstruction_reserve_seconds']))
        models={name:dict(path='inputs/'+name+'.pt',role='baseline',description=filename) for name,filename in cfg['baselines'].items()}
        for spec in finalists:
            for kind,filename in [('selected','selected.pt'),('final','last.pt')]:
                models[spec['family']+'_'+kind]=dict(path=f'models/{spec["name"]}/{filename}',role=kind,spec=spec,selected_new_update=final_results[spec['family']]['selected_new_update'],last_new_update=final_results[spec['family']]['last_new_update'])
        combinations=[]
        for op in ['best_pretrain','last_pretrain','endpoint_selected','all_steps_selected']:
            operator=load_operator(root/models[op]['path'])
            for rec_name,rec_path in [('original','inputs/reconstruction.pt')]+[(n,d['path']) for n,d in reconstruction_results.items()]:
                rec=ReconstructionNet();rec.load_state_dict(torch.load(root/rec_path,weights_only=False)['model']);rec.eval();rank,metric=validation_metrics(operator,rec,val,val_cases,scale,root/'mechanism.yaml')
                combinations.append(dict(operator=op,reconstruction=rec_name,reconstruction_path=rec_path,rank=rank,validation=metric))
        combined=min(combinations,key=lambda d:tuple(d['rank']))
        models['combined_selected']=dict(path=models[combined['operator']]['path'],role='combined',operator=combined['operator'],reconstruction_path=combined['reconstruction_path'],reconstruction_sha256=sha(root/combined['reconstruction_path']))
        for definition in models.values():definition['sha256']=sha(root/definition['path'])
        selection=dict(status='passed',frozen_at=datetime.now(timezone.utc).isoformat(),models=models,finalists=finalists,reconstruction_controls=reconstruction_results,validation_factorial=combinations,combined_choice=combined,test_used_for_selection=False,new_test_archives_not_yet_read=True,per_finalist_wall_allocation=allocation)
        write(root/'selection_frozen.json',selection)
        evaluate(root,cfg,selection,all_cases,scale,reconstruction,deadline);verify(root)
        from .react_refinement_report import report
        report(root,deadline)
        r=json.loads((root/'report/complete.json').read_text());assert r['status']=='passed' and all(sha(root/'report'/p)==h for p,h in r['output_sha256'].items())
        write(root/'complete.json',dict(status='passed',wall_seconds=time.perf_counter()-started,finalists={k:{n:v[n] for n in ['last_new_update','selected_new_update','stop_reason','active_seconds','last_lr']} for k,v in final_results.items()},verification_sha256=sha(root/'verification.json'),report_complete_sha256=sha(root/'report/complete.json')))
        write(root/'status.json',dict(phase='complete',status='passed'));write(root/'exit.json',dict(returncode=0))
    except BaseException as exc:
        write(root/'failure.json',dict(status='failed',error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));write(root/'status.json',dict(phase='failed',error=str(exc)));write(root/'exit.json',dict(returncode=1));raise
    finally:lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','run'],required=True);p.add_argument('--root',type=Path,default=Path('results/react_refinement/run_001'));p.add_argument('--config',type=Path,default=Path('configs/react_refinement.json'));a=p.parse_args()
    if a.phase=='prepare':prepare(a.root.resolve(),a.config)
    else:execute(a.root.resolve())


if __name__=='__main__':main()
