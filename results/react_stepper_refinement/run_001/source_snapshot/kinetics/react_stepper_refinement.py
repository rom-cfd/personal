"""Bounded continuation of the self-fed 20us React-DeepONet baseline.

Old runs, data, normalization and models remain immutable. This is a declared
stabilization experiment, not an exact replay of unavailable author code.
Test arrays remain closed until validation-only factorial selection is frozen.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/react-stepper-refinement-mpl')
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,json,shutil,time,traceback,fcntl
import numpy as np
import torch
import cantera as ct
from .react_deeponet_experiment import FifthRootScale,ORIGINALS
from .react_deeponet_v2_model import DetailedReactDeepONet
from .react_deeponet_reconstruction import ReconstructionNet
from .react_stepper_reproduction import feedback,score,state_valid,sha,write,csv_write,REP,REST,ACTIVE

BASE=Path('results/react_deeponet_rebuild/run_001')
VALIDATION=Path('results/react_refinement/run_001')
FOLLOW=Path('/workspace/react_stepper_refinement_followup/run_001')
SPECS=[dict(name=f'{mode}_lr{label}',mode=mode,lr=lr) for mode in ['uniform','fixed20']
       for label,lr in [('1e5',1e-5),('3e6',3e-6)]]
NEW_SOURCES=['kinetics/react_stepper_refinement.py','kinetics/react_stepper_refinement_report.py',
 'scripts/monitor_react_stepper_refinement.py','scripts/supervise_react_stepper_refinement.py',
 'tests/test_react_stepper_refinement.py']
DEPENDENCIES=['kinetics/react_stepper_reproduction.py','kinetics/react_deeponet_experiment.py',
 'kinetics/react_deeponet_v2_model.py','kinetics/react_deeponet_diagnostics.py',
 'kinetics/react_deeponet_reconstruction.py','kinetics/physics.py','kinetics/data.py']

def curriculum(update,stage=200):
    return 1 if update<=stage else 2 if update<=2*stage else 5

def sample_operator(states,phis,batch,generator,mode,k):
    i=torch.randint(len(states),(batch,),generator=generator)
    origin=torch.randint(states.shape[1]-400*k,(batch,),generator=generator)
    drawn=torch.randint(1,401,(batch,),generator=generator)
    offset=drawn if mode=='uniform' else torch.full_like(drawn,200)
    return states[i,origin],offset[:,None].float()*1e-7,phis[i,None],states[i,origin+k*offset]

def endpoint_loss(model,state,tau,phi,target,k):
    for _ in range(k):state=model(state,tau,phi)
    return (state-target).square().mean()

def training_losses(model,compile_enabled):
    result={}
    for k in [1,2,5]:
        def loss_function(x,t,p,y,k=k):return endpoint_loss(model,x,t,p,y,k)
        result[k]=torch.compile(loss_function,fullgraph=True,dynamic=False) if compile_enabled else loss_function
    return result

def sample_reconstruction(states,batch,generator,early):
    i=torch.randint(len(states),(batch,),generator=generator)
    full=torch.randint(states.shape[1],(batch,),generator=generator)
    focused=torch.randint(min(1001,states.shape[1]),(batch,),generator=generator)
    mix=torch.rand(batch,generator=generator)<.5
    t=torch.where(mix,focused,full) if early else full
    return states[i,t]

def reference(case,mechanism):
    gas=ct.Solution(str(mechanism));gas.TP=case['T0'],ct.one_atm
    gas.set_equivalence_ratio(case['phi'],'H2','O2:1,N2:3.76')
    reactor=ct.IdealGasConstPressureReactor(gas,clone=False);net=ct.ReactorNet([reactor])
    net.rtol=1e-10;net.atol=1e-20;net.max_time_step=1e-7
    a=np.empty((10001,11));a[0]=np.r_[gas.T,gas.Y]
    for j in range(1,10001):net.advance(j*1e-7);a[j]=np.r_[gas.T,gas.Y]
    assert a[:,1:].min()>=-1e-14
    a[:,1:]=np.maximum(a[:,1:],0) # Reference roundoff only, never model predictions.
    return a

def holdout_cases(fixture=False):
    known=[]
    for filename in [BASE/'cases.json',VALIDATION/'cases.json',
        Path('results/react_stepper_reproduction/run_001/cases_frozen.json'),
        FOLLOW/'fixture_001/cases.json']:
        if filename.exists():
            obj=json.loads(filename.read_text())
            if isinstance(obj,list):known.extend(obj)
    used={(round(c['T0'],8),round(c['phi'],8)) for c in known if 'T0' in c and 'phi' in c}
    rng=np.random.default_rng(2026111701 if fixture else 20261117);n=1 if fixture else 32;cases=[]
    for split in ['interpolation','extrapolation']:
        for j in range(n):
            while True:
                t=float(rng.uniform(980,1240));p=float(rng.uniform(.6,1.3))
                inside=1000<=t<=1200 and .75<=p<=1.15
                if inside!=(split=='interpolation'):continue
                key=(round(t,8),round(p,8))
                if key not in used:break
            used.add(key);cases.append(dict(IC=len(cases),split=split,T0=t,phi=p))
    return cases

def configuration(fixture=False):
    return dict(seed=11,threads=2,fixture=fixture,total_budget_seconds=180 if fixture else 21600,
      evaluation_reserve_seconds=30 if fixture else 1800,reconstruction_reserve_seconds=15 if fixture else 600,
      pilot_budget_seconds=40 if fixture else 1800,batch_size=32 if fixture else 20000,
      probe_updates=6 if fixture else 600,curriculum_stage_updates=2 if fixture else 200,
      target_new_updates=12 if fixture else 20000,validation_every=6 if fixture else 1000,
      history_every=1 if fixture else 50,compile_operator=not fixture,
      reconstruction_updates=4 if fixture else 20000,reconstruction_batch=32 if fixture else 4096,
      reconstruction_validation_every=4 if fixture else 1000,reconstruction_lr=1e-4,
      specs=SPECS,primary_step_us=20,secondary_step_us=2,train_reference_dt=1e-7,
      training_change='Same endpoint MSE and full BPTT, curriculum K1->K2->K5; compare fixed20us with positive uniform0.1..40us. No tau0 samples.',
      reconstruction_change='Same64x64 tanh ANN and MSE; compare uniform time sampling with half uniform/half initial0.1ms.',
      sources='Kumar-Echekki Fuel365(2024)131212 and Kumar thesis chapter3; declared stabilization changes, not exact author replay',
      limits=['Public Burke10species25Cantera entries differs from unavailable author20-reaction file.',
              'Three-way branch/trunk/ParamNet product remains an independent reconstruction assumption.',
              'No soft physics loss, training repair, inference clipping, element projection or enthalpy repair.'])

def prepare(root,fixture=False):
    if root.exists():raise FileExistsError(root)
    if fixture:assert str(root.resolve()).startswith(str(FOLLOW))
    started=time.perf_counter();root.mkdir(parents=True)
    for directory in ['inputs','data','models','evaluations','report','preflight','source_snapshot']:
        (root/directory).mkdir()
    cfg=configuration(fixture);write(root/'config.json',cfg)
    assert all(sha(p)==h for p,h in ORIGINALS.items())
    assert json.loads((BASE/'complete.json').read_text())['status']=='passed'
    for src,name in [(BASE/'models/pretrain_072500.pt','operator.pt'),
        (BASE/'models/reconstruction_selected.pt','reconstruction.pt')]:
        shutil.copyfile(src,root/'inputs'/name)
    for name in ['mechanism.yaml','preprocessing.json']:shutil.copyfile(BASE/name,root/name)
    for src,name in [(BASE/'data/train.npz','train.npz'),(VALIDATION/'data/val.npz','val.npz')]:
        if fixture:
            a=np.load(src)['states'];a=a[:2] if name=='train.npz' else a[[0,12]]
            np.savez_compressed(root/'data'/name,states=a)
        else:shutil.copyfile(src,root/'data'/name)
    all_cases=json.loads((VALIDATION/'cases.json').read_text())
    train=[c for c in all_cases if c['split']=='train'];val=[c for c in all_cases if c['split']=='val']
    if fixture:train=train[:2];val=[val[0],val[12]]
    write(root/'training_cases.json',train);write(root/'validation_cases.json',val)
    cases=holdout_cases(fixture);write(root/'cases.json',cases)
    # Selection cannot open these independent arrays before freezing its result.
    for split in ['interpolation','extrapolation']:
        array=np.stack([reference(c,root/'mechanism.yaml') for c in cases if c['split']==split])
        np.savez_compressed(root/'data'/f'test_{split}.npz',states=array)
    manifest={}
    for name in NEW_SOURCES+DEPENDENCIES:
        dest=root/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(name,dest);manifest[name]=sha(dest)
    (root/'source_snapshot/kinetics/__init__.py').write_text('"""Frozen20us feedback refinement."""\n')
    write(root/'source_manifest.json',manifest)
    if not fixture:
        for name in ['preflight.json','unit_tests.log']:
            if (FOLLOW/name).exists():shutil.copyfile(FOLLOW/name,root/'preflight'/name)
    inputs={str(p.relative_to(root)):sha(p) for p in [root/'config.json',root/'cases.json',
        root/'training_cases.json',root/'validation_cases.json',root/'preprocessing.json',root/'mechanism.yaml',
        *sorted((root/'inputs').iterdir()),*sorted((root/'data').iterdir())]}
    write(root/'input_manifest.json',inputs)
    prior=json.loads((FOLLOW/'preflight.json').read_text())['elapsed_seconds'] if not fixture and (FOLLOW/'preflight.json').exists() else 0
    write(root/'prepared.json',dict(status='passed',source_manifest_sha256=sha(root/'source_manifest.json'),
        input_manifest_sha256=sha(root/'input_manifest.json'),prepare_seconds=time.perf_counter()-started,
        prior_compute_seconds=prior,originals=ORIGINALS,old_root_complete_sha256=sha(BASE/'complete.json')))
    write(root/'status.json',dict(phase='prepared',training_started=False))
    print(json.dumps(dict(prepared=True,root=str(root),test_cases=len(cases),charged_seconds=prior+time.perf_counter()-started)))

def load_operator(path):
    m=DetailedReactDeepONet();m.load_state_dict(torch.load(path,weights_only=False)['model']);m.eval();return m

def load_reconstruction(path):
    m=ReconstructionNet();m.load_state_dict(torch.load(path,weights_only=False)['model']);m.eval();return m

def validation_metrics(operator,reconstruction,truth,cases,scale):
    operator.eval();reconstruction.eval()
    pred,info=feedback(operator,reconstruction,scale,truth[:,0],[c['phi'] for c in cases],200,10000)
    domains={d:dict(total=0,rep_both=0,full_both=0,full=0,complete=0) for d in sorted({c['domain'] for c in cases})}
    worst5=[];worst9=[];dense_positive=0
    for i,c in enumerate(cases):
        _,_,a=score(truth[i],pred[i]);_,_,b=score(truth[i,:1001],pred[i,:1001])
        rep_ok=bool(info['representative_boundary_valid'][i]);full_ok=bool(info['raw_boundary_valid'][i])
        rep_both=rep_ok and bool(np.all(a[REP]<=.1) and np.all(b[REP]<=.1))
        full_both=full_ok and bool(np.all(a[ACTIVE]<=.1) and np.all(b[ACTIVE]<=.1))
        d=domains[c['domain']];d['total']+=1;d['rep_both']+=int(rep_both);d['full_both']+=int(full_both)
        d['full']+=int(full_ok and np.all(a[ACTIVE]<=.1));d['complete']+=int(full_ok)
        worst5.append(float(max(np.max(a[REP]),np.max(b[REP]))));worst9.append(float(max(np.max(a[ACTIVE]),np.max(b[ACTIVE]))))
        dense_positive+=int(state_valid(pred[i]).all())
    operator.train();reconstruction.eval()
    safe=lambda a:float(np.median(np.nan_to_num(a,nan=1e9,posinf=1e9,neginf=1e9)))
    metric=dict(domains=domains,rep_both=sum(d['rep_both'] for d in domains.values()),full_both=sum(d['full_both'] for d in domains.values()),
        full=sum(d['full'] for d in domains.values()),complete=sum(d['complete'] for d in domains.values()),
        balanced_rep=min(d['rep_both']/d['total'] for d in domains.values()),
        balanced_full=min(d['full_both']/d['total'] for d in domains.values()),median_worst5=safe(worst5),median_worst9=safe(worst9),dense_positive=dense_positive)
    return metric

def operator_rank(m):
    return (-m['balanced_rep'],-m['rep_both'],-m['balanced_full'],-m['full_both'],-m['complete'],m['median_worst5'])

def combined_rank(m):
    return (-m['balanced_full'],-m['full_both'],-m['full'],-m['balanced_rep'],-m['rep_both'],-m['complete'],m['median_worst9'])

def train_operator(root,cfg,spec,encoded,phis,val,cases,scale,cnet,target,deadline,phase):
    folder=root/'models'/spec['name'];folder.mkdir(exist_ok=True);last=folder/'last.pt'
    saved=torch.load(last if last.exists() else root/'inputs/operator.pt',weights_only=False)
    model=DetailedReactDeepONet();model.load_state_dict(saved['model'])
    opt=torch.optim.Adamax(model.parameters());opt.load_state_dict(saved['optimizer'])
    generator=torch.Generator();generator.set_state(saved['generator_state'])
    if 'torch_rng' in saved:torch.set_rng_state(saved['torch_rng'])
    n=int(saved.get('new_update',0));active=float(saved.get('active_seconds',0));history=[]
    if (folder/'training.csv').exists():history=list(csv.DictReader((folder/'training.csv').open()))
    if (folder/'selection.json').exists():
        selection=json.loads((folder/'selection.json').read_text());best=tuple(selection['best_rank']);best_update=selection['selected_new_update'];best_k=selection['selected_K'];validations=selection['validations']
    else:
        metric=validation_metrics(model,cnet,val,cases,scale);best=operator_rank(metric);best_update=0;best_k=1
        validations=[dict(new_update=0,K=1,**metric)];shutil.copyfile(root/'inputs/operator.pt',folder/'selected.pt')
    compiled=training_losses(model,cfg['compile_operator'])
    reason='probe_updates_completed' if phase=='probing' else 'requested_updates_completed';start=time.perf_counter();last_loss=None;latest=validations[-1]
    write(root/'status.json',dict(phase=phase,candidate=spec['name'],new_update=n,target=target,K=curriculum(max(1,n),cfg['curriculum_stage_updates'])))
    def checkpoint(k):
        nonlocal best,best_update,best_k,latest
        metric=validation_metrics(model,cnet,val,cases,scale);latest=dict(new_update=n,K=k,**metric);validations.append(latest)
        payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),generator_state=generator.get_state(),torch_rng=torch.get_rng_state(),
            base_global_update=72500,new_update=n,active_seconds=active,spec=spec,K=k)
        torch.save(payload,last)
        rank=operator_rank(metric)
        if rank<best:best=rank;best_update=n;best_k=k;torch.save(payload,folder/'selected.pt')
        write(folder/'selection.json',dict(spec=spec,best_rank=best,selected_new_update=best_update,selected_K=best_k,
            last_new_update=n,final_K=k,active_seconds=active,last_lr=opt.param_groups[0]['lr'],stop_reason=reason,validations=validations,latest=latest))
        if history:csv_write(folder/'training.csv',history)
    while n<target:
        if time.perf_counter()>=deadline:reason='overall_time_cap';break
        attempted=n+1;k=curriculum(attempted,cfg['curriculum_stage_updates']);lr=spec['lr']*.9**((attempted-1)/5000)
        opt.param_groups[0]['lr']=lr;tick=time.perf_counter()
        x,t,p,y=sample_operator(encoded,phis,cfg['batch_size'],generator,spec['mode'],k)
        opt.zero_grad(set_to_none=True);loss=compiled[k](x,t,p,y)
        if not torch.isfinite(loss):active+=time.perf_counter()-tick;reason='nonfinite_loss';break
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):active+=time.perf_counter()-tick;reason='nonfinite_gradient';break
        opt.step();n=attempted;active+=time.perf_counter()-tick;last_loss=float(loss.detach())
        if n==1 or n%cfg['history_every']==0:
            history.append(dict(candidate=spec['name'],phase=phase,new_update=n,global_update=72500+n,K=k,loss=last_loss,lr=lr,active_seconds=active))
        boundary=n in [cfg['curriculum_stage_updates'],2*cfg['curriculum_stage_updates'],cfg['probe_updates']]
        if boundary or n%cfg['validation_every']==0 or n==target:checkpoint(k)
        if n==1 or n%100==0:write(root/'status.json',dict(phase=phase,candidate=spec['name'],new_update=n,target=target,K=k,lr=lr,active_seconds=active,latest_validation=latest))
    k=curriculum(max(1,n),cfg['curriculum_stage_updates']);checkpoint(k)
    result=json.loads((folder/'selection.json').read_text());result.update(stop_reason=reason,stage_wall_seconds=time.perf_counter()-start)
    write(folder/'selection.json',result)
    if reason.startswith('nonfinite'):write(folder/'numerical_failure.json',dict(reason=reason,attempted_new_update=n+1))
    return result

def reconstruction_rank(model,encoded_val,truth,cases,scale):
    model.eval()
    with torch.inference_mode():z=model(encoded_val[:,:,REP].reshape(-1,5));loss=float((z-encoded_val[:,:,REST].reshape(-1,6)).square().mean())
    pred=scale.subset(REST).inverse(z.numpy()).reshape(len(truth),10001,6)
    domains={d:dict(total=0,both=0) for d in sorted({c['domain'] for c in cases})};errors=[]
    for i,c in enumerate(cases):
        a=score(truth[i][:,REST],pred[i])[2][:4];b=score(truth[i,:1001][:,REST],pred[i,:1001])[2][:4]
        ok=bool(np.isfinite(pred[i]).all() and pred[i].min()>=0 and np.all(a<=.1) and np.all(b<=.1))
        domains[c['domain']]['total']+=1;domains[c['domain']]['both']+=int(ok);errors.append(max(float(np.max(a)),float(np.max(b))))
    balance=min(d['both']/d['total'] for d in domains.values());both=sum(d['both'] for d in domains.values());worst=float(np.median(errors))
    return (-balance,-both,worst,loss),dict(domains=domains,balanced_both=balance,teacher_forced_both=both,median_worst4=worst,normalized_MSE=loss)

def train_reconstructions(root,cfg,encoded,val,val_cases,scale,deadline):
    v=torch.from_numpy(scale.forward(val));results={};allocation=max(0.,(deadline-time.perf_counter())/2)
    for name,early in [('cnet_uniform',False),('cnet_early50',True)]:
        folder=root/'models'/name;folder.mkdir();saved=torch.load(root/'inputs/reconstruction.pt',weights_only=False)
        model=ReconstructionNet();model.load_state_dict(saved['model']);opt=torch.optim.Adam(model.parameters());opt.load_state_dict(saved['optimizer'])
        gen=torch.Generator();gen.set_state(saved['generator_state']);best,metric=reconstruction_rank(model,v,val,val_cases,scale)
        best_update=0;shutil.copyfile(root/'inputs/reconstruction.pt',folder/'selected.pt');history=[];active=0.;n=0
        end=min(deadline,time.perf_counter()+allocation);reason='requested_updates_completed'
        for attempted in range(1,cfg['reconstruction_updates']+1):
            if time.perf_counter()>=end:reason='overall_time_cap';break
            lr=cfg['reconstruction_lr']*.9**((attempted-1)/5000);opt.param_groups[0]['lr']=lr
            tick=time.perf_counter();batch=sample_reconstruction(encoded,cfg['reconstruction_batch'],gen,early)
            model.train();opt.zero_grad(set_to_none=True);loss=(model(batch[:,REP])-batch[:,REST]).square().mean()
            if not torch.isfinite(loss):reason='nonfinite_loss';break
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                active+=time.perf_counter()-tick;reason='nonfinite_gradient';break
            opt.step();n=attempted;active+=time.perf_counter()-tick
            if n==1 or n%100==0:history.append(dict(new_update=n,loss=float(loss.detach()),lr=lr,active_seconds=active))
            if n==1 or n%cfg['reconstruction_validation_every']==0 or n==cfg['reconstruction_updates']:
                rank,metric=reconstruction_rank(model,v,val,val_cases,scale)
                payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),generator_state=gen.get_state(),new_update=n,active_seconds=active)
                if rank<best:best=rank;best_update=n;torch.save(payload,folder/'selected.pt')
                csv_write(folder/'training.csv',history)
                write(root/'status.json',dict(phase='reconstruction',candidate=name,new_update=n,target=cfg['reconstruction_updates'],latest_validation=metric))
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator_state=gen.get_state(),new_update=n,active_seconds=active),folder/'last.pt')
        final_rank,final_metric=reconstruction_rank(model,v,val,val_cases,scale)
        results[name]=dict(selected_new_update=best_update,last_new_update=n,best_rank=best,final_validation=final_metric,
            stop_reason=reason,active_seconds=active,last_lr=opt.param_groups[0]['lr'])
        write(folder/'selection.json',results[name])
    write(root/'reconstruction_controls.json',results);return results

def check_publication(root,publication,cfg):
    if cfg['fixture']:
        assert str(root.resolve()).startswith(str(FOLLOW));return
    p=json.loads(Path(publication).read_text())
    assert p['status']=='passed' and p['remote_verified'] and p['branch']=='bw'
    assert len(p['commit_sha'])==40 and p['prepared_sha256']==sha(root/'prepared.json')
    assert p['source_manifest_sha256']==sha(root/'source_manifest.json')

def run(root,publication,deadline=None):
    if (root/'worker_started.json').exists():raise RuntimeError('Never restart this bounded run')
    cfg=json.loads((root/'config.json').read_text());prep=json.loads((root/'prepared.json').read_text())
    check_publication(root,publication,cfg)
    assert sha(root/'source_manifest.json')==prep['source_manifest_sha256']
    assert sha(root/'input_manifest.json')==prep['input_manifest_sha256']
    for name,h in json.loads((root/'source_manifest.json').read_text()).items():assert sha(root/'source_snapshot'/name)==h,name
    for name,h in json.loads((root/'input_manifest.json').read_text()).items():assert sha(root/name)==h,name
    assert all(sha(p)==h for p,h in ORIGINALS.items())
    lock=(root/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.perf_counter();remaining=cfg['total_budget_seconds']-prep['prepare_seconds']-prep['prior_compute_seconds']
    deadline=min(deadline or float('inf'),started+remaining)
    write(root/'worker_started.json',dict(pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),deadline_monotonic=deadline,remaining_seconds=remaining))
    torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    train=np.load(root/'data/train.npz')['states'];val=np.load(root/'data/val.npz')['states']
    tc=json.loads((root/'training_cases.json').read_text());vc=json.loads((root/'validation_cases.json').read_text())
    p=json.loads((root/'preprocessing.json').read_text());scale=FifthRootScale(p['minimum'],p['maximum'])
    encoded=torch.from_numpy(scale.forward(train));native=encoded[:,:,REP].contiguous();phis=torch.tensor([c['phi'] for c in tc])
    cnet=load_reconstruction(root/'inputs/reconstruction.pt')
    training_deadline=deadline-cfg['evaluation_reserve_seconds']-cfg['reconstruction_reserve_seconds']
    # Compile before allocating equal pilot slots so the first candidate does
    # not lose its slot to compilation. No optimizer step or RNG reuse here.
    if cfg['compile_operator']:
        warm_started=time.perf_counter();warm=load_operator(root/'inputs/operator.pt');warm.train()
        warm_generator=torch.Generator().manual_seed(1729);losses=training_losses(warm,True)
        for k in [1,2,5]:
            write(root/'status.json',dict(phase='compilation_warmup',K=k,training_updates=0))
            batch=sample_operator(native,phis,cfg['batch_size'],warm_generator,'uniform',k)
            warm.zero_grad(set_to_none=True);loss=losses[k](*batch);loss.backward()
            if time.perf_counter()>=training_deadline:raise TimeoutError('Compilation consumed training allocation')
        write(root/'compilation_warmup.json',dict(status='passed',seconds=time.perf_counter()-warm_started,optimizer_updates=0))
        del warm,losses
    probe_deadline=min(training_deadline,time.perf_counter()+cfg['pilot_budget_seconds']);probes={}
    for i,spec in enumerate(cfg['specs']):
        end=min(probe_deadline,time.perf_counter()+max(0.,(probe_deadline-time.perf_counter())/(len(cfg['specs'])-i)))
        probes[spec['name']]=train_operator(root,cfg,spec,native,phis,val,vc,scale,cnet,cfg['probe_updates'],end,'probing')
        write(root/'probe_results.json',probes)
    promoted=[]
    for mode in ['uniform','fixed20']:
        eligible=[s for s in cfg['specs'] if s['mode']==mode and probes[s['name']]['last_new_update']==cfg['probe_updates']
                  and not probes[s['name']]['stop_reason'].startswith('nonfinite')]
        if eligible:promoted.append(min(eligible,key=lambda s:operator_rank(probes[s['name']]['latest'])))
    write(root/'promotion_frozen.json',dict(status='passed',promoted=promoted,uses_validation_only=True,
        selection='literal last probe validation; continue exact last weights/Adam/RNG',probes=probes))
    final={}
    write(root/'final_training.json',final)
    for i,spec in enumerate(promoted):
        end=min(training_deadline,time.perf_counter()+max(0.,(training_deadline-time.perf_counter())/(len(promoted)-i)))
        final[spec['name']]=train_operator(root,cfg,spec,native,phis,val,vc,scale,cnet,cfg['target_new_updates'],end,'continuation')
        write(root/'final_training.json',final)
    recon=train_reconstructions(root,cfg,encoded,val,vc,scale,deadline-cfg['evaluation_reserve_seconds'])
    operators={'baseline':root/'inputs/operator.pt'}
    for spec in promoted:operators[spec['mode']+'_selected']=root/'models'/spec['name']/'selected.pt'
    reconstructions={'original':root/'inputs/reconstruction.pt',
        **{name:root/'models'/name/'selected.pt' for name in recon}}
    factorial=[]
    for oname,opath in operators.items():
        for cname,cpath in reconstructions.items():
            m=validation_metrics(load_operator(opath),load_reconstruction(cpath),val,vc,scale)
            factorial.append(dict(operator=oname,reconstruction=cname,rank=combined_rank(m),validation=m))
    choice=min(factorial,key=lambda r:tuple(r['rank']))
    models={'baseline':dict(operator='inputs/operator.pt',reconstruction='inputs/reconstruction.pt')}
    for spec in promoted:
        for role in ['selected','final']:
            models[spec['mode']+'_'+role]=dict(operator=f"models/{spec['name']}/"+('selected.pt' if role=='selected' else 'last.pt'),
                reconstruction='inputs/reconstruction.pt',training=final[spec['name']])
    models['cnet_only']=dict(operator='inputs/operator.pt',reconstruction=str(reconstructions[choice['reconstruction']].relative_to(root)))
    models['combined_selected']=dict(operator=str(operators[choice['operator']].relative_to(root)),
        reconstruction=str(reconstructions[choice['reconstruction']].relative_to(root)))
    for value in models.values():
        value['operator_sha256']=sha(root/value['operator']);value['reconstruction_sha256']=sha(root/value['reconstruction'])
    write(root/'selection_frozen.json',dict(status='passed',at=datetime.now(timezone.utc).isoformat(),
        combined_choice=choice,validation_factorial=factorial,models=models,test_used_for_selection=False,
        note='Update0 selection is an unchanged-baseline fallback, not a training improvement.'))
    write(root/'status.json',dict(phase='test_evaluation',selection_frozen=True))
    from .react_stepper_refinement_report import evaluate,report,verify
    evaluate(root,cfg,scale,models,deadline)
    report(root,cfg,deadline)
    verification=verify(root);write(root/'verification.json',verification);assert verification['status']=='passed'
    assert all(sha(p)==h for p,h in ORIGINALS.items())
    outputs={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and p.name not in ['status.json','complete.json','exit.json','worker.lock','worker_started.json']}
    write(root/'complete.json',dict(status='passed',at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
        verification_sha256=sha(root/'verification.json'),report_complete_sha256=sha(root/'report/complete.json'),
        selection_sha256=sha(root/'selection_frozen.json'),output_sha256=outputs))
    write(root/'status.json',dict(phase='complete',status='passed'));write(root/'exit.json',dict(returncode=0,complete_sha256=sha(root/'complete.json')))
    print(json.dumps(dict(complete=True,root=str(root),elapsed_seconds=time.perf_counter()-started)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--fixture',action='store_true');p.add_argument('--publication',type=Path,default=FOLLOW/'publication.json');p.add_argument('--deadline',type=float)
    a=p.parse_args()
    try:prepare(a.root,a.fixture) if a.action=='prepare' else run(a.root,a.publication,a.deadline)
    except Exception:
        if a.root.exists():write(a.root/'failure.json',dict(traceback=traceback.format_exc()));write(a.root/'exit.json',dict(returncode=1))
        raise

if __name__=='__main__':main()
