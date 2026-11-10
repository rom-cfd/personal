"""Detailed-method reconstruction after a documented failed first approximation.

Old sources/models/results are immutable. Training sees only train/validation
archives. A publication gate is required by the CLI before long training.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/react-rebuild-mpl')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
import argparse,csv,json,time,shutil,re,traceback,fcntl
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import cantera as ct
import torch
from .react_deeponet_experiment import FifthRootScale,write,csv_write,sha,scores,valid_state,ORIGINALS
from .react_deeponet_reconstruction import ReconstructionNet
from .react_deeponet_v2_model import DetailedReactDeepONet,BoundPhiOperator,recursive_endpoint,sample_batch
from .physics import project_species

REP=np.array([0,1,2,4,6]);REST=np.array([3,5,7,8,9,10])
EVALUATION_MODES=['representative_unconstrained','representative_raw','raw_full','repaired_full',
                  'representative_protocol_final_unconstrained','representative_protocol_final_raw']


def definitions(cfg):
    r=cfg['reported'];x=cfg['explicit_choices_and_limits'];cases=[];used=set()
    training={(t,p) for t in r['train_T0'] for p in r['train_phi']}
    grids=[('train',r['train_T0'],r['train_phi']),('val',x['validation_T0'],x['validation_phi']),
           ('paper_test',r['paper_grid_T0'],r['paper_grid_phi']),('fresh_test',x['fresh_test_T0'],x['fresh_test_phi'])]
    for split,temperatures,phis in grids:
        for t in temperatures:
            for p in phis:
                if split=='paper_test' and (t,p) in training:continue
                assert (t,p) not in used,'IC split overlap'
                used.add((t,p));cases.append(dict(case_id=len(cases),split=split,T0=float(t),phi=float(p)))
    return cases


def prepare(root,cfg_path):
    if root.exists():raise FileExistsError('Use a new output root')
    root.mkdir(parents=True);cfg=json.loads(Path(cfg_path).read_text());write(root/'config.json',cfg)
    for name in ['models','data','evaluations','report','source_snapshot/kinetics']:(root/name).mkdir(parents=True,exist_ok=True)
    sources=['react_deeponet_rebuild.py','react_deeponet_rebuild_report.py','react_deeponet_v2_model.py',
             'react_deeponet_diagnostics.py','react_deeponet_reconstruction.py','react_deeponet_experiment.py','physics.py','data.py']
    manifest={}
    for name in sources:
        p=Path('kinetics')/name;shutil.copyfile(p,root/'source_snapshot/kinetics'/name);manifest['kinetics/'+name]=sha(p)
    (root/'source_snapshot/kinetics/__init__.py').write_text('"""Frozen rebuild."""\n')
    (root/'source_snapshot/run.py').write_text('from kinetics.react_deeponet_rebuild import main\nmain()\n')
    shutil.copyfile('scripts/supervise_react_rebuild.py',root/'source_snapshot/supervise.py')
    for name in ['kinetics/__init__.py','run.py','supervise.py']:manifest[name]=sha(root/'source_snapshot'/name)
    write(root/'source_manifest.json',manifest)
    mechanism=Path(cfg['explicit_choices_and_limits']['mechanism']);shutil.copyfile(mechanism,root/'mechanism.yaml')
    assert {p:sha(p) for p in ORIGINALS}==ORIGINALS
    started=time.perf_counter();cases=definitions(cfg);write(root/'cases.json',cases)
    dt=cfg['reported']['reference_dt_s'];n=int(round(cfg['reported']['horizon_s']/dt))
    arrays={};dataset_hashes={};variables=None
    for split in ['train','val','paper_test','fresh_test']:
        rows=[]
        for case in [c for c in cases if c['split']==split]:
            if time.perf_counter()-started>300:raise TimeoutError('Reference-generation guard exceeded')
            gas=ct.Solution(str(root/'mechanism.yaml'));gas.TP=case['T0'],ct.one_atm;gas.set_equivalence_ratio(case['phi'],'H2','O2:1,N2:3.76')
            if variables is None:variables=['T']+gas.species_names
            reactor=ct.IdealGasConstPressureReactor(gas,clone=False);network=ct.ReactorNet([reactor]);network.rtol=1e-10;network.atol=1e-20;network.max_time_step=dt
            trace=np.empty((n+1,11));trace[0]=np.r_[gas.T,gas.Y]
            for j in range(1,n+1):network.advance(j*dt);trace[j]=np.r_[gas.T,gas.Y]
            if trace[:,1:].min() < -1e-14:raise ValueError('Negative reference species beyond roundoff')
            trace[:,1:]=np.maximum(trace[:,1:],0);rows.append(trace)
            write(root/'status.json',dict(phase='preparing',split=split,case_id=case['case_id'],total_cases=len(cases)))
        array=np.asarray(rows);np.savez_compressed(root/'data'/f'{split}.npz',states=array)
        dataset_hashes[split]=sha(root/'data'/f'{split}.npz')
        if split=='train':arrays['train']=array
    scale=FifthRootScale.fit(arrays['train']);write(root/'preprocessing.json',scale.json())
    meta=dict(variables=variables,dt=dt,horizon=cfg['reported']['horizon_s'],
        counts={s:sum(c['split']==s for c in cases) for s in ['train','val','paper_test','fresh_test']},
        data_sha256=dataset_hashes,mechanism_sha256=sha(root/'mechanism.yaml'),prepare_seconds=time.perf_counter()-started,
        mechanism_limit=cfg['explicit_choices_and_limits']['mechanism_limit'],cantera=ct.__version__)
    write(root/'dataset.json',meta)
    prior=[]
    for p in [Path('results')/s/'run_001/complete.json' for s in ['react_deeponet_diagnostics','react_deeponet_refine_probe','react_deeponet_single_ic','react_deeponet_sampling_probe']]:
        if p.exists():prior.append(dict(path=str(p),sha256=sha(p),seconds=json.loads(p.read_text())['elapsed_seconds']))
    preflight=Path('/workspace/react_rebuild_followup/run_001/preflight.json')
    if preflight.exists():
        shutil.copyfile(preflight,root/'software_preflight.json')
        prior.append(dict(path='software_preflight.json',sha256=sha(root/'software_preflight.json'),seconds=json.loads(preflight.read_text())['elapsed_seconds']))
    write(root/'prepared.json',dict(status='passed',config_sha256=sha(root/'config.json'),source_manifest_sha256=sha(root/'source_manifest.json'),
        dataset_metadata_sha256=sha(root/'dataset.json'),
        cases_sha256=sha(root/'cases.json'),preprocessing_sha256=sha(root/'preprocessing.json'),prior_diagnostics=prior,
        prior_diagnostic_seconds=sum(p['seconds'] for p in prior),original_data_sha256=ORIGINALS))
    write(root/'status.json',dict(phase='prepared',training_started=False))
    print('PREPARED',json.dumps(meta),flush=True)


def macro(model,scale,initial,phi,steps=50,dt=20e-6):
    trace=[initial.copy()]
    with torch.inference_mode():
        for _ in range(steps):
            z=torch.from_numpy(scale.forward(trace[-1][None,:]));tau=torch.tensor([[dt]],dtype=torch.float32);p=torch.tensor([[phi]],dtype=torch.float32)
            next_state=scale.inverse(model(z,tau,p).numpy())[0]
            if not valid_state(next_state):break
            trace.append(next_state)
    return np.asarray(trace)


def validation(model,val,val_cases,scale,fixed):
    model.eval()
    with torch.inference_mode():
        x,t,p,y=fixed;mse=float(torch.mean((model(x,t,p)-y)**2))
    if not np.isfinite(mse):raise FloatingPointError('Nonfinite validation loss')
    errors=[];passed=0;complete=0;physical=0
    for truth,case in zip(val,val_cases):
        reference=truth[:,REP];pred,valid=mathematical_representative(model,scale,reference[0],case['phi'])
        if len(pred)==len(reference):
            complete+=1;physical+=int(valid);e=float(np.max(scores(reference,pred)));errors.append(e);passed+=int(e<=.1)
    median=float(np.median(errors)) if errors else float('inf');model.train()
    return (-complete,-passed,median,mse),dict(window_MSE=mse,numerically_complete=complete,curve_joint5=passed,strict_physical=physical,total=len(val),median_worst5=median if errors else None)


def mathematical_representative(model,scale,initial,phi):
    """Paper-style normalized feedback, with physical violations exposed.

    Never clip or replace values. This diagnostic is not an admissible raw
    chemistry solver merely because its decoded curves agree with Cantera.
    """
    trace=[initial.copy()];z=scale.forward(initial[None,:]);valid=True
    tau=torch.arange(1,201,dtype=torch.float32)[:,None]*1e-7
    parameter=torch.full((200,1),float(phi),dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(50):
            out=model(torch.from_numpy(np.repeat(z,200,axis=0)),tau,parameter).numpy()
            if not np.isfinite(out).all():break
            physical=scale.inverse(out)
            if not np.isfinite(physical).all():break
            valid=valid and bool(np.all((physical[:,0]>=200)&(physical[:,0]<=6000)) and np.min(physical[:,1:])>=0)
            trace.extend(physical);z=out[-1:]
    return np.asarray(trace),bool(valid and len(trace)==10001)


def train_reconstruction(root,cfg,train,val,scale,deadline):
    e=cfg['execution'];torch.manual_seed(e['seed']+1000);model=ReconstructionNet();opt=torch.optim.Adam(model.parameters(),lr=e['reconstruction_lr'])
    gen=torch.Generator().manual_seed(e['seed']+1001);x=torch.from_numpy(scale.forward(train).reshape(-1,11));v=torch.from_numpy(scale.forward(val).reshape(-1,11))
    best=float('inf');best_file=None;active=0.;history=[];started=time.perf_counter()
    for i in range(1,e['reconstruction_updates']+1):
        if time.perf_counter()>deadline:raise TimeoutError('Budget exhausted during reconstruction training')
        tick=time.perf_counter();batch=x[torch.randint(len(x),(e['reconstruction_batch'],),generator=gen)];opt.zero_grad(set_to_none=True)
        loss=torch.mean((model(batch[:,REP])-batch[:,REST])**2)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reconstruction loss')
        loss.backward();opt.step();active+=time.perf_counter()-tick
        if i==1 or i%100==0:history.append(dict(update=i,loss=float(loss.detach()),lr=e['reconstruction_lr'],active_seconds=active))
        if i==1 or i%1000==0 or i==e['reconstruction_updates']:
            with torch.inference_mode():metric=float(torch.mean((model(v[:,REP])-v[:,REST])**2))
            if metric<best:
                best=metric;best_file=root/'models'/f'reconstruction_{i:06d}.pt'
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator_state=gen.get_state(),update=i,validation_MSE=metric),best_file)
            csv_write(root/'reconstruction_training.csv',history);write(root/'status.json',dict(phase='training_reconstruction',update=i,target=e['reconstruction_updates'],best_validation_MSE=best))
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator_state=gen.get_state(),update=e['reconstruction_updates']),root/'models/reconstruction_final.pt')
    checkpoint=torch.load(best_file,weights_only=False);model.load_state_dict(checkpoint['model']);model.eval();shutil.copyfile(best_file,root/'models/reconstruction_selected.pt')
    write(root/'reconstruction_selection.json',dict(update=checkpoint['update'],total_updates=e['reconstruction_updates'],validation_MSE=best,active_seconds=active,wall_seconds=time.perf_counter()-started))
    return model


def train_operator(root,cfg,train,val,cases,scale,deadline):
    e,r=cfg['execution'],cfg['reported'];torch.manual_seed(e['seed']);model=DetailedReactDeepONet();opt=torch.optim.Adamax(model.parameters(),lr=r['initial_lr'])
    # Keep the original module as the checkpoint and inference owner. Compilation
    # only optimizes its training forward/backward, with the same parameters.
    training_forward=torch.compile(model,fullgraph=True,dynamic=False) if e.get('compile_operator',False) else model
    generator=torch.Generator().manual_seed(e['seed']+7);vg=torch.Generator().manual_seed(e['seed']+8)
    x=torch.from_numpy(scale.forward(train[:,:,REP]));v=torch.from_numpy(scale.forward(val[:,:,REP]))
    train_cases=[c for c in cases if c['split']=='train'];val_cases=[c for c in cases if c['split']=='val']
    phi=torch.tensor([c['phi'] for c in train_cases],dtype=torch.float32);vphi=torch.tensor([c['phi'] for c in val_cases],dtype=torch.float32)
    fixed=sample_batch(v,vphi,8192,400,1,vg);best_rank=None;best_path=None;history=[];selections=[];active=0.;global_step=0;started=time.perf_counter();phase_bests={}
    stopped=False
    for phase,count,k in [('pretrain',r['pretrain_updates'],1),('refine',r['refine_updates'],r['refine_K'])]:
        phase_best=None
        for i in range(1,count+1):
            if time.perf_counter()>deadline:stopped=True;break
            global_step+=1;lr=r['initial_lr']*r['decay_rate']**((global_step-1)/r['decay_steps']);opt.param_groups[0]['lr']=lr
            tick=time.perf_counter();a,t,p,target=sample_batch(x,phi,r['batch_size'],400,k,generator)
            opt.zero_grad(set_to_none=True);pred=recursive_endpoint(training_forward,a,t,p,k);loss=torch.mean((pred-target)**2)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss in {phase} at {i}')
            loss.backward();opt.step();active+=time.perf_counter()-tick
            if global_step==1 or global_step%e['history_every']==0:history.append(dict(phase=phase,phase_update=i,global_update=global_step,K=k,loss=float(loss.detach()),lr=lr,active_seconds=active,wall_seconds=time.perf_counter()-started))
            if i==1 or i%e['checkpoint_every']==0 or i==count:
                rank,metric=validation(model,val,val_cases,scale,fixed)
                path=root/'models'/f'{phase}_{i:06d}.pt'
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator_state=generator.get_state(),torch_rng=torch.get_rng_state(),global_update=global_step,phase=phase,phase_update=i,validation=metric),path)
                selections.append(dict(phase=phase,phase_update=i,global_update=global_step,**metric))
                if best_rank is None or rank<best_rank:best_rank=rank;best_path=path
                if phase_best is None or rank<phase_best[0]:phase_best=(rank,path)
                phase_bests[phase]=str(phase_best[1].relative_to(root))
                write(root/'selection_progress.json',dict(best_checkpoint=str(best_path.relative_to(root)),validations=selections,phase_bests=phase_bests))
            if global_step==1 or global_step%100==0:
                csv_write(root/'training.csv',history);write(root/'status.json',dict(phase='training_'+phase,update=i,target=count,global_update=global_step,K=k,lr=lr,active_seconds=active,elapsed_seconds=time.perf_counter()-started,latest_validation=selections[-1] if selections else None))
        if stopped:break
    if best_path is None:raise RuntimeError('No checkpoint available before budget guard')
    # A capped phase may end between checkpoint boundaries. Preserve those exact
    # last weights/Adam/RNG states separately; never reset or restart the budget.
    if selections[-1]['global_update']!=global_step:
        path=root/'models'/f'{phase}_capped_{global_step:06d}.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator_state=generator.get_state(),global_update=global_step,phase=phase),path)
    # Also evaluate the literal last protocol weights, independently of selecting
    # the best validation checkpoint. Never hide a degraded refinement phase.
    final_file=root/'models/operator_protocol_final.pt'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator_state=generator.get_state(),torch_rng=torch.get_rng_state(),global_update=global_step,phase=phase),final_file)
    chosen=torch.load(best_path,weights_only=False);model.load_state_dict(chosen['model']);model.eval();shutil.copyfile(best_path,root/'models/operator_selected.pt')
    target=r['pretrain_updates']+r['refine_updates'];csv_write(root/'training.csv',history)
    write(root/'selection_frozen.json',dict(status='passed',selected_checkpoint=str(best_path.relative_to(root)),selected_global_update=chosen['global_update'],selected_phase=chosen['phase'],selected_validation=chosen['validation'],validations=selections,phase_bests=phase_bests,
        total_updates=global_step,target_updates=target,update_target_completed=global_step==target,stop_reason='requested_updates_completed' if global_step==target else 'overall_time_cap',parameters=sum(p.numel() for p in model.parameters()),
        active_seconds=active,wall_seconds=time.perf_counter()-started,final_lr=opt.param_groups[0]['lr'],test_used_for_selection=False,optimizer_continued_into_refinement=True,
        protocol_final_checkpoint=str(final_file.relative_to(root)),protocol_final_phase=phase,protocol_final_update=global_step))
    return model


def rollout(model,reconstruction,initial,phi,scale,mode,mechanism,horizon=.001,dt=1e-7):
    representative=mode=='representative_raw';current=initial.copy();trace=[current[REP].copy() if representative else current.copy()]
    rs,ss=scale.subset(REP),scale.subset(REST);gas=ct.Solution(str(mechanism));matrix=np.array([[gas.n_atoms(k,e)*gas.atomic_weight(e)/gas.molecular_weights[k] for k in range(gas.n_species)] for e in gas.element_names])
    tau=torch.arange(1,201,dtype=torch.float32)[:,None]*dt;condition=torch.full((200,1),float(phi),dtype=torch.float32);complete_windows=0;reason='complete'
    with torch.inference_mode():
        for _ in range(int(round(horizon/(200*dt)))):
            encoded=torch.from_numpy(np.repeat(rs.forward(current[REP][None,:]),200,axis=0));z=model(encoded,tau,condition).numpy();reps=rs.inverse(z)
            if representative:candidate=reps
            else:
                other=ss.inverse(reconstruction(torch.from_numpy(z)).numpy());candidate=np.empty((200,11));candidate[:,REP]=reps;candidate[:,REST]=other
                if mode=='repaired_full':
                    gas.TPY=current[0],ct.one_atm,current[1:];h=gas.enthalpy_mass
                    for j in range(200):
                        try:
                            y=project_species(np.clip(candidate[j,1:],0,1),current[1:],matrix);gas.HPY=h,ct.one_atm,y;candidate[j]=np.r_[gas.T,y]
                        except (ValueError,RuntimeError):candidate[j:]=np.nan;break
            for row in candidate:
                if not valid_state(row):reason='invalid_state_or_repair_failure';break
                trace.append(row.copy())
            if reason!='complete':break
            complete_windows+=1
            if representative:current[REP]=candidate[-1]
            else:current=candidate[-1].copy()
    return np.asarray(trace),dict(complete=len(trace)==int(round(horizon/dt))+1,points=len(trace),windows=complete_windows,reason=reason)


def evaluate(root,cfg,model,reconstruction,cases,scale,deadline):
    metadata=json.loads((root/'dataset.json').read_text());variables=metadata['variables'];active=np.array([0,1,2,3,4,5,6,7,8]);rows=[];metrics=[];reconstruction_diagnostics=[]
    final_model=DetailedReactDeepONet();final_model.load_state_dict(torch.load(root/'models/operator_protocol_final.pt',weights_only=False)['model']);final_model.eval()
    for split in ['val','paper_test','fresh_test','train']:
        split_cases=[c for c in cases if c['split']==split]
        with np.load(root/'data'/f'{split}.npz') as f:states=f['states']
        if split=='train':split_cases=[c for c in split_cases if c['T0']==1080 and c['phi']==1]
        lookup={c['case_id']:i for i,c in enumerate([c for c in cases if c['split']==split])}
        target=root/'evaluations'/split;target.mkdir()
        for j,case in enumerate(split_cases):
            if time.perf_counter()>deadline:raise TimeoutError('Evaluation deadline reached; no success marker written')
            truth=states[lookup[case['case_id']]];traces={'reference':truth}
            with torch.inference_mode():
                encoded=scale.subset(REP).forward(truth[:,REP]);decoded=scale.subset(REST).inverse(reconstruction(torch.from_numpy(encoded)).numpy())
            rec_errors=scores(truth[:,REST],decoded)
            for ri,c in enumerate(REST):reconstruction_diagnostics.append(dict(case_id=case['case_id'],split=split,variable=variables[c],diagnostic='reconstruction_given_Cantera_representatives_not_a_rollout',NRMSE=float(rec_errors[ri]) if np.isfinite(rec_errors[ri]) else None,negative_points=int(np.sum(decoded[:,ri]<0))))
            for mode in EVALUATION_MODES:
                is_rep=mode.startswith('representative_')
                selected_model=final_model if 'protocol_final' in mode else model
                if mode.endswith('_unconstrained'):
                    pred,physical_valid=mathematical_representative(selected_model,scale.subset(REP),truth[0,REP],case['phi'])
                    info=dict(complete=len(pred)==len(truth),points=len(pred),windows=(len(pred)-1)//200,reason='finite_unconstrained_diagnostic')
                else:
                    pred,info=rollout(selected_model,reconstruction,truth[0],case['phi'],scale,'representative_raw' if 'protocol_final' in mode else mode,root/'mechanism.yaml');physical_valid=info['complete']
                traces[mode]=pred;columns=REP if is_rep else np.arange(11);ref=truth[:,columns];selected=np.arange(5) if is_rep else active
                full=scores(ref,pred) if info['complete'] else None;early=scores(ref[:1001],pred[:1001]) if len(pred)>=1001 else None
                worst=float(np.max(full[selected])) if full is not None else None
                def signed_normalized(a):
                    s=scale.subset(columns);return 2*(np.sign(a)*np.abs(a)**.2-s.minimum)/s.span-1
                mae=float(np.mean(np.abs(signed_normalized(ref)-signed_normalized(pred)))/2) if info['complete'] else None
                rows.append(dict(**case,mode=mode,**info,joint='joint5' if is_rep else 'joint9',curve_accuracy_pass=bool(worst is not None and worst<=.1),physical_valid=physical_valid,
                    joint_pass=bool(worst is not None and worst<=.1 and physical_valid),worst_NRMSE=worst,negative_species_points=int(np.sum(np.any(pred[:,1:]<0,axis=1))),
                    minimum_predicted_species=float(np.min(pred[:,1:])),early_0p1ms_worst_NRMSE=float(np.max(early[selected])) if early is not None else None,paper_normalized_MAE_ratio=mae))
                for c,column in enumerate(columns):metrics.append(dict(case_id=case['case_id'],split=split,T0=case['T0'],phi=case['phi'],mode=mode,variable=variables[column],complete=info['complete'],NRMSE=float(full[c]) if full is not None and np.isfinite(full[c]) else None,early_0p1ms_NRMSE=float(early[c]) if early is not None and np.isfinite(early[c]) else None,reference_RMS=float(np.sqrt(np.mean(ref[:,c]**2)))))
            np.savez_compressed(target/f'IC{case["case_id"]}.npz',**traces)
            write(root/'status.json',dict(phase='evaluating',split=split,completed_cases=j+1,total_cases=len(split_cases)))
    csv_write(root/'evaluations/rollouts.csv',rows);csv_write(root/'evaluations/NRMSE.csv',metrics);csv_write(root/'evaluations/reconstruction_given_truth.csv',reconstruction_diagnostics)


def verify(root,cfg,cases):
    meta=json.loads((root/'dataset.json').read_text());manifest=json.loads((root/'source_manifest.json').read_text())
    assert all(sha(root/'source_snapshot'/p)==h for p,h in manifest.items())
    assert all(sha(root/'data'/f'{s}.npz')==h for s,h in meta['data_sha256'].items())
    assert {p:sha(p) for p in ORIGINALS}==ORIGINALS
    assert len({(c['T0'],c['phi']) for c in cases})==len(cases)
    checks=0;early_checks=0;variables=meta['variables']
    rows=list(csv.DictReader((root/'evaluations/NRMSE.csv').open()))
    rollouts={(r['case_id'],r['mode']):r for r in csv.DictReader((root/'evaluations/rollouts.csv').open())}
    for split in sorted({r['split'] for r in rows}):
        for case_id in sorted({r['case_id'] for r in rows if r['split']==split}):
            with np.load(root/'evaluations'/split/f'IC{case_id}.npz') as traces:
                for mode in EVALUATION_MODES:
                    pred=traces[mode];columns=REP if mode.startswith('representative_') else np.arange(11);ref=traces['reference'][:,columns];complete=len(pred)==len(ref)
                    if complete:values=scores(ref,pred)
                    early=scores(ref[:1001],pred[:1001]) if len(pred)>=1001 else None
                    evidence=rollouts[(case_id,mode)];selected=np.arange(5) if mode.startswith('representative_') else np.arange(9)
                    physical=bool(complete and np.isfinite(pred).all() and np.all((pred[:,0]>=200)&(pred[:,0]<=6000)) and np.min(pred[:,1:])>=0)
                    curve=bool(complete and np.max(values[selected])<=.1)
                    assert (evidence['complete']=='True')==complete
                    assert (evidence['physical_valid']=='True')==physical
                    assert (evidence['curve_accuracy_pass']=='True')==curve
                    assert (evidence['joint_pass']=='True')==(curve and physical)
                    for row in [r for r in rows if r['case_id']==case_id and r['mode']==mode]:
                        assert (row['complete']=='True')==complete
                        j=list(variables[c] for c in columns).index(row['variable'])
                        if early is not None and np.isfinite(early[j]):
                            assert np.isclose(float(row['early_0p1ms_NRMSE']),early[j],rtol=1e-11,atol=1e-13);early_checks+=1
                        else:assert not row['early_0p1ms_NRMSE']
                        if not complete:assert not row['NRMSE'];continue
                        if np.isfinite(values[j]):assert np.isclose(float(row['NRMSE']),values[j],rtol=1e-11,atol=1e-13);checks+=1
                        else:assert not row['NRMSE']
    checked_files=[root/n for n in ['config.json','prepared.json','source_manifest.json','cases.json','preprocessing.json','dataset.json','training.csv','reconstruction_training.csv','selection_frozen.json','reconstruction_selection.json']]
    if (root/'software_preflight.json').exists():checked_files.append(root/'software_preflight.json')
    checked_files += [p for folder in ['models','evaluations'] for p in (root/folder).rglob('*') if p.is_file()]
    write(root/'verification.json',dict(status='passed',NRMSE_recomputations=checks,early_NRMSE_recomputations=early_checks,accuracy_and_physical_validity_recomputed=True,
        output_sha256={str(p.relative_to(root)):sha(p) for p in checked_files},old_data_unchanged=True,frozen_sources_unchanged=True,disjoint_ICs=True,test_after_selection=True,
        definition='RMSE(Cantera,NN)/RMS(Cantera); ratio, not percent; no full score for incomplete',training_physics=False))


def execute(root,require_publication=True):
    lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'worker_started.json').exists():raise RuntimeError('No duplicate worker or budget reset')
    cfg=json.loads((root/'config.json').read_text());prepared=json.loads((root/'prepared.json').read_text());meta=json.loads((root/'dataset.json').read_text())
    assert prepared['status']=='passed' and sha(root/'config.json')==prepared['config_sha256']
    assert sha(root/'source_manifest.json')==prepared['source_manifest_sha256']
    assert sha(root/'dataset.json')==prepared['dataset_metadata_sha256']
    assert sha(root/'cases.json')==prepared['cases_sha256'] and sha(root/'preprocessing.json')==prepared['preprocessing_sha256']
    assert all(sha(root/'source_snapshot'/p)==h for p,h in json.loads((root/'source_manifest.json').read_text()).items())
    assert all(sha(root/'data'/f'{s}.npz')==h for s,h in meta['data_sha256'].items())
    assert sha(root/'mechanism.yaml')==meta['mechanism_sha256']
    if require_publication:
        p=json.loads(Path('/workspace/react_rebuild_followup/run_001/publication.json').read_text())
        assert p['status']=='passed' and p['remote_verified'] and p['source_manifest_sha256']==sha(root/'source_manifest.json')
    write(root/'worker_started.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),budget_seconds=cfg['execution']['total_budget_seconds']))
    torch.set_num_threads(cfg['execution']['threads']);torch.set_num_interop_threads(1)
    started=time.perf_counter();deadline=started+cfg['execution']['total_budget_seconds']-prepared['prior_diagnostic_seconds']-meta['prepare_seconds'];train_deadline=deadline-cfg['execution']['reserve_evaluation_seconds']
    try:
        with np.load(root/'data/train.npz') as f:train=f['states']
        with np.load(root/'data/val.npz') as f:val=f['states']
        cases=json.loads((root/'cases.json').read_text());info=json.loads((root/'preprocessing.json').read_text());scale=FifthRootScale(info['minimum'],info['maximum'])
        reconstruction=train_reconstruction(root,cfg,train,val,scale,train_deadline)
        model=train_operator(root,cfg,train,val,cases,scale.subset(REP),train_deadline)
        evaluate(root,cfg,model,reconstruction,cases,scale,deadline);verify(root,cfg,cases)
        from .react_deeponet_rebuild_report import report
        report(root)
        complete=json.loads((root/'report/complete.json').read_text());assert complete['status']=='passed'
        assert all(sha(root/'report'/p)==h for p,h in complete['output_sha256'].items())
        selection=json.loads((root/'selection_frozen.json').read_text())
        write(root/'complete.json',dict(status='passed',update_target_completed=selection['update_target_completed'],stop_reason=selection['stop_reason'],wall_seconds=time.perf_counter()-started,
            verification_sha256=sha(root/'verification.json'),report_complete_sha256=sha(root/'report/complete.json')))
        write(root/'status.json',dict(phase='complete',status='passed',stop_reason=selection['stop_reason']));write(root/'exit.json',dict(returncode=0))
    except BaseException as exc:
        write(root/'failure.json',dict(status='failed',error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));write(root/'status.json',dict(phase='failed',error=str(exc)));write(root/'exit.json',dict(returncode=1));raise
    finally:lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','run'],required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/react_deeponet_v2.json'));a=p.parse_args()
    if a.phase=='prepare':prepare(a.root.resolve(),a.config)
    else:execute(a.root.resolve())


if __name__=='__main__':main()
