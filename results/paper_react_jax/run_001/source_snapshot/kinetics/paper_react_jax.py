"""Native-JAX 2023 React-DeepONet protocol, with unresolved choices exposed.

This is independent code, NOT the unpublished Kumar implementation. Canonical
modified-MLP mathematics and initialization follow the explicitly credited Wang
base notebook, pinned in the configuration. Existing experiments are read-only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/paper-react-jax-mpl')
import argparse, csv, hashlib, importlib.util, json, pickle, shutil, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.example_libraries import optimizers

REPO = Path('/workspace/personal')
ORIGINALS = {
 'results/long10ms_data/trajectories.npz':'8d41f19f2145e12f9e89ee3eaa88fb0b1b93a84dcaaee65dcaa2e8b264077563',
 'results/long10ms_data/training_pairs.npz':'44db17b16e167ea389631dacb8015528f76cbd5c2072e9fb919701e54d50fa3e'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''): h.update(block)
    return h.hexdigest()

def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True, exist_ok=True)
    def clean(x):
        if isinstance(x, dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)

def csvout(p, rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def initialize(sizes, key):
    def layer(k,a,b):return random.normal(k,(a,b))*jnp.sqrt(2./(a+b)), jnp.zeros(b)
    u,v=random.split(key)
    enc_u=layer(u,sizes[0],sizes[1]);enc_v=layer(v,sizes[0],sizes[1])
    keys=random.split(key,len(sizes))[1:]
    dense=[layer(random.split(k)[0],a,b) for k,a,b in zip(keys,sizes[:-1],sizes[1:])]
    return dense,enc_u,enc_v

def network(params, x):
    dense,(uw,ub),(vw,vb)=params
    u=jnp.tanh(x@uw+ub);v=jnp.tanh(x@vw+vb)
    for w,_unused_bias in dense[:-1]:
        gate=jnp.tanh(x@w);x=gate*u+(1-gate)*v
    w,b=dense[-1];return x@w+b

def model(params, x, tau, time_unit=1e-6):
    branch,trunk,shift=params
    coordinate=jnp.log(tau/time_unit)+network(shift,x)
    a=network(branch,x).reshape((-1,5,40));b=network(trunk,coordinate).reshape((-1,5,40))
    return jnp.sum(a*b,axis=-1)

def encode(x,lo,span):return (2*(np.asarray(x)**.2-lo)/span-1).astype(np.float32)
def decode(z,lo,span):return ((np.asarray(z,dtype=np.float64)+1)*.5*span+lo)**5
def nrmse(ref,pred):
    den=np.linalg.norm(ref,axis=0)
    return np.divide(np.linalg.norm(pred-ref,axis=0),den,out=np.full(den.shape,np.nan),where=den>0)
def valid(a):return bool(np.isfinite(a).all() and np.all((a[...,0]>=200)&(a[...,0]<=6000)) and np.min(a[...,1:])>=0)

def cases_for(cfg):
    r,c,e=cfg['reported'],cfg['reconstruction_choices'],cfg['execution'];rows=[];used=set()
    for split,ts,ps in [('train',r['generated_T0'],c['train_phi']),('val',c['validation_T0'],c['validation_phi']),
                       ('paper_test',r['generated_T0'],[.6,.7,1.3])]:
        for t in ts:
            for p in ps:
                assert (t,p) not in used;used.add((t,p));rows.append(dict(case_id=len(rows),split=split,T0=float(t),phi=float(p)))
    # Freeze independent ICs before any model is trained or selected.
    rng=np.random.default_rng(e['fresh_test_seed'])
    for _ in range(e['fresh_test_cases']):
        t=float(rng.uniform(1020,1200));p=float(rng.uniform(.8,1.2));assert (t,p) not in used
        used.add((t,p));rows.append(dict(case_id=len(rows),split='fresh_test',T0=t,phi=p))
    return rows

def ann_replay(root):
    """Execute unmodified author simulateData; no project postprocessor."""
    import torch
    torch.set_num_threads(2)
    source=REPO/'third_party/zhang_kinetics';manifest=json.loads((source/'source_manifest.json').read_text())
    for item in manifest['files']:assert sha(source/item['path'])==item['sha256']
    spec=importlib.util.spec_from_file_location('pinned_zhang_modelclass',source/'modelclass.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=root/'ann_author_replay';out.mkdir();rows=[];scores=[];arrays={}
    for name in ['642w_msDsClipESH2_constP','528w_msDsH2_constP']:
        d=source/'Model'/('model'+name)/'checkpoint';settings=json.loads((d/'settings5000.json').read_text())
        net=module.ModelClass();net.buildModel(settings);net.loadModel(str(d/'model5000.pt'),device='cpu');net.net.eval()
        for t in [1100,1200,1300,1400,1500,1600]:
            with torch.inference_mode():ref,pred=net.simulateData(settings,1.,t,1.,500,str(source/'Chem/ESH2.yaml'),'H2','constP',1e-8)
            # Author columns are T, pressure-atm, then the seven mechanism species.
            active=[0,2,3,4,5,6,7];err=nrmse(ref,pred)
            key=name+'_T'+str(t);arrays[key+'_reference']=ref;arrays[key+'_prediction']=pred
            rows.append(dict(model=name,T0=t,steps=500,dt_s=1e-6,joint_active7=bool(np.max(err[active])<=.1),max_active_NRMSE=float(np.max(err[active]))))
            for j,var in enumerate(['T','P_atm','H','O','H2O','OH','O2','H2','N2']):scores.append(dict(model=name,T0=t,variable=var,NRMSE=float(err[j])))
    np.savez_compressed(out/'profiles.npz',**arrays);csvout(out/'summary.csv',rows);csvout(out/'NRMSE.csv',scores)
    write(out/'complete.json',dict(status='passed',type='released_author_checkpoint_inference_not_training',upstream_commit=manifest['commit'],
        no_project_repair=True,output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))

def prepare(root,config):
    import cantera as ct
    if root.exists():raise FileExistsError('Use a new root; old results are immutable')
    started=time.perf_counter();cfg=json.loads(Path(config).read_text());root.mkdir(parents=True)
    write(root/'config.json',cfg);rows=cases_for(cfg);write(root/'cases.json',rows)
    for d in ['data','models','report','evaluations','source_snapshot/kinetics','source_snapshot/scripts']:(root/d).mkdir(parents=True)
    sources=['kinetics/paper_react_jax.py','scripts/supervise_paper_react.py','scripts/monitor_paper_react.py']
    for name in sources:shutil.copyfile(REPO/name,root/'source_snapshot'/name)
    (root/'source_snapshot/kinetics/__init__.py').write_text('"""Frozen paper protocol."""\n')
    write(root/'source_manifest.json',{str(p.relative_to(root/'source_snapshot')):sha(p) for p in (root/'source_snapshot').rglob('*.py')})
    mech=REPO/cfg['reconstruction_choices']['mechanism'];shutil.copyfile(mech,root/'mechanism.yaml')
    r,c=cfg['reported'],cfg['reconstruction_choices'];dt=c['reference_dt_s'];n=int(round(r['horizon_s']/dt));datasets={}
    for split in ['train','val','paper_test','fresh_test']:
        data=[]
        for case in [x for x in rows if x['split']==split]:
            g=ct.Solution(str(mech));g.TP=case['T0'],ct.one_atm;g.set_equivalence_ratio(case['phi'],'H2','O2:1,N2:3.76')
            reactor=ct.IdealGasConstPressureReactor(g,clone=False);solver=ct.ReactorNet([reactor]);solver.rtol=1e-10;solver.atol=1e-20
            rep=[g.species_index(s)+1 if s!='T' else 0 for s in r['representatives']]
            trace=np.empty((n+1,g.n_species+1));trace[0]=np.r_[g.T,g.Y]
            for j in range(1,n+1):solver.advance(j*dt);trace[j]=np.r_[g.T,g.Y]
            assert np.min(trace[:,1:])>=-1e-14;trace[:,1:]=np.maximum(trace[:,1:],0);data.append(trace)
        data=np.asarray(data);np.savez_compressed(root/'data'/f'{split}.npz',states=data)
        datasets[split]=dict(cases=len(data),sha256=sha(root/'data'/f'{split}.npz'))
        if split=='train':
            a=data[:,:,rep]**.2;lo=a.min(axis=(0,1));hi=a.max(axis=(0,1));assert np.all(hi>lo)
    write(root/'preprocessing.json',dict(minimum=lo.tolist(),span=(hi-lo).tolist(),representative_indices=rep,fit='train_only'))
    ann_replay(root)
    assert {name:sha(REPO/name) for name in ORIGINALS}==ORIGINALS
    write(root/'dataset.json',dict(splits=datasets,mechanism_sha256=sha(root/'mechanism.yaml'),representative_indices=rep,
        variables=['T']+g.species_names,dt_s=dt,originals=ORIGINALS))
    write(root/'prepared.json',dict(status='passed',prepare_seconds=time.perf_counter()-started,
        config_sha256=sha(root/'config.json'),source_manifest_sha256=sha(root/'source_manifest.json'),
        dataset_sha256=sha(root/'dataset.json'),preprocessing_sha256=sha(root/'preprocessing.json'),cases_sha256=sha(root/'cases.json'),
        ann_complete_sha256=sha(root/'ann_author_replay/complete.json')))
    write(root/'status.json',dict(phase='prepared',training_started=False))
    print('PREPARED',root,flush=True)

def initial_params(cfg):
    r=cfg['reported'];return tuple(initialize(r[name],random.PRNGKey(seed)) for name,seed in [('branch',1234),('trunk',4321),('shift',5678)])

def draw(key,states,batch,offset,dt):
    key,ki,kt,kd=random.split(key,4)
    i=random.randint(ki,(batch,),0,len(states));t=random.randint(kt,(batch,),0,states.shape[1]-offset)
    length=random.randint(kd,(batch,),1,offset+1)
    return key,(states[i,t],length[:,None]*dt,states[i,t+length])

def compiled_training(cfg, train, count):
    r,c=cfg['reported'],cfg['reconstruction_choices'];opt_init,opt_update,get=optimizers.adam(r['learning_rate'])
    def loss(p,batch):
        x,t,y=batch;return jnp.mean((model(p,x,t,c['time_unit_s'])-y)**2)
    def one(carry,_):
        state,key,idx=carry
        key,batch=draw(key,train,r['batch_size'],int(round(r['window_s']/c['reference_dt_s'])),c['reference_dt_s'])
        value,grad=jax.value_and_grad(loss)(get(state),batch)
        state=opt_update(idx,grad,state)
        return (state,key,idx+1),value
    @jax.jit
    def chunk(state,key,idx):return jax.lax.scan(one,(state,key,idx),None,length=count)
    return opt_init,get,jax.jit(loss),chunk

def save_checkpoint(path,state,key,update):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:pickle.dump(jax.device_get(dict(optimizer=state,key=key,update=update)),f,protocol=5)
    os.replace(tmp,path)

def run(root,publication,deadline):
    started=time.perf_counter();cfg=json.loads((root/'config.json').read_text());r,c,e=cfg['reported'],cfg['reconstruction_choices'],cfg['execution']
    if (root/'worker_started.json').exists():raise RuntimeError('This run cannot restart or duplicate')
    pub=json.loads(Path(publication).read_text());assert pub['status']=='passed' and pub['remote_verified'] and pub['branch']=='bw'
    assert pub['prepared_sha256']==sha(root/'prepared.json') and pub['source_manifest_sha256']==sha(root/'source_manifest.json')
    prepared=json.loads((root/'prepared.json').read_text())
    for name in ['config','preprocessing','cases','dataset']:assert sha(root/(name+'.json'))==prepared[name+'_sha256']
    assert sha(root/'ann_author_replay/complete.json')==prepared['ann_complete_sha256']
    manifest=json.loads((root/'source_manifest.json').read_text())
    assert all(sha(root/'source_snapshot'/name)==digest for name,digest in manifest.items())
    data_meta=json.loads((root/'dataset.json').read_text());prep=json.loads((root/'preprocessing.json').read_text());rep=prep['representative_indices']
    lo=np.array(prep['minimum']);span=np.array(prep['span']);arrays={}
    for split in ['train','val']:
        p=root/'data'/f'{split}.npz';assert sha(p)==data_meta['splits'][split]['sha256']
        with np.load(p) as f:arrays[split]=jnp.asarray(encode(f['states'][:,:,rep],lo,span))
    write(root/'worker_started.json',dict(pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),jax=jax.__version__,devices=[str(d) for d in jax.devices()],deadline_monotonic=deadline))
    count=e['history_chunk'];opt_init,get,loss,chunk=compiled_training(cfg,arrays['train'],count)
    params=initial_params(cfg);state=opt_init(params);key=random.PRNGKey(e['seed']);update=0;active=0.;best=float('inf');history=[];validation=[]
    _,fixed=draw(random.PRNGKey(9321),arrays['val'],e['validation_batch'],int(round(r['window_s']/c['reference_dt_s'])),c['reference_dt_s'])
    # Compilation is charged to this run, and does not commit an optimizer update.
    compile_start=time.perf_counter();chunk.lower(state,key,jnp.asarray(0)).compile();compile_seconds=time.perf_counter()-compile_start
    while update+count<=r['updates'] and time.perf_counter()<deadline-e['evaluation_reserve_seconds']:
        tick=time.perf_counter();(state,key,idx),values=chunk(state,key,jnp.asarray(update));values=np.asarray(values);active+=time.perf_counter()-tick;update=int(idx)
        if not np.isfinite(values).all():raise FloatingPointError('Nonfinite original-protocol training loss')
        history.append(dict(update=update,mean_loss=float(values.mean()),last_loss=float(values[-1]),lr=r['learning_rate'],active_seconds=active))
        if update==count or update%e['checkpoint_every']==0 or update==r['updates']:
            val=float(loss(get(state),fixed));validation.append(dict(update=update,validation_MSE=val))
            save_checkpoint(root/'models'/f'update_{update:06d}.pkl',state,key,update)
            if np.isfinite(val) and val<best:best=val;save_checkpoint(root/'models/validation_selected.pkl',state,key,update)
            write(root/'selection_progress.json',dict(best_validation_MSE=best,validations=validation))
        csvout(root/'training.csv',history);write(root/'status.json',dict(phase='training',update=update,target=r['updates'],lr=r['learning_rate'],active_seconds=active,compile_seconds=compile_seconds))
    if update==0:raise TimeoutError('No training update fit before the evaluation reserve')
    save_checkpoint(root/'models/protocol_final.pkl',state,key,update)
    with (root/'models/validation_selected.pkl').open('rb') as f:chosen=pickle.load(f)
    write(root/'selection_frozen.json',dict(status='passed',primary='protocol_final',protocol_update=update,selected_update=chosen['update'],selected_validation_MSE=best,
        target_updates=r['updates'],stop_reason='requested_updates_completed' if update==r['updates'] else 'overall_time_cap',
        active_seconds=active,compile_seconds=compile_seconds,wall_seconds=time.perf_counter()-started,final_lr=r['learning_rate'],
        parameters=sum(x.size for x in jax.tree_util.tree_leaves(params)),test_used_for_selection=False,
        selected_sha256=sha(root/'models/validation_selected.pkl'),final_sha256=sha(root/'models/protocol_final.pkl')))
    evaluate(root,cfg,{'protocol_final':get(state),'validation_selected':get(chosen['optimizer'])},deadline)
    outputs={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name not in ['status.json','exit.json','complete.json','failure.json']}
    write(root/'complete.json',dict(status='passed',at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,output_sha256=outputs,
        verification_sha256=sha(root/'verification.json'),report_complete_sha256=sha(root/'report/complete.json')))
    write(root/'status.json',dict(phase='complete',verified=True,update=update,target=r['updates']))

def evaluate(root,cfg,models,deadline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    prep=json.loads((root/'preprocessing.json').read_text());lo=np.array(prep['minimum']);span=np.array(prep['span']);rep=prep['representative_indices']
    r,c=cfg['reported'],cfg['reconstruction_choices'];cases=json.loads((root/'cases.json').read_text());meta=json.loads((root/'dataset.json').read_text())
    bysplit={};allrows=[];metricrows=[];saved={};checks=0
    for split in ['train','val','paper_test','fresh_test']:
        p=root/'data'/f'{split}.npz';assert sha(p)==meta['splits'][split]['sha256']
        with np.load(p) as f:bysplit[split]=f['states'][:,:,rep]
    predictors={name:jax.jit(lambda x,t,p=p:model(p,x,t,c['time_unit_s'])) for name,p in models.items()}
    nwindow=int(round(r['window_s']/c['reference_dt_s']));windows=int(round(r['horizon_s']/r['feedback_step_s']));tau=jnp.arange(1,nwindow+1,dtype=jnp.float32)[:,None]*c['reference_dt_s']
    report=root/'report'
    with PdfPages(report/'all_profiles.pdf') as pdf, PdfPages(report/'ignition_zoom.pdf') as zoom:
        for split in ['paper_test','fresh_test','val','train']:
            splitcases=[x for x in cases if x['split']==split]
            for case,truth in zip(splitcases,bysplit[split]):
                if split=='train' and case['phi']!=1.:continue
                if time.perf_counter()>deadline-15:raise TimeoutError('Evaluation exceeded the fixed run deadline')
                key='case_'+str(case['case_id']);saved[key+'_truth']=truth
                fig,axes=plt.subplots(2,3,figsize=(13,6));zf,zaxes=plt.subplots(2,3,figsize=(13,6))
                tim=np.arange(len(truth))*c['reference_dt_s']*1e3
                dT=np.gradient(truth[:,0]);peak=int(np.argmax(dT));zlo=max(0,peak-250);zhi=min(len(truth)-1,peak+500)
                for j,var in enumerate(r['representatives']):
                    axes.flat[j].plot(tim,truth[:,j],'k',label='Cantera');zaxes.flat[j].plot(tim[zlo:zhi+1],truth[zlo:zhi+1,j],'k',label='Cantera')
                for name,predict in predictors.items():
                    z=encode(truth[0:1],lo,span);trace=[truth[0]];boundary_valid=True;dense_valid=True
                    for _ in range(windows):
                        encoded=np.asarray(predict(jnp.repeat(jnp.asarray(z),nwindow,axis=0),tau))
                        physical=decode(encoded,lo,span)
                        if not np.isfinite(physical).all():break
                        trace.extend(physical);boundary_valid=boundary_valid and valid(physical[-1:]);dense_valid=dense_valid and valid(physical);z=encoded[-1:]
                    pred=np.asarray(trace);finite=len(pred)==len(truth);boundary_valid=bool(boundary_valid and finite);dense_valid=bool(dense_valid and finite)
                    saved[key+'_'+name]=pred
                    errs=[];earlyerrs=[];mae=None
                    if finite:
                        errs=nrmse(truth,pred);earlyerrs=nrmse(truth[:1001],pred[:1001])
                        # Diagnostic normalized MAE is evaluated directly in the signed fifth-root coordinate, without clipping.
                        mae=float(np.mean(np.abs((np.sign(pred)*np.abs(pred)**.2-truth**.2)/span)))
                        checks+=len(errs)*2
                    row=dict(model=name,split=split,case_id=case['case_id'],T0=case['T0'],phi=case['phi'],finite_complete=finite,raw_boundary_complete=boundary_valid,dense_physical_valid=dense_valid,
                        curve_joint5_full=bool(finite and max(errs)<=.1),curve_joint5_full_early=bool(finite and max(errs)<=.1 and max(earlyerrs)<=.1),
                        strict_joint5_full_early=bool(dense_valid and max(errs)<=.1 and max(earlyerrs)<=.1),normalized_root_MAE=mae,primary_literal_weights=name=='protocol_final')
                    allrows.append(row)
                    for j,var in enumerate(r['representatives']):
                        for window,a,b in [('full_0p5ms',truth,pred),('early_0p1ms',truth[:1001],pred[:1001])]:
                            metricrows.append(dict(model=name,case_id=case['case_id'],split=split,variable=var,window=window,raw_complete=boundary_valid,
                                NRMSE=float(nrmse(a,b)[j]) if boundary_valid else None,diagnostic_NRMSE=float(nrmse(a,b)[j]) if finite else None))
                        axes.flat[j].plot(tim[:len(pred)],pred[:,j],'--',label=name)
                        zaxes.flat[j].plot(tim[zlo:min(zhi+1,len(pred))],pred[zlo:zhi+1,j],'--',label=name)
                for axs in [axes,zaxes]:
                    for j,var in enumerate(r['representatives']):axs.flat[j].set_title(var);axs.flat[j].set_xlabel('Time [ms]')
                    axs.flat[-1].axis('off');axs.flat[0].legend(fontsize=7)
                label=f'{split}: T0={case["T0"]:.2f} K, phi={case["phi"]:.4f}; 25 self-fed 20us steps; no repair'
                fig.suptitle(label);zf.suptitle(label+'; Cantera ignition window');fig.tight_layout();zf.tight_layout();pdf.savefig(fig);zoom.savefig(zf);plt.close(fig);plt.close(zf)
                write(root/'status.json',dict(phase='evaluating',cases_finished=len(allrows)//len(models)))
    np.savez_compressed(root/'evaluations/profiles.npz',**saved);csvout(root/'evaluations/cases.csv',allrows);csvout(root/'evaluations/NRMSE.csv',metricrows)
    summary=[]
    for name in models:
        for split in ['paper_test','fresh_test','val','train']:
            rr=[x for x in allrows if x['model']==name and x['split']==split]
            summary.append(dict(model=name,split=split,total=len(rr),**{key:sum(x[key] for x in rr) for key in ['finite_complete','raw_boundary_complete','dense_physical_valid','curve_joint5_full','curve_joint5_full_early','strict_joint5_full_early']},
                mean_normalized_root_MAE=float(np.mean([x['normalized_root_MAE'] for x in rr if x['normalized_root_MAE'] is not None]))))
    csvout(report/'summary.csv',summary)
    # Recompute every serialized NRMSE using independent saved trace arrays.
    with np.load(root/'evaluations/profiles.npz') as traces:
        for row in metricrows:
            a=traces['case_'+str(row['case_id'])+'_truth'];b=traces['case_'+str(row['case_id'])+'_'+row['model']]
            if len(a)!=len(b):continue
            if row['window']=='early_0p1ms':a,b=a[:1001],b[:1001]
            j=r['representatives'].index(row['variable']);expected=np.sqrt(np.mean((b[:,j]-a[:,j])**2))/np.sqrt(np.mean(a[:,j]**2))
            assert np.isclose(expected,row['diagnostic_NRMSE'],rtol=1e-11,atol=1e-13)
    assert {name:sha(REPO/name) for name in ORIGINALS}==ORIGINALS
    history=list(csv.DictReader((root/'training.csv').open()));fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].semilogy([int(x['update']) for x in history],[float(x['mean_loss']) for x in history]);ax[0].set_title('Native JAX original-protocol training MSE');ax[1].plot([int(x['update']) for x in history],[float(x['lr']) for x in history]);ax[1].set_title('Fixed original Adam learning rate');fig.tight_layout();fig.savefig(report/'loss_and_lr.png',dpi=160);plt.close(fig)
    write(root/'verification.json',dict(status='passed',NRMSE_recomputed=checks,original_data_unchanged=True,formula='RMSE(Cantera,NN)/RMS(Cantera); dimensionless ratio',
        scientific_success_not_implied=True,selection_frozen_sha256=sha(root/'selection_frozen.json')))
    write(report/'complete.json',dict(status='passed',output_sha256={p.name:sha(p) for p in report.iterdir() if p.is_file()}))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path);p.add_argument('--publication',type=Path);p.add_argument('--deadline',type=float);a=p.parse_args()
    try:
        if a.phase=='prepare':prepare(a.root,a.config)
        else:run(a.root,a.publication,a.deadline);write(a.root/'exit.json',dict(returncode=0,complete_sha256=sha(a.root/'complete.json')))
    except BaseException:
        if a.root.exists():write(a.root/'failure.json',dict(traceback=traceback.format_exc()));write(a.root/'exit.json',dict(returncode=1))
        raise

if __name__=='__main__':main()
