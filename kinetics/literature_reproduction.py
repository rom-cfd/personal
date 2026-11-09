"""Reproduce the released H2 0D ANN first; compare post-processors without training.

All model calls use the unmodified author Network and netOneStep. The additional
BCT projection is Eqs. (6)-(9) of Wang et al. 2025 with a common HP closure; this
diagnostic uses the older released 7-species network, not that paper's 9-species
trained model. Old project data, checkpoints and result directories are read-only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/literature-reproduction-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/literature-reproduction-cache')
import argparse,csv,hashlib,importlib.util,json,time,warnings,shutil,traceback
from pathlib import Path
import numpy as np
import cantera as ct
import torch
from .physics import element_matrix,project_species

AUTHOR_COMMIT='bc07e5a210ba357b8a5350b460a751a68a002174'
MODELS=('642w_msDsClipESH2_constP','528w_msDsH2_constP')
MODES=('author_original','published_BCT_projection_common_HP','current_projection_common_HP')
RUN_ROOT=None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False));temp.replace(path)


def csv_file(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def ratio(reference,prediction):
    reference=np.asarray(reference);prediction=np.asarray(prediction)
    den=np.linalg.norm(reference,axis=0)
    return np.divide(np.linalg.norm(prediction-reference,axis=0),den,out=np.full_like(den,np.nan,dtype=float),where=den>0)


def bct_element_projection(proposal,current,matrix,gamma=.1):
    """Published transformed-error weighting, Eqs.6-9; no extra positivity fix.

    E = A(Y_current-Y_proposal). This sign is required by Eq.7. Appendix text
    extractions can lose the minus sign; the conservation equation is verified.
    """
    y=np.asarray(proposal,dtype=np.float64)
    if np.any(y<0) or not np.isfinite(y).all():raise ValueError('Published BCT projection requires nonnegative finite composition')
    scale=y**(1-gamma);b=matrix*scale[None,:]
    error=matrix@(current-y)
    multiplier=np.linalg.solve(b@b.T,error)
    correction=scale*(b.T@multiplier)
    result=y+correction
    if not np.isfinite(result).all() or np.any(result<0):raise ValueError('Published projection produced invalid composition; no silent clipping')
    if np.max(np.abs(matrix@(result-current)))>2e-10:raise ValueError('Element correction failed its exact affine constraint')
    return result


def verify_sources(root):
    m=json.loads((root/'source_manifest.json').read_text());assert m['commit']==AUTHOR_COMMIT
    for f in m['files']:
        path=root/f['path'];raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==f['sha256']
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==f['git_blob_sha']
    return m


def load_author(root):
    spec=importlib.util.spec_from_file_location('published_zhang_modelclass',root/'modelclass.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def initial_state(mechanism,T,phi,P):
    # Match simulateData literally: the initial object is set to numeric P in Pa,
    # but its pressure number is stored as an atm-valued feature. Only composition
    # is taken from that object; ctOneStep subsequently multiplies the stored number
    # by one_atm. This legacy quirk must not be silently "fixed" in a parity test.
    gas=ct.Solution(str(mechanism));gas.TP=T,P;gas.set_equivalence_ratio(phi,'H2','O2:1,N2:3.76')
    return gas,np.r_[gas.T,gas.P,gas.Y]


def reference_original(model,state,steps,dt,mechanism):
    out=[state.copy()]
    for _ in range(steps):out.append(model.ctOneStep(out[-1].copy(),dt,str(mechanism),'constP',1e-8).ravel())
    return np.asarray(out)


def independent_reference(mechanism,state,steps,dt):
    gas=ct.Solution(str(mechanism));gas.TPY=state[0],state[1]*ct.one_atm,state[2:]
    reactor=ct.IdealGasConstPressureReactor(gas,clone=False);net=ct.ReactorNet([reactor]);net.max_time_step=1e-8
    out=[state.copy()]
    for j in range(steps):
        net.advance((j+1)*dt);out.append(np.r_[gas.T,gas.P/ct.one_atm,gas.Y])
    return np.asarray(out)


def rollout(model,args,state,steps,mode,mechanism,deadline):
    thermo=ct.Solution(str(mechanism));matrix=element_matrix(thermo)
    thermo.TPY=state[0],state[1]*ct.one_atm,state[2:];h0=thermo.enthalpy_mass
    out=[state.copy()];domain_count=0;nn_seconds=0.;repair_seconds=0.;reason='complete';last={}
    def observer(module,inputs,output):last['out']=output.detach().cpu().numpy().copy()
    handle=model.net.register_forward_hook(observer)
    try:
        for _ in range(steps):
            if time.perf_counter()>deadline:reason='wall_limit';break
            current=out[-1];start=time.perf_counter()
            pred=model.netOneStep(current.copy(),args).ravel();nn_seconds+=time.perf_counter()-start
            encoded=current.copy();encoded[2:]=(current[2:]**args['power_transform']-1)/args['power_transform']
            proposed_encoded=encoded+(last['out'].ravel()*np.asarray(args['Ystd'])+args['Ymu'])*args['delta_t']
            inverse_base=1+args['power_transform']*proposed_encoded[2:]
            domain_count+=int(np.any(inverse_base<0))
            if mode!='author_original':
                start=time.perf_counter()
                try:
                    if mode=='published_BCT_projection_common_HP':
                        y=bct_element_projection(pred[2:],current[2:],matrix,args['power_transform'])
                    else:
                        # Exactly the current project's guarded inverse followed by its
                        # existing generic weighted nonnegative elemental projection.
                        guarded=np.clip(inverse_base,0,1)**(1/args['power_transform'])
                        y=project_species(guarded,current[2:],matrix)
                    thermo.TPY=current[0],current[1]*ct.one_atm,current[2:];h=thermo.enthalpy_mass
                    thermo.HPY=h,current[1]*ct.one_atm,y
                    pred=np.r_[thermo.T,current[1],y]
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    reason=type(exc).__name__+': '+str(exc);break
                repair_seconds+=time.perf_counter()-start
            if not np.isfinite(pred).all() or pred[0]<200 or pred[0]>6000 or np.any(pred[2:]<0) or pred[2:].sum()<=0:
                reason='invalid_state';break
            out.append(pred)
    finally:handle.remove()
    trace=np.asarray(out);h_errors=[]
    for row in trace:
        thermo.TPY=row[0],row[1]*ct.one_atm,row[2:];h_errors.append(abs(thermo.enthalpy_mass-h0))
    return trace,dict(reason=reason,complete=len(trace)==steps+1,points=len(trace),
        inverse_domain_violating_steps=domain_count,nn_seconds=nn_seconds,repair_seconds=repair_seconds,
        mass_sum_max=float(np.max(np.abs(trace[:,2:].sum(1)-1))),
        element_max=float(np.max(np.abs((trace[:,2:]-state[2:])@matrix.T))),enthalpy_max_J_per_kg=float(max(h_errors)))


def main():
    global RUN_ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=Path('third_party/zhang_kinetics'))
    parser.add_argument('--output',type=Path,default=Path('results/literature_reproduction/run_001'))
    parser.add_argument('--steps',type=int,default=500,help='500 is the released .5-ms demo; larger values are a separately labelled horizon diagnostic')
    parser.add_argument('--max-wall-seconds',type=float,default=600);parser.add_argument('--execute',action='store_true')
    cmd=parser.parse_args();source=cmd.source.resolve();manifest=verify_sources(source)
    if cmd.steps<500:raise ValueError('At least 500 steps are needed for original-script parity')
    if not cmd.execute:
        print(json.dumps(dict(source_verified=True,action='Use --execute for pretrained inference only; no training',models=MODELS)));return
    root=cmd.output
    if root.exists():raise FileExistsError('Use a new immutable output directory')
    root.mkdir(parents=True);(root/'traces').mkdir();(root/'figures').mkdir()
    RUN_ROOT=root
    shutil.copyfile(__file__,root/'frozen_source.py')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);started=time.perf_counter();deadline=started+cmd.max_wall_seconds
    module=load_author(source);mech=source/'Chem/ESH2.yaml'
    gas=ct.Solution(str(mech));variables=['T','P_atm']+gas.species_names
    active=[0]+[i+2 for i,name in enumerate(gas.species_names) if name!='N2']
    write_json(root/'config.json',dict(upstream=manifest,models=MODELS,modes=MODES,temperatures=[1100,1200,1300,1400,1500,1600],
        phi=1.,pressure_atm=1.,dt=1e-6,steps=cmd.steps,horizon_s=cmd.steps*1e-6,species=gas.species_names,reactions=gas.n_reactions,
        new_training=False,precision='author float32 network and float64 preprocessing',threads=2,wall_limit_s=cmd.max_wall_seconds,
        joint_criterion='T+6 reactive species all NRMSE<=.1; N2 and P separate; not the old joint9 problem',
        diagnostic_scope='2025 element formula on released older 7-species network, with common HP closure; not full 2025 reproduction'))
    config=json.loads((root/'config.json').read_text());config['runner_sha256']=sha(__file__)
    config['initialization_note']='Literal author simulateData initialization; pressure feature is interpreted in atm by ctOneStep.'
    write_json(root/'config.json',config)
    results=[];metrics=[];references=[];parity=[];model_info=[];arrays={}
    with torch.inference_mode():
        for modelname in MODELS:
            directory=source/'Model'/('model'+modelname)/'checkpoint';args=json.loads((directory/'settings5000.json').read_text())
            assert args['delta_t']==1e-6 and args['layers']==[9,1600,800,400,9]
            model=module.ModelClass();model.buildModel(args);model.loadModel(str(directory/'model5000.pt'),device='cpu');model.net.eval()
            model_info.append(dict(model=modelname,parameters=sum(p.numel() for p in model.net.parameters()),
                train_samples=args['trainnum'],validation_samples=args['testnum'],loss=args['lossfun'],activation=args['actfun'],
                checkpoint_sha256=sha(directory/'model5000.pt')))
            for T in [1100,1200,1300,1400,1500,1600]:
                if time.perf_counter()>deadline:raise TimeoutError('Literature inference budget exhausted')
                _,state=initial_state(mech,T,1.,1.)
                reference=reference_original(model,state,cmd.steps,1e-6,mech)
                independent=independent_reference(mech,state,cmd.steps,1e-6)
                diff=ratio(independent,reference)
                references.append(dict(model=modelname,T0=T,temperature_RMSE_K=float(np.sqrt(np.mean((reference[:,0]-independent[:,0])**2))),
                    maximum_active_NRMSE=float(np.max(diff[active]))))
                arrays[f'{modelname}_T{T}_reference']=reference
                for mode in MODES:
                    trace,record=rollout(model,args,state,cmd.steps,mode,mech,deadline)
                    key=f'{modelname}_T{T}_{mode}';arrays[key]=trace
                    full=bool(record['complete']);ratios=ratio(reference,trace) if full else np.full(9,np.nan)
                    record.update(model=modelname,T0=T,mode=mode,maximum_active_NRMSE=float(np.max(ratios[active])) if full else None,
                        joint_active7_pass=bool(full and np.isfinite(ratios[active]).all() and np.max(ratios[active])<=.1),
                        temperature_RMSE_K=float(np.sqrt(np.mean((reference[:,0]-trace[:,0])**2))) if full else None)
                    results.append(record)
                    for j,var in enumerate(variables):metrics.append(dict(model=modelname,T0=T,mode=mode,variable=var,complete=full,
                        nrmse=float(ratios[j]) if full else None,reference_RMS=float(np.sqrt(np.mean(reference[:,j]**2))),
                        rmse=float(np.sqrt(np.mean((reference[:,j]-trace[:,j])**2))) if full else None))
                    print('CASE',modelname,T,mode,record['complete'],record['maximum_active_NRMSE'],flush=True)
                    write_json(root/'status.json',dict(phase='evaluating_released_models',finished=len(results),total=36))
                # One complete reference run checks our explicit rollout loop against
                # the author script itself, not another independently designed model.
                if T==1100:
                    author_ref,author_pred=model.simulateData(args,1.,T,1.,500,str(mech),'H2','constP',1e-8)
                    np.testing.assert_array_equal(author_ref,reference[:501])
                    np.testing.assert_array_equal(author_pred,arrays[f'{modelname}_T{T}_author_original'][:501])
                    parity.append(dict(model=modelname,full_501_state_original_script_parity=True))
    np.savez_compressed(root/'traces/all_profiles.npz',**arrays)
    csv_file(root/'summary.csv',results);csv_file(root/'NRMSE.csv',metrics);csv_file(root/'reference_consistency.csv',references)
    write_json(root/'models.json',model_info)
    # Recompute every full-profile CSV score from saved trace bytes.
    with np.load(root/'traces/all_profiles.npz') as archive:
        for r in results:
            if not r['complete']:continue
            reference=archive[f'{r["model"]}_T{r["T0"]}_reference'];pred=archive[f'{r["model"]}_T{r["T0"]}_{r["mode"]}']
            values=ratio(reference,pred)
            for j,var in enumerate(variables):
                row=next(v for v in metrics if v['model']==r['model'] and v['T0']==r['T0'] and v['mode']==r['mode'] and v['variable']==var)
                assert np.isclose(values[j],row['nrmse'],rtol=1e-12,atol=1e-14)
    verify_sources(source)
    write_json(root/'verification.json',dict(status='passed',original_script_parity=parity,source_hashes_unchanged=True,
        NRMSE_checks=sum(r['complete'] for r in results)*len(variables),formula='RMSE(Cantera,NN)/RMS(Cantera), dimensionless ratio',
        independent_reference_max_NRMSE=max(r['maximum_active_NRMSE'] for r in references)))
    # Scientific result figures only. No slide decks, narration, or presentation assets.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
    styles=[('author_original','Released ANN','#2563EB'),('published_BCT_projection_common_HP','Published BCT projection + HP','#D97706'),('current_projection_common_HP','Current projection + HP','#0F766E')]
    for modelname in MODELS:
        for T in [1100,1200,1300,1400,1500,1600]:
            ref=arrays[f'{modelname}_T{T}_reference'];fig,axes=plt.subplots(2,4,figsize=(14,6.1))
            for ax,j in zip(axes.flat,[0]+list(range(2,9))):
                ax.plot(np.arange(len(ref))*.001,ref[:,j],color='black',lw=1.8,label='Cantera')
                for mode,label,color in styles:
                    pred=arrays[f'{modelname}_T{T}_{mode}'];ax.plot(np.arange(len(pred))*.001,pred[:,j],ls='--',color=color,label=label)
                ax.set_title(variables[j]);ax.set_xlabel('Time [ms]')
                if j>1:ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2),useMathText=False)
            axes[0,0].legend(fontsize=7);fig.suptitle(f'Author checkpoint {modelname}: T0={T} K, P=1 atm, phi=1; no retraining')
            fig.tight_layout()
            for ext in ['png','pdf']:fig.savefig(root/'figures'/f'{modelname}_T{T}.{ext}',dpi=180)
            plt.close(fig)
    grouped=[]
    for modelname in MODELS:
        for mode in MODES:
            rr=[r for r in results if r['model']==modelname and r['mode']==mode]
            grouped.append(dict(model=modelname,mode=mode,complete=sum(r['complete'] for r in rr),
                joint_active7_pass=sum(r['joint_active7_pass'] for r in rr),total=6,
                max_NRMSE=max((r['maximum_active_NRMSE'] for r in rr if r['complete']),default=None),
                max_mass_sum_error=max(r['mass_sum_max'] for r in rr)))
    write_json(root/'aggregate.json',grouped)
    outputs={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name not in ['status.json','complete.json']}
    write_json(root/'complete.json',dict(status='passed',elapsed_seconds=time.perf_counter()-started,
        trained_new_models=0,author_cases=12,diagnostic_runs=24,output_sha256=outputs))
    write_json(root/'status.json',dict(phase='complete',status='passed',finished=36))
    print('REPRODUCTION COMPLETE',json.dumps(grouped),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if RUN_ROOT is not None:
            write_json(RUN_ROOT/'failure.json',dict(status='failed',error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
