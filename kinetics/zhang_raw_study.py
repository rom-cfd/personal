"""Multi-scale sampling + Zhang full-state increments, no physics correction.

Train/freeze all validation a priori singleton choices, then assess 10/20-ms
raw rollouts. A matched raw-I/O manifold control at gamma=.1 isolates sampling.
"""
import argparse,gc,hashlib,json,time
from pathlib import Path
import cantera as ct
import numpy as np
import pandas as pd
from .zhang_model import ZhangPredictor,ZhangCommittee
from .batched_training import train_group
from .committee import fit_metric
from .data import mechanism,samples
from .metrics import evaluate
from .posteriori import metrics,ignition
from .long_committee_study import horizon_summary


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2))


def raw_rollouts(p,truth,pressures,ids,dt,budget):
    state=np.c_[truth[:,0],pressures];histories=[[s.copy()] for s in state]
    active=np.ones(len(ids),bool);statuses=['completed']*len(ids);events=[]
    for step in range(1,truth.shape[1]):
        live=np.flatnonzero(active)
        if not len(live):break
        output=p.predict_full(state[live])
        for j,k in enumerate(live):
            y=output[j];finite=np.isfinite(y).all()
            huge=bool(finite and (abs(y[0])>1e6 or np.max(np.abs(y[1:10]))>1e3 or abs(y[-1])>1e15))
            accepted=bool(finite and not huge)
            events.append(dict(trajectory=int(ids[k]),step=step,time=step*dt,selected_seed=int(p.last_diagnostics['selected_seed'][j]),
                finite_members=int(p.last_diagnostics['valid_count'][j]),accepted=accepted,T=y[0],sum_Y=np.sum(y[1:10]),P=y[-1]))
            if accepted:state[k]=y;histories[k].append(y.copy())
            else:active[k]=False;statuses[k]='numeric_magnitude_guard' if huge else 'undefined_BCT_inverse_or_nonfinite'
        if step%500==0:
            budget();print('RAW_STEP',step,'active',int(active.sum()),flush=True)
    return [np.asarray(h) for h in histories],statuses,events


def assess(trace,truth,pressure,dt,matrix,horizon,trajectory,status,refined_tau):
    steps=round(horizon/dt);a=trace[:steps+1]
    row,sp=metrics(a[:,:10],truth,dt,matrix,steps)
    bad=(np.any(a[:,1:10]<-1e-12,axis=1)|np.any(a[:,1:10]>1+1e-12,axis=1)|
         (a[:,0]<200)|(a[:,0]>6000)|(np.abs(a[:,1:10].sum(axis=1)-1)>.01)|
         (np.max(np.abs((a[:,1:10]-truth[0,1:])@matrix.T),axis=1)>.01)|
         (np.abs(a[:,-1]-pressure)>.01*pressure))
    violations=np.flatnonzero(bad)
    predicted=np.isfinite(row['predicted_tau400']);actual=np.isfinite(row['reference_tau400'])
    cross=np.flatnonzero(a[:,0]>=truth[0,0]+400)
    valid_ign=bool(predicted and len(cross) and (not len(violations) or cross[0]<violations[0]))
    tau=refined_tau if refined_tau<=horizon else np.nan
    row.update(trajectory=int(trajectory),status='completed' if row['complete'] else status,
        physically_admissible_complete=bool(row['complete'] and not bad.any()),
        first_physical_violation_seconds=float(violations[0]*dt) if len(violations) else np.nan,
        minimum_mass_fraction=float(a[:,1:10].min()),maximum_mass_fraction=float(a[:,1:10].max()),
        minimum_temperature=float(a[:,0].min()),maximum_temperature=float(a[:,0].max()),
        pressure_relative_drift_max=float(np.max(np.abs(a[:,-1]-pressure))/pressure),
        reference_ignites=bool(actual),prediction_ignites=bool(predicted),ignition_classification_correct=bool(actual==predicted and (predicted or row['complete'])),
        ignition_physically_admissible=valid_ign,reference_tau400_refined=tau,
        tau400_relative_error_refined=abs(row['predicted_tau400']-tau)/tau if predicted and np.isfinite(tau) else np.nan)
    # Report early and late intervals separately, requiring the entire interval.
    for name,lo,hi in [('early_0_2ms',0,.002),('late_2_10ms',.002,.01),('tail_10_20ms',.01,.02)]:
        left,right=round(lo/dt),round(hi/dt)
        row[name+'_complete']=bool(hi<=horizon and len(trace)>right)
        if row[name+'_complete']:
            error=trace[left:right+1,0]-truth[left:right+1,0]
            row[name+'_T_RMSE']=float(np.sqrt(np.trapezoid(error**2,dx=dt)/(hi-lo)))
        else:row[name+'_T_RMSE']=np.nan
    return row,sp


def build_references(data,ids,dt,horizon,out,budget):
    steps=round(horizon/dt);refs=[];taus=[];overlap=[]
    for i in ids:
        budget();g=mechanism();initial=data['states'][i,0];p=data['pressures'][i]
        g.TPY=initial[0],p,initial[1:]
        r=ct.IdealGasConstPressureReactor(g,clone=True);net=ct.ReactorNet([r]);net.rtol=1e-10;net.atol=1e-20
        t=np.empty((steps+1,10));t[0]=initial
        for j in range(1,steps+1):
            net.advance(j*dt);t[j]=np.r_[r.T,np.maximum(r.phase.Y,0)]
        refs.append(t);overlap.append(float(np.max(np.abs(t[:5001,0]-data['states'][i,:,0]))))
        # Fine reference ignition crossing only, no ANN input or model selection.
        fine=mechanism();fine.TPY=initial[0],p,initial[1:]
        rr=ct.IdealGasConstPressureReactor(fine,clone=True);nn=ct.ReactorNet([rr]);nn.rtol=1e-10;nn.atol=1e-20
        last=initial[0];tau=np.nan;fine_dt=dt/10
        for j in range(1,round(horizon/fine_dt)+1):
            nn.advance(j*fine_dt)
            if rr.T>=initial[0]+400:
                tau=((j-1)+(initial[0]+400-last)/(rr.T-last))*fine_dt;break
            last=rr.T
        taus.append(tau)
        print('REFERENCE_20MS',int(i),'tau',tau,flush=True)
    np.savez_compressed(out/'reference20ms.npz',states=refs,ids=ids,dt=dt,pressures=data['pressures'][ids],refined_tau400=taus)
    write(out/'reference_checks.json',dict(max_overlap_temperature_difference_K=max(overlap),initial_conditions=ids.tolist(),fine_dt=dt/10))
    return {int(i):r for i,r in zip(ids,refs)},{int(i):tau for i,tau in zip(ids,taus)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=Path('configs/zhang_raw_study.json'));parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());root=Path(cfg['output']);start=time.perf_counter()
    if root.exists() and not args.resume:raise FileExistsError('Use a new output or --resume')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists():assert json.loads((root/'config.json').read_text())==cfg
    write(root/'config.json',cfg)
    spent=json.loads((root/'progress.json').read_text()).get('elapsed_seconds',0.) if (root/'progress.json').exists() else 0.
    def budget():
        if spent+time.perf_counter()-start>cfg['max_seconds']:raise TimeoutError('Shared 30-minute experiment budget exceeded')
    def progress(phase):write(root/'progress.json',dict(phase=phase,elapsed_seconds=spent+time.perf_counter()-start))
    with np.load(cfg['source']) as d:data={k:d[k] for k in d.files}
    matrix=data['element_matrix'];vx,vy,_=samples(data,'val')
    input_paths=[Path(cfg['source'])]+[Path(r['training_pairs']) for r in cfg['regimes']]
    hashes={str(p):sha(p) for p in input_paths}
    if (root/'input_hashes.json').exists():assert json.loads((root/'input_hashes.json').read_text())==hashes
    write(root/'input_hashes.json',hashes)
    write(root/'protocol.json',dict(input='T_K,linear_P_atm,literal_BCT_Y9',output='delta_T_K,delta_P_atm,delta_BCT_Y9',
        normalization='separate input and output Z-score, train only',target='BCT(Ynext)-BCT(Ynow), NOT BCT(Ynext-Ynow)',
        physics=False,enthalpy_closure=False,element_projection=False,decoder_clip=False,mass_renormalization=False,
        pressure_evolution='NN pressure output is self-fed, not externally pinned',
        loss='L1 on standardized increments',hidden_layers=[64,64,64],activation='tanh',paper_gamma=.1,
        gamma_ablation=[.2,.5],paper_physics_exception='Zhang text recovers T by enthalpy; user explicitly disables this and uses all 11 NN outputs',
        sampling='independent log-space per-scale and hot two-step labels; no perturbation; matched pressure extension to .5-5 atm',
        selection='all 90000 validation one-step pairs; physical-unit score; ties ascending seed; undefined all-invalid scores treated as infinity',
        all_invalid_singleton='lowest seed retained only as deterministic failure diagnostic; not a usable best model',
        test_or_rollout_for_selection=False,finite_candidates='all finite raw outputs retained, including nonphysical values',
        inverse_domain='undefined literal BCT inverse is NaN; no signed extension',
        horizons=cfg['horizons'],numerical_termination=dict(abs_T_K=1e6,abs_Y=1e3,abs_P_Pa=1e15),
        matched_control='gamma=.1, same architecture, L1, seeds, update count, raw I/O; only training data differ'))
    frozen=[]
    try:
        for regime in cfg['regimes']:
            with np.load(regime['training_pairs']) as d:pairs={k:d[k] for k in d.files}
            if regime['name']=='multiscale':
                assert str(pairs['sampling_method'])=='independent_multiscale_log_sampling' and not bool(pairs['used_manifold_data'])
            else:assert set(pairs['parent'][:,0])<=set(np.flatnonzero(data['split']=='train'))
            x,y=pairs['x'],pairs['y'];assert float(pairs['dt'])==cfg['dt']
            metric=fit_metric(x[:,:10])
            for gamma in regime['gammas']:
                label=f'{regime["name"]}_g{gamma:g}';out=root/label
                for directory in ('models','methods','traces'):(out/directory).mkdir(parents=True,exist_ok=True)
                write(out/'distance_metric.json',metric)
                records=json.loads((out/'members.json').read_text()) if (out/'members.json').exists() else []
                for r in records:assert sha(r['path']+'.keras')==r['sha256'] and sha(r['path']+'.json')==r['metadata_sha256']
                remaining=[s for s in cfg['seeds'] if s not in {r['seed'] for r in records}]
                history=pd.read_csv(out/'history.csv').to_dict('records') if (out/'history.csv').exists() else []
                groups=pd.read_csv(out/'training_groups.csv').to_dict('records') if (out/'training_groups.csv').exists() else []
                for j in range(0,len(remaining),cfg['group_size']):
                    budget();seeds=remaining[j:j+cfg['group_size']];print('TRAIN',label,seeds,flush=True)
                    members,seconds,h=train_group(dict(gamma=gamma),seeds,cfg['updates_per_member'],x,y,cfg,start+cfg['max_seconds']-spent,True,ZhangPredictor)
                    for seed,p in zip(seeds,members):
                        path=out/'models'/f'seed{seed}'
                        if Path(str(path)+'.keras').exists():raise FileExistsError('Unregistered model')
                        p.save(path);records.append(dict(seed=seed,gamma=gamma,path=str(path),updates=cfg['updates_per_member'],
                            sha256=sha(str(path)+'.keras'),metadata_sha256=sha(str(path)+'.json')))
                        write(out/'members.json',records)
                    history.extend(h);groups.append(dict(seeds=','.join(map(str,seeds)),seconds=seconds))
                    pd.DataFrame(history).to_csv(out/'history.csv',index=False);pd.DataFrame(groups).to_csv(out/'training_groups.csv',index=False)
                    progress('training');del members;gc.collect()
                records.sort(key=lambda r:r['seed']);assert len(records)==10
                paths=[r['path'] for r in records]
                p=ZhangCommittee(paths,cfg['seeds'],metric)
                predictions=np.empty((10,len(vx),10))
                for j in range(0,len(vx),1024):
                    budget();predictions[:,j:j+1024]=p.candidates(vx[j:j+1024])[:,:,:10]
                ranking=[]
                for seed,pred in zip(cfg['seeds'],predictions):
                    with np.errstate(all='ignore'):m=evaluate(vy,pred,vx,matrix)
                    if not np.isfinite(m['selection_score']):m['selection_score']=float('inf')
                    ranking.append(dict(seed=seed,samples=len(vx),**m))
                rank=pd.DataFrame(ranking).sort_values(['selection_score','seed']);rank.to_csv(out/'validation_member_ranking.csv',index=False)
                seed=int(rank.iloc[0].seed);chosen=dict(regime=regime['name'],gamma=gamma,label=label,selected_seed=seed,
                    validation_score=float(rank.iloc[0].selection_score),usable_validation_model=bool(np.isfinite(rank.iloc[0].selection_score)),
                    model_path=paths[cfg['seeds'].index(seed)],test_used=False,rollout_used=False)
                if (out/'selection_frozen.json').exists():assert json.loads((out/'selection_frozen.json').read_text())==chosen
                write(out/'selection_frozen.json',chosen);frozen.append(chosen)
                single=ZhangCommittee([chosen['model_path']],[seed],metric,'single');single.save(out/'methods/single_best.json')
                for name,rule in [('mean10','mean'),('medoid10','medoid')]:p.rule=rule;p.save(out/'methods'/f'{name}.json')
                progress('selection');print('FROZEN',chosen,flush=True);del p,single,predictions;gc.collect()
        write(root/'all_selections_frozen.json',frozen)
        ids=np.flatnonzero(np.isin(data['split'],['val','test']))
        if (root/'reference20ms.npz').exists():
            with np.load(root/'reference20ms.npz') as d:
                refs={int(i):r for i,r in zip(d['ids'],d['states'])};taus={int(i):r for i,r in zip(d['ids'],d['refined_tau400'])}
        else:refs,taus=build_references(data,ids,cfg['dt'],max(cfg['horizons']),root,budget)
        summaries=[];details=[];species=[];one=[];times=[];checks=[]
        for definition in frozen:
            out=root/definition['label'];common=dict(regime=definition['regime'],gamma=definition['gamma'])
            for name in ('single_best','mean10','medoid10'):
                p=ZhangCommittee.load(out/'methods'/f'{name}.json')
                for split in ('val','test'):
                    budget();x,y,ids=samples(data,split);pred=np.empty_like(y)
                    for j in range(0,len(x),1024):budget();pred[j:j+1024]=p.predict(x[j:j+1024])
                    base=dict(**common,method=name,split=split)
                    with np.errstate(all='ignore'):one.append(dict(**base,samples=len(x),**evaluate(y,pred,x,matrix)))
                    truth=np.asarray([refs[int(i)] for i in ids]);pressures=data['pressures'][ids]
                    first=p.predict_full(np.c_[truth[:,0],pressures]);print('EVALUATE',base,flush=True)
                    t=time.perf_counter();traces,statuses,events=raw_rollouts(p,truth,pressures,ids,cfg['dt'],budget)
                    times.append(dict(**base,seconds=time.perf_counter()-t))
                    np.savez_compressed(out/'traces'/f'{name}_{split}.npz',**{f'trajectory_{int(i)}':a for i,a in zip(ids,traces)})
                    pd.DataFrame(events).to_csv(out/f'events_{name}_{split}.csv.gz',index=False)
                    for horizon in cfg['horizons']:
                        rows=[]
                        for i,a,reference,pressure,status in zip(ids,traces,truth,pressures,statuses):
                            r,sp=assess(a,reference,pressure,cfg['dt'],matrix,horizon,int(i),status,taus[int(i)])
                            rows.append(r);species.extend(dict(**base,horizon=horizon,trajectory=int(i),**{**s,'species':str(data['species'][s['species']])}) for s in sp)
                        summary=horizon_summary(rows);summary['median_ignition_error_refined']=float(pd.DataFrame(rows).tau400_relative_error_refined.median())
                        summaries.append(dict(**base,horizon=horizon,**summary));details.extend(dict(**base,horizon=horizon,**r) for r in rows)
                        print('RESULT',base,horizon,summary,flush=True)
                    restored=ZhangCommittee.load(out/'methods'/f'{name}.json')
                    np.testing.assert_allclose(first,restored.predict_full(np.c_[truth[:,0],pressures]),atol=1e-10,rtol=1e-9,equal_nan=True)
                    for j,a in enumerate(traces):
                        if len(a)>1:np.testing.assert_allclose(first[j],a[1],atol=1e-10,rtol=1e-9)
                        else:assert not np.isfinite(first[j]).all() or abs(first[j,0])>1e6 or np.max(np.abs(first[j,1:10]))>1e3 or abs(first[j,-1])>1e15
                    checks.append(dict(**base,reload_match=True,first_self_fed_step_match=True,physics=False))
                    for filename,values in [('summary',summaries),('trajectories',details),('species_profiles',species),('one_step',one),('timing',times),('reload_checks',checks)]:
                        pd.DataFrame(values).to_csv(root/f'{filename}.csv',index=False)
                    progress('evaluation');del restored,pred,events
                del p;gc.collect()
        assert hashes=={str(p):sha(p) for p in input_paths}
        write(root/'manifest.json',dict(status='completed',physics=False,fresh_members=40,
            represented_member_updates=400000,execution_seconds=spent+time.perf_counter()-start,input_hashes=hashes,
            source_hashes={str(p):sha(p) for p in sorted(Path('kinetics').glob('*.py'))}))
        progress('completed');print('COMPLETED',spent+time.perf_counter()-start,flush=True)
    except BaseException:progress('interrupted');raise


if __name__=='__main__':main()
