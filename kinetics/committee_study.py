"""Matched-budget original 10-member mean/medoid study, fixed train/val/test.

python -m kinetics.committee_study --config configs/committee.json
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .committee import Committee,fit_metric
from .refinement import train_updates
from .data import samples
from .posteriori import metrics
from .metrics import evaluate


def run_rollouts(p,truth,pressures,ids,matrix,dt,progress_callback=None):
    """Only truth[:,0] enters prediction; later reference states are metrics only."""
    state=truth[:,0].copy();histories=[[a.copy()] for a in state]
    active=np.ones(len(ids),bool);events=[];previous=np.full(len(ids),-1);reasons=['completed']*len(ids)
    for step in range(1,truth.shape[1]):
        live=np.flatnonzero(active)
        if not len(live):break
        x=np.c_[state[live],pressures[live]]
        pred=p.predict(x)
        diag=getattr(p,'last_diagnostics',{})
        for j,k in enumerate(live):
            out=pred[j]
            valid=bool(np.all(np.isfinite(out)) and 200<=out[0]<=6000 and np.all(out[1:]>=0) and out[1:].sum()>0)
            selected=int(diag.get('selected_seed',np.full(len(live),-1))[j])
            switched=bool(previous[k]>=0 and selected>=0 and previous[k]!=selected)
            events.append(dict(trajectory=int(ids[k]),step=step,time=step*dt,selected_seed=selected,
                switched=switched,valid_count=int(diag.get('valid_count',np.full(len(live),-1))[j]),
                spread=float(diag.get('spread',np.full(len(live),np.nan))[j]),
                central_cost=float(diag.get('central_cost',np.full(len(live),np.nan))[j]),
                correction_norm=float(diag.get('correction_norm',np.full(len(live),np.nan))[j]),
                current_temperature=float(state[k,0]),predicted_temperature=float(out[0]),accepted=valid))
            if not valid:
                active[k]=False;reasons[k]='no_valid_prediction';continue
            previous[k]=selected;state[k]=out;histories[k].append(out.copy())
        if progress_callback is not None and step%500==0:progress_callback(step,int(active.sum()))
    rows=[];species=[]
    for k,trace in enumerate(histories):
        trace=np.asarray(trace);histories[k]=trace
        row,sp=metrics(trace,truth[k],dt,matrix,truth.shape[1]-1)
        actual=np.isfinite(row['reference_tau400']);predicted=np.isfinite(row['predicted_tau400'])
        row['ignition_classification_correct']=bool(actual==predicted and (predicted or row['complete']))
        row['reference_ignites']=bool(actual);row['prediction_ignites']=bool(predicted)
        row.update(trajectory=int(ids[k]),status=reasons[k])
        rows.append(row)
        species.extend(dict(trajectory=int(ids[k]),**s) for s in sp)
    frame=pd.DataFrame(rows)
    classification=float(frame.ignition_classification_correct.mean())
    # Penalize missing/false ignition and unfinished trajectories, never prefix-only error.
    penalties=[]
    for r in rows:
        if r['reference_ignites'] and r['prediction_ignites']:v=min(r['tau400_relative_error'],5.)
        elif r['reference_ignites']!=r['prediction_ignites']:v=2.
        else:v=0.
        penalties.append(v)
    temp=np.where(frame.complete,np.minimum(frame.full_temperature_rmse.fillna(2000)/1000,5),2.)
    paired=frame.tau400_relative_error.dropna()
    summary=dict(completion_fraction=float(frame.complete.mean()),physical_fraction=float(frame.physically_admissible_complete.mean()),
        ignition_classification=classification,ignition_pairs=int(len(paired)),median_ignition_relative_error=float(paired.median()),
        mean_ignition_relative_error=float(paired.mean()),temperature_profile_mean=float(frame.full_temperature_rmse.mean()),
        temperature_profile_pooled=float(np.sqrt(np.nanmean(frame.full_temperature_rmse**2))),
        species_profile_mean=float(frame.full_species_rmse.mean()),mass_sum_max=float(frame.mass_sum_max.max()),elemental_max=float(frame.elemental_max.max()))
    summary['score']=5*(1-summary['completion_fraction'])+3*(1-summary['physical_fraction'])+3*(1-classification)+float(np.mean(penalties))+float(np.mean(temp))
    return summary,rows,species,events,histories


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/committee.json');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(cfg['output'])
    if out.exists():raise FileExistsError('Use a new output; preserve all previous experiments')
    out.mkdir();(out/'models').mkdir();(out/'traces').mkdir();(out/'methods').mkdir()
    started=time.perf_counter();deadline=started+cfg['max_seconds']
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    data=np.load(cfg['source']);x,y,ids_train=samples(data,'train')
    if float(data['dt'])!=cfg['dt']:raise ValueError('inconsistent timestep')
    cfg_train={**cfg};paths=[];model_records=[];history=[]
    spec=dict(name='boxcox_g0.1',kind='boxcox',gamma=cfg['gamma'])
    for seed in cfg['seeds']:
        reused=seed in cfg['reuse_seeds']
        if reused:
            path=Path('results/refinement_controls/models')/f'final_original_boxcox_g0.1_seed{seed}'
            if not Path(str(path)+'.keras').exists():raise FileNotFoundError(path)
            seconds=0.
        else:
            p,seconds,h=train_updates(spec,seed,cfg['updates_per_member'],x,y,cfg_train,deadline)
            path=out/'models'/f'original_boxcox_g0.1_seed{seed}';p.save(path)
            history.extend(dict(seed=seed,**r) for r in h)
        paths.append(str(path))
        model_records.append(dict(seed=seed,path=str(path),reused=reused,updates=cfg['updates_per_member'],
            train_seconds=seconds,sha256=hashlib.sha256(Path(str(path)+'.keras').read_bytes()).hexdigest()))
        (out/'members.json').write_text(json.dumps(model_records,indent=2))
        pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
        print('MEMBER',seed,'reused',reused,'seconds',round(seconds,2),flush=True)
    metric=fit_metric(x[:,:10],cfg['species_distance_floor']);(out/'distance_metric.json').write_text(json.dumps(metric,indent=2))
    seeds=cfg['seeds'];single=seeds.index(cfg['single_reference_seed'])
    definitions=[dict(name='single_original_seed33',rule='single',indices=[single],distance='scaled'),
        dict(name='mean3_before_repair',rule='mean_before_repair',indices=[0,1,2],distance='scaled'),
        dict(name='mean10_before_repair',rule='mean_before_repair',indices=list(range(10)),distance='scaled'),
        dict(name='mean10_after_repair',rule='mean_after_repair',indices=list(range(10)),distance='scaled'),
        dict(name='medoid10_scaled',rule='medoid',indices=list(range(10)),distance='scaled'),
        dict(name='medoid10_softlog',rule='medoid',indices=list(range(10)),distance='softlog')]
    pd.DataFrame(model_records).to_csv(out/'training_cost.csv',index=False)
    valref=np.load('results/refinement/validation_reference.npz')
    summaries=[];all_rows=[];all_species=[];event_frames=[];one_step=[];noise_rows=[];timings=[];confidence=[]
    source_species=data['species'];matrix=data['element_matrix'];dt=cfg['dt']
    methods={}
    for definition in definitions:
        indices=definition['indices'];p=Committee([paths[i] for i in indices],[seeds[i] for i in indices],metric,definition['rule'],definition['distance'])
        p.save(out/'methods'/f'{definition["name"]}.json');methods[definition['name']]=p
    def assess(name,split,ref):
        if time.perf_counter()>deadline:raise TimeoutError('committee study budget exhausted')
        p=methods[name]
        # Warm compiled graph before measuring simulation; includes physics warm-up.
        p.predict(np.c_[ref['states'][:,0],ref['pressures']])
        p.timing={k:0. for k in p.timing}
        began=time.perf_counter()
        result,rows,sp,events,traces=run_rollouts(p,ref['states'],ref['pressures'],ref['ids'],matrix,dt)
        seconds=time.perf_counter()-began
        timings.append(dict(method=name,split=split,wall_seconds=seconds,**p.timing))
        if split=='test':
            fine=pd.read_csv('results/verified/posteriori/reference_ignition_refined.csv').set_index('trajectory')
            for r in rows:
                actual=float(fine.loc[r['trajectory'],'tau400']);actual_peak=float(fine.loc[r['trajectory'],'tau_dTdt'])
                predicted=r['predicted_tau400'];predicted_peak=r['predicted_tau_dTdt']
                r['reference_tau400_refined']=actual
                r['tau400_relative_error_refined']=abs(predicted-actual)/actual if np.isfinite(predicted) and np.isfinite(actual) else float('nan')
                r['tau_peak_relative_error_refined']=abs(predicted_peak-actual_peak)/actual_peak if np.isfinite(predicted_peak) and np.isfinite(actual_peak) else float('nan')
            f=pd.DataFrame(rows)
            result['median_ignition_relative_error_refined']=float(f.tau400_relative_error_refined.median())
            result['median_peak_relative_error_refined']=float(f.tau_peak_relative_error_refined.median())
        summary=dict(method=name,split=split,rollout_seconds=seconds,**result);summaries.append(summary)
        all_rows.extend(dict(method=name,split=split,**r) for r in rows)
        all_species.extend(dict(method=name,split=split,**{**r,'species':str(source_species[r['species']])}) for r in sp)
        ef=pd.DataFrame(events);ef['method']=name;ef['split']=split;event_frames.append(ef)
        ef.to_csv(out/f'selection_{name}_{split}.csv',index=False)
        np.savez_compressed(out/'traces'/f'{name}_{split}.npz',**{f'trajectory_{int(i)}':trace for i,trace in zip(ref['ids'],traces)})
        sx,sy,_=samples(data,split)
        pred=p.predict(sx);diag={k:v.copy() for k,v in p.last_diagnostics.items()}
        m=evaluate(sy,pred,sx,matrix);one_step.append(dict(method=name,split=split,**m))
        # Confidence check on validation reference states only; not used for fallback.
        if split=='val' and definition_by_name[name]['rule'] in ('medoid','mean_after_repair'):
            error=np.sqrt(np.mean(((pred[:,1:]-sy[:,1:])/np.asarray(metric['species_scale']))**2,axis=1))
            spread=diag['spread'];mask=np.isfinite(error)&np.isfinite(spread)
            corr=spearmanr(spread[mask],error[mask])
            confidence.append(dict(method=name,states=int(mask.sum()),spearman_spread_error=float(corr.statistic),pvalue=float(corr.pvalue)))
            pd.DataFrame(dict(spread=spread,scaled_species_error=error,valid_count=diag['valid_count'])).to_csv(out/f'validation_confidence_{name}.csv',index=False)
        # Identical predetermined subset and random noise, no perturbation tuning.
        subset=sx[::12];clean=p.predict(subset);rng=np.random.default_rng(100)
        for amp in (1e-16,1e-12,1e-8):
            noisy=subset.copy();noisy[:,1:9]=np.maximum(noisy[:,1:9]+rng.normal(size=(len(subset),8))*amp,0);noisy[:,9]=1-noisy[:,1:9].sum(axis=1)
            changed=p.predict(noisy);delta=changed[:,1:]-clean[:,1:]
            noise_rows.append(dict(method=name,split=split,noise_std=amp,output_change_rmse=float(np.sqrt(np.nanmean(delta**2))),invalid_fraction=float(np.mean(~np.all(np.isfinite(changed),axis=1)))))
        for filename,values in [('summary.csv',summaries),('trajectories.csv',all_rows),('species_profiles.csv',all_species),
                                ('one_step.csv',one_step),('noise.csv',noise_rows),('timing.csv',timings),('validation_confidence.csv',confidence)]:
            pd.DataFrame(values).to_csv(out/filename,index=False)
        print('RESULT',summary,flush=True)
    definition_by_name={d['name']:d for d in definitions}
    for d in definitions:assess(d['name'],'val',valref)
    val=pd.DataFrame(summaries);rank=val[val.method.str.startswith('medoid')].sort_values(['score','method'])
    selected=str(rank.iloc[0].method)
    frozen=dict(selected_medoid=selected,criterion='validation 0-D score only',default_data='matched-budget original; no augmentation',
        definitions=definitions,distance_fit='training only',test_used_for_selection=False,ode_fallback=False,hysteresis=False,
        correction_order='medoid: repair each then select; primary mean comparator: repair each then average and recover enthalpy; legacy means: average then repair')
    (out/'selection_frozen.json').write_text(json.dumps(frozen,indent=2))
    # Held-out ICs are only loaded after metric selection is frozen.
    testref=np.load('results/verified/posteriori/cantera_reference.npz')
    for name in ['single_original_seed33','mean3_before_repair','mean10_before_repair','mean10_after_repair',selected]:assess(name,'test',testref)
    # Round-trip serialized committee predictions and medoid seed identities.
    checks=[]
    xx,_,_=samples(data,'val')
    for d in definitions:
        restored=Committee.load(out/'methods'/f'{d["name"]}.json');p=methods[d['name']]
        a=p.predict(xx[:30]);selected_a=p.last_diagnostics['selected_seed'].copy();b=restored.predict(xx[:30])
        np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-11,equal_nan=True)
        np.testing.assert_array_equal(selected_a,restored.last_diagnostics['selected_seed'])
        checks.append(dict(method=d['name'],prediction_reload=True,selection_reload=True))
    pd.DataFrame(checks).to_csv(out/'reload_checks.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(status='completed',wall_seconds=time.perf_counter()-started,
        data_sha256=hashlib.sha256(Path(cfg['source']).read_bytes()).hexdigest(),train_ids=ids_train.tolist(),
        total_updates_represented=cfg['updates_per_member']*len(seeds),new_updates=cfg['updates_per_member']*(len(seeds)-len(cfg['reuse_seeds'])),
        augmented=False,dt=dt,teacher_forcing=False,fallback_to_ode=False,selected_medoid=selected),indent=2))
    print('COMPLETE',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
