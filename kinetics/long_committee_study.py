"""100 independently trained members on equilibrium-thinned 10 ms chemistry.

Preparation is separate from expensive execution. Default runtime budget is
1800 s. Use --budget-seconds 5400 only after explicit long-run authorization.
Completed training groups and completed evaluations can be resumed safely.
"""
import argparse,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .batched_training import train_group
from .committee import Committee,fit_metric
from .committee_study import run_rollouts
from .posteriori import metrics,ignition
from .data import mechanism
from .metrics import evaluate
import cantera as ct


def refine_reference(data,ids,destination,dt):
    destination=Path(destination)
    if destination.exists():return pd.read_csv(destination)
    fine_dt=dt/10;steps=round((data['states'].shape[1]-1)*dt/fine_dt);rows=[]
    for i in ids:
        initial=data['states'][i,0];gas=mechanism();gas.TPY=initial[0],data['pressures'][i],initial[1:]
        r=ct.IdealGasConstPressureReactor(gas,clone=True);net=ct.ReactorNet([r]);net.rtol=1e-10;net.atol=1e-20
        temperatures=np.empty(steps+1);temperatures[0]=initial[0]
        for j in range(1,steps+1):net.advance(j*fine_dt);temperatures[j]=r.T
        rows.append(dict(trajectory=int(i),reference_dt=fine_dt,**ignition(temperatures[:,None],fine_dt,initial[0])))
    frame=pd.DataFrame(rows);frame.to_csv(destination,index=False);return frame


def horizon_summary(rows):
    d=pd.DataFrame(rows);paired=d.tau400_relative_error.dropna()
    classification=float(d.ignition_classification_correct.mean())
    return dict(trajectories=len(d),completed=int(d.complete.sum()),physical_completed=int(d.physically_admissible_complete.sum()),
        completion_fraction=float(d.complete.mean()),physical_fraction=float(d.physically_admissible_complete.mean()),
        ignition_classification=classification,ignition_pairs=len(paired),median_ignition_error=float(paired.median()),
        mean_ignition_error=float(paired.mean()),temperature_profile_mean=float(d.full_temperature_rmse.mean()),
        temperature_profile_pooled=float(np.sqrt(np.nanmean(d.full_temperature_rmse**2))),species_profile_mean=float(d.full_species_rmse.mean()),
        mass_sum_max=float(d.mass_sum_max.max()),elemental_max=float(d.elemental_max.max()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/committee100_10ms.json');parser.add_argument('--budget-seconds',type=int,default=1800)
    parser.add_argument('--phase',choices=['train','evaluate','all'],default='all');parser.add_argument('--resume',action='store_true');parser.add_argument('--train-limit',type=int);args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(cfg['output']);began=time.perf_counter()
    if out.exists() and not args.resume:raise FileExistsError('Use --resume to verify and continue, or choose a new output')
    out.mkdir(exist_ok=True);(out/'models').mkdir(exist_ok=True);(out/'methods').mkdir(exist_ok=True);(out/'traces').mkdir(exist_ok=True)
    cfg_path=out/'config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text())!=cfg:raise ValueError('Configuration changed during resume')
    cfg_path.write_text(json.dumps(cfg,indent=2))
    pair_data=np.load(cfg['training_pairs']);x=pair_data['x'];y=pair_data['y']
    dense=np.load(cfg['source']);data={k:dense[k] for k in dense.files}
    if float(pair_data['dt'])!=cfg['dt'] or float(data['dt'])!=cfg['dt']:raise ValueError('dt mismatch')
    if not np.isclose((data['states'].shape[1]-1)*cfg['dt'],cfg['horizon']):raise ValueError('reference horizon mismatch')
    if not set(pair_data['parent'][:,0])<=set(np.flatnonzero(data['split']=='train')):raise ValueError('split leakage')
    pair_hash=hashlib.sha256(Path(cfg['training_pairs']).read_bytes()).hexdigest()
    progress_file=out/'progress.json'
    old=json.loads(progress_file.read_text()) if progress_file.exists() else dict(elapsed_seconds=0.,pair_sha256=pair_hash)
    if old['pair_sha256']!=pair_hash:raise ValueError('Training data changed during resume')
    spent=float(old['elapsed_seconds']);budget=min(args.budget_seconds,cfg['max_seconds']);deadline=began+max(0,budget-spent)
    records=json.loads((out/'members.json').read_text()) if (out/'members.json').exists() else []
    def persist(phase):
        state=dict(phase=phase,completed_members=len(records),target_members=len(cfg['seeds']),elapsed_seconds=spent+time.perf_counter()-began,
                   pair_sha256=pair_hash,budget_seconds=budget)
        progress_file.write_text(json.dumps(state,indent=2))
    try:
        for record in records:
            if hashlib.sha256(Path(record['path']+'.keras').read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Saved model hash mismatch')
        if args.phase in ('train','all'):
            remaining=[s for s in cfg['seeds'] if s not in {r['seed'] for r in records}]
            if args.train_limit is not None:remaining=remaining[:args.train_limit]
            spec=dict(name='boxcox_g0.1',kind='boxcox',gamma=cfg['gamma'])
            groups=pd.read_csv(out/'training_groups.csv').to_dict('records') if (out/'training_groups.csv').exists() else []
            histories=pd.read_csv(out/'history.csv').to_dict('records') if (out/'history.csv').exists() else []
            for start in range(0,len(remaining),cfg['group_size']):
                seeds=remaining[start:start+cfg['group_size']]
                print('GROUP_START',seeds,'complete',len(records),flush=True);persist('training')
                members,seconds,h=train_group(spec,seeds,cfg['updates_per_member'],x,y,cfg,deadline,progress=True)
                for seed,member in zip(seeds,members):
                    path=out/'models'/f'long10ms_boxcox_g0.1_seed{seed}'
                    if Path(str(path)+'.keras').exists():raise FileExistsError('Unregistered model file; preserve before resume')
                    member.save(path)
                    record=dict(seed=seed,path=str(path),updates=cfg['updates_per_member'],train_data_sha256=pair_hash,
                                sha256=hashlib.sha256(Path(str(path)+'.keras').read_bytes()).hexdigest(),group_seconds=seconds,group_size=len(seeds))
                    records.append(record);(out/'members.json').write_text(json.dumps(records,indent=2))
                groups.append(dict(seeds=','.join(map(str,seeds)),members=len(seeds),updates_per_member=cfg['updates_per_member'],seconds=seconds))
                histories.extend(h);pd.DataFrame(groups).to_csv(out/'training_groups.csv',index=False);pd.DataFrame(histories).to_csv(out/'history.csv',index=False)
                persist('training');print('GROUP_COMPLETE',len(records),round(seconds,2),flush=True)
                del members;gc.collect()
            persist('training_complete' if len(records)==len(cfg['seeds']) else 'training_partial')
        if args.phase=='train':return
        if len(records)!=len(cfg['seeds']):raise ValueError('All members must be trained before full evaluation')
        paths_by_seed={r['seed']:r['path'] for r in records};metric=fit_metric(x[:,:10],cfg['species_distance_floor'])
        physics_backend=cfg.get('physics_backend','vectorized_nasa')
        (out/'evaluation_config.json').write_text(json.dumps(dict(physics_backend=physics_backend,
            thermodynamics='mechanism NASA-7 ideal-gas enthalpy, checked against Cantera; scalar fallback for ambiguous cases',
            target_dt=cfg['dt'],horizon=cfg['horizon']),indent=2))
        (out/'distance_metric.json').write_text(json.dumps(metric,indent=2))
        definitions=[dict(name=f'{rule}{size}',size=size,rule='medoid' if rule=='medoid' else 'mean_after_repair')
                     for size in cfg['comparison_sizes'] for rule in ['mean','medoid']]
        # No test-driven tuning: member count and scaled distance were specified before this run.
        (out/'selection_frozen.json').write_text(json.dumps(dict(default_method='medoid100',definitions=definitions,distance='scaled',
            metric_fit='new training pairs only',test_used_for_selection=False,ode_fallback=False,hysteresis=False),indent=2))
        all_summary=pd.read_csv(out/'summary.csv').to_dict('records') if (out/'summary.csv').exists() else []
        all_details=pd.read_csv(out/'trajectories.csv').to_dict('records') if (out/'trajectories.csv').exists() else []
        all_species=pd.read_csv(out/'species_profiles.csv').to_dict('records') if (out/'species_profiles.csv').exists() else []
        one_steps=pd.read_csv(out/'one_step_probes.csv').to_dict('records') if (out/'one_step_probes.csv').exists() else []
        timing=pd.read_csv(out/'timing.csv').to_dict('records') if (out/'timing.csv').exists() else []
        cache={}
        for split in ('val','test'):
            ids=np.flatnonzero(data['split']==split);states=data['states'][ids];pressures=data['pressures'][ids]
            fine=refine_reference(data,ids,out/f'reference_ignition_{split}.csv',cfg['dt']).set_index('trajectory')
            for definition in definitions:
                name=definition['name'];size=definition['size']
                if any(r['method']==name and r['split']==split and np.isclose(r['horizon'],cfg['horizon']) for r in all_summary):continue
                if time.perf_counter()>deadline:raise TimeoutError('Evaluation budget exhausted')
                if size not in cache:
                    seeds=cfg['seeds'][:size]
                    cache[size]=Committee([paths_by_seed[s] for s in seeds],seeds,metric,collect_mean_diagnostics=False,physics_backend=physics_backend)
                p=cache[size];p.rule=definition['rule'];p.distance='scaled'
                p.save(out/'methods'/f'{name}.json')
                p.predict(np.c_[states[:,0],pressures]);p.timing={k:0. for k in p.timing}
                print('EVALUATION_START',split,name,'steps',states.shape[1]-1,flush=True);start=time.perf_counter()
                def report_step(step,active):print('EVAL_PROGRESS',split,name,step,'active',active,flush=True)
                _,_,_,events,traces=run_rollouts(p,states,pressures,ids,data['element_matrix'],cfg['dt'],report_step)
                seconds=time.perf_counter()-start
                timing.append(dict(method=name,split=split,seconds=seconds,**p.timing))
                np.savez_compressed(out/'traces'/f'{name}_{split}.npz',**{f'trajectory_{int(i)}':t for i,t in zip(ids,traces)})
                pd.DataFrame(events).to_csv(out/f'selection_{name}_{split}.csv',index=False)
                for horizon in (.002,cfg['horizon']):
                    rr=[];ss=[]
                    for k,(i,trace) in enumerate(zip(ids,traces)):
                        row,species=metrics(trace,states[k],cfg['dt'],data['element_matrix'],round(horizon/cfg['dt']))
                        actual=np.isfinite(row['reference_tau400']);predicted=np.isfinite(row['predicted_tau400'])
                        row.update(trajectory=int(i),reference_ignites=bool(actual),prediction_ignites=bool(predicted),
                                   ignition_classification_correct=bool(actual==predicted and (predicted or row['complete'])))
                        tau=float(fine.loc[int(i),'tau400']);tau=tau if tau<=horizon else float('nan')
                        row['reference_tau400_refined']=tau
                        row['tau400_relative_error_refined']=abs(row['predicted_tau400']-tau)/tau if np.isfinite(tau) and predicted else float('nan')
                        rr.append(row);ss.extend(dict(trajectory=int(i),**{**r,'species':str(data['species'][r['species']])}) for r in species)
                    summary=horizon_summary(rr);summary['median_ignition_error_refined']=float(pd.DataFrame(rr).tau400_relative_error_refined.median())
                    common=dict(method=name,split=split,horizon=horizon)
                    all_summary.append(dict(**common,**summary));all_details.extend(dict(**common,**r) for r in rr);all_species.extend(dict(**common,**r) for r in ss)
                    print('LONG_RESULT',common,summary,flush=True)
                # Limited, predeclared one-step probe; full autoregressive profiles are NOT thinned.
                j=np.unique(np.r_[np.arange(8),np.linspace(0,states.shape[1]-2,cfg['one_step_probe_per_trajectory']-8).astype(int)])
                px=np.c_[states[:,j].reshape(-1,10),np.repeat(pressures,len(j))];py=states[:,j+1].reshape(-1,10)
                pred=p.predict(px);one_steps.append(dict(method=name,split=split,probe_count=len(px),**evaluate(py,pred,px,data['element_matrix'])))
                for filename,rows in [('summary.csv',all_summary),('trajectories.csv',all_details),('species_profiles.csv',all_species),('one_step_probes.csv',one_steps),('timing.csv',timing)]:pd.DataFrame(rows).to_csv(out/filename,index=False)
                persist('evaluating')
        persist('completed')
        (out/'manifest.json').write_text(json.dumps(dict(status='completed',members=len(records),training_pairs=len(x),horizon=cfg['horizon'],dt=cfg['dt'],
            train_pair_sha256=pair_hash,dense_sha256=hashlib.sha256(Path(cfg['source']).read_bytes()).hexdigest(),
            represented_member_updates=cfg['updates_per_member']*len(records),total_execution_seconds=spent+time.perf_counter()-began),indent=2))
        print('LONG_STUDY_COMPLETE',spent+time.perf_counter()-began,flush=True)
    except BaseException:
        persist('interrupted');raise


if __name__=='__main__':main()
