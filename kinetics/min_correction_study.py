"""Frozen ten-member ensembles, selecting the least-corrected candidate per step."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .committee import Committee
from .committee_study import run_rollouts
from .data import samples
from .long_committee_study import horizon_summary
from .metrics import evaluate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=Path('results/preprocess_tanh10'))
    parser.add_argument('--output',type=Path,default=Path('results/min_correction_tanh10'))
    args=parser.parse_args();root=args.output;old=args.baseline
    if root.exists():raise FileExistsError('Preserve results; choose a new output')
    root.mkdir(parents=True)
    cfg=json.loads((old/'config.json').read_text())
    config={**cfg,'output':str(root),'baseline':str(old),'training':False,'rule':'min_correction'}
    (root/'config.json').write_text(json.dumps(config,indent=2))
    started=time.perf_counter()
    def budget():
        if time.perf_counter()-started>1800:raise TimeoutError('30-minute experiment cap exceeded')
    protocol=dict(rule='min_correction',fixed_before_evaluation=True,
        norm='sqrt(((T_corrected-T_ANN)/sT)^2 + sum_j((Y_corrected_j-Y_ANN_j)/sY_j)^2)/sqrt(10)',
        temperature_scale='max(training input T standard deviation,100 K)',
        species_scale='max(training input Y 95th percentile,1e-8), each species separately',
        raw_definition='ANN next-state proposal after the existing common transformed [0,1] decoder guard',
        temperature_head='ANN T is included in the correction norm, but corrected T is recovered from conserved enthalpy',
        comparison='corrected next state minus ANN next-state proposal; not next state minus current state',
        invalid='exclude nonfinite or rejected corrections; all invalid stops trajectory',
        ties='first member in ascending seed order',same_current_state_for_all_members=True,
        select='one actual corrected candidate per IC per timestep',test_used_for_tuning=False,
        retraining=False,ode_fallback=False,hysteresis=False,physics_backend='gamma-matched Wang',
        common_control=json.loads((old/'global_single_frozen.json').read_text()))
    (root/'selection_protocol.json').write_text(json.dumps(protocol,indent=2))
    paths=[Path(cfg['source']),Path(cfg['training_pairs']),old/'summary.csv',old/'one_step.csv',old/'global_single_frozen.json']
    model_records={}
    for gamma in cfg['gammas']:
        out=root/f'g{gamma:g}';(out/'methods').mkdir(parents=True);(out/'traces').mkdir()
        records=json.loads((old/f'g{gamma:g}'/'members.json').read_text());model_records[gamma]=records
        assert len(records)==10
        for r in records:
            assert digest(r['path']+'.keras')==r['sha256']
            assert digest(r['path']+'.json')==r['metadata_sha256']
            paths.extend([Path(r['path']+'.keras'),Path(r['path']+'.json')])
    hashes={str(p):digest(p) for p in paths}
    (root/'input_hashes.json').write_text(json.dumps(hashes,indent=2))
    with np.load(cfg['source']) as archive:data={k:archive[k] for k in archive.files}
    matrix=data['element_matrix'];dt=float(data['dt'])
    assert dt==cfg['dt'] and np.isclose((data['states'].shape[1]-1)*dt,cfg['horizon'])
    summaries=[];details=[];species=[];timing=[];diagnostics=[];one_steps=[];checks=[];usage=[]
    for gamma in cfg['gammas']:
        out=root/f'g{gamma:g}'
        p=Committee.load(old/f'g{gamma:g}'/'methods/medoid10.json')
        assert len(p.seeds)==10 and np.all(np.diff(p.seeds)>0)
        assert p.vector_physics.gamma==gamma and p.metric['fit_split']=='train'
        p.rule='min_correction';p.save(out/'methods/min_correction10.json')
        for split in ('val','test'):
            budget();common=dict(gamma=gamma,method='min_correction10',split=split,horizon=cfg['horizon'])
            x,y,ids=samples(data,split);pred=np.empty_like(y)
            for begin in range(0,len(x),1024):
                budget();pred[begin:begin+1024]=p.predict(x[begin:begin+1024])
            one_steps.append(dict(**common,samples=len(x),**evaluate(y,pred,x,matrix)))
            x0=np.c_[data['states'][ids,0],data['pressures'][ids]]
            raw=p.proposals(x0);corrected=p.repair(raw,x0)
            scale=np.r_[p.metric['temperature_scale'],p.metric['species_scale']]
            costs=np.sqrt(np.mean(((corrected-raw)/scale)**2,axis=-1))
            costs=np.where(np.isfinite(costs),costs,np.inf)
            expected=np.argmin(costs,axis=0)
            live=np.isfinite(costs.min(axis=0))
            first=p.predict(x0);first_seeds=p.last_diagnostics['selected_seed'].copy()
            np.testing.assert_allclose(first[live],corrected[expected[live],np.flatnonzero(live)],rtol=1e-10,atol=1e-10)
            np.testing.assert_array_equal(first_seeds[live],p.seeds[expected[live]])
            assert np.isnan(first[~live]).all()
            p.timing={k:0. for k in p.timing};p.vector_physics.counts={k:0 for k in p.vector_physics.counts}
            begin=time.perf_counter()
            def progress(step,active):
                budget();print('ROLLOUT',gamma,split,step,active,flush=True)
            _,rows,sp,events,traces=run_rollouts(p,data['states'][ids],data['pressures'][ids],ids,matrix,dt,progress)
            timing.append(dict(**common,seconds=time.perf_counter()-begin,**p.timing))
            diagnostics.append(dict(**common,**p.vector_physics.counts))
            np.savez_compressed(out/'traces'/f'min_correction10_{split}.npz',**{f'trajectory_{int(i)}':t for i,t in zip(ids,traces)})
            ef=pd.DataFrame(events);ef.to_csv(out/f'selection_{split}.csv.gz',index=False)
            accepted=ef[ef.accepted]
            assert np.isfinite(accepted.correction_norm).all() and (accepted.correction_norm>=0).all()
            for seed in p.seeds:
                use=accepted[accepted.selected_seed==seed]
                usage.append(dict(**common,seed=int(seed),selected_steps=len(use),
                    fraction=len(use)/max(len(accepted),1),mean_correction_norm=float(use.correction_norm.mean())))
            fine=pd.read_csv(Path(cfg['reference_root'])/f'reference_ignition_{split}.csv').set_index('trajectory')
            for row in rows:
                tau=float(fine.loc[row['trajectory'],'tau400'])
                row['reference_tau400_refined']=tau
                row['tau400_relative_error_refined']=abs(row['predicted_tau400']-tau)/tau if np.isfinite(tau) and row['prediction_ignites'] else np.nan
                ee=ef[ef.trajectory==row['trajectory']]
                row['member_switches']=int(ee.switched.sum())
                row['unique_selected_members']=int(ee[ee.accepted].selected_seed.nunique())
                row['mean_correction_norm']=float(ee[ee.accepted].correction_norm.mean())
            summary=horizon_summary(rows)
            summary['median_ignition_error_refined']=float(pd.DataFrame(rows).tau400_relative_error_refined.median())
            summary['switches']=int(ef.switched.sum())
            summaries.append(dict(**common,**summary));details.extend(dict(**common,**r) for r in rows)
            species.extend(dict(**common,**{**r,'species':str(data['species'][r['species']])}) for r in sp)
            restored=Committee.load(out/'methods/min_correction10.json')
            np.testing.assert_allclose(first,restored.predict(x0),rtol=1e-9,atol=1e-10,equal_nan=True)
            np.testing.assert_array_equal(first_seeds,restored.last_diagnostics['selected_seed'])
            np.testing.assert_allclose(first,np.array([t[1] if len(t)>1 else np.full(10,np.nan) for t in traces]),rtol=1e-9,atol=1e-10,equal_nan=True)
            checks.append(dict(**common,minimal_correction_first_step=True,actual_candidate=True,reload_match=True,first_self_fed_step_match=True))
            for name,values in [('summary',summaries),('trajectories',details),('species_profiles',species),('one_step',one_steps),
                    ('timing',timing),('correction_diagnostics',diagnostics),('selection_usage',usage),('reload_checks',checks)]:
                pd.DataFrame(values).to_csv(root/f'{name}.csv',index=False)
            print('RESULT',common,summary,flush=True)
            del restored,pred,events,ef
        del p;gc.collect()
    for filename in ('summary','one_step'):
        baseline=pd.read_csv(old/f'{filename}.csv')
        current=pd.read_csv(root/f'{filename}.csv')
        pd.concat([baseline,current],ignore_index=True).to_csv(root/f'comparison_{filename}.csv',index=False)
    assert hashes=={str(p):digest(p) for p in paths}
    (root/'manifest.json').write_text(json.dumps(dict(status='completed',no_retraining=True,execution_seconds=time.perf_counter()-started,
        original_inputs_unchanged=True,verified_method_split_pairs=4,
        source_hashes={str(p):digest(p) for p in sorted(Path('kinetics').glob('*.py'))}),indent=2))
    print('COMPLETED',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
