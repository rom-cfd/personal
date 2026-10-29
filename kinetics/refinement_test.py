"""Freeze validation choice, then evaluate fixed held-out ICs exactly once.

Requires a validation family passing the agreed ignition criterion. Never used
for choosing perturbation strength, number, optimization steps or model seed.
"""
import argparse,hashlib,json,time
from pathlib import Path
import cantera as ct
import numpy as np
import pandas as pd
from .data import samples,mechanism
from .study import Predictor
from .physics import PhysicalPredictor
from .ensemble import load_candidate
from .refinement import posteriori
from .metrics import evaluate,distributions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='results/refinement');parser.add_argument('--controls',default='results/refinement_controls');parser.add_argument('--output',default='results/refinement_test');args=parser.parse_args()
    root=Path(args.root);controls_root=Path(args.controls);out=Path(args.output)
    cfg=json.loads((root/'config.json').read_text())
    min_class=cfg.get('goal_ignition_classification',.9)
    max_error=cfg.get('goal_median_ignition_relative_error',.2)
    if out.exists():raise FileExistsError('Do not rerun held-out evaluation into an existing directory')
    augmented=json.loads((root/'final_models.json').read_text())
    controls=json.loads((controls_root/'final_models.json').read_text())
    ensembles=json.loads((root/'ensemble_validation/models.json').read_text())
    entries=augmented+controls+ensembles
    validation=pd.concat([pd.read_csv(root/'validation_summary.csv'),pd.read_csv(controls_root/'validation_summary.csv'),
                          pd.read_csv(root/'ensemble_validation/validation_summary.csv')],ignore_index=True)
    detail=pd.read_csv(root/'validation_trajectories.csv')
    final=detail[(detail.phase=='final')&(detail['mode']=='physics')]
    family=final.groupby('name').agg(classification=('ignition_classification_correct','mean'),
        median_error=('tau400_relative_error','median'),physical_fraction=('physically_admissible_complete','mean'))
    passing=family[(family.classification>=min_class)&(family.median_error<=max_error)&(family.physical_fraction>=.95)]
    if passing.empty:raise RuntimeError('Validation goal not met; continue validation-only refinement before test access')
    eligible=validation[(validation.phase.isin(['final','ensemble']))&(validation['mode']=='physics')&
         (validation.ignition_classification>=min_class)&(validation.median_ignition_relative_error<=max_error)&(validation.physical_fraction>=.95)].sort_values('score')
    best=eligible.iloc[0]
    primary=next(e for e in entries if e['name']==best['name'] and e['seed']==int(best['seed']))
    out.mkdir();(out/'traces').mkdir()
    frozen=dict(primary=primary,primary_mode='physics',selection='minimum validation a posteriori score among goal-passing candidates; before test access',
                family_validation=family.reset_index().to_dict('records'),entries=entries,test_used_for_selection=False)
    (out/'selection_frozen.json').write_text(json.dumps(frozen,indent=2))
    # Test data are only opened after the validation-selected descriptor is persisted.
    source=np.load('results/verified/dataset.npz');ref=np.load('results/verified/posteriori/cantera_reference.npz')
    fine=pd.read_csv('results/verified/posteriori/reference_ignition_refined.csv').set_index('trajectory')
    x,y,_=samples(source,'test');matrix=source['element_matrix'];dt=float(source['dt']);species=source['species']
    start=time.perf_counter();summary=[];details=[];species_rows=[];one_step=[];one_bins=[];noise=[];energy=[];reloads=[]
    gas=mechanism()
    for entry in entries:
        p=load_candidate(entry);name=entry['name'];seed=entry['seed'];tag=f'{name}_seed{seed}'
        for mode in ('raw','physics'):
            began=time.perf_counter()
            result,rows,sp,traces=posteriori(p,ref['states'],ref['pressures'],ref['ids'],matrix,dt,mode)
            for row in rows:
                ref_tau=float(fine.loc[row['trajectory'],'tau400']);pred_tau=row['predicted_tau400']
                row['reference_tau400_refined']=ref_tau
                row['tau400_relative_error_refined']=abs(pred_tau-ref_tau)/ref_tau if np.isfinite(ref_tau) and np.isfinite(pred_tau) else float('nan')
                tr=float(fine.loc[row['trajectory'],'tau_dTdt']);tp=row['predicted_tau_dTdt']
                row['tau_dTdt_relative_error_refined']=abs(tp-tr)/tr if np.isfinite(tr) and np.isfinite(tp) else float('nan')
            details.extend(dict(name=name,seed=seed,**r) for r in rows)
            species_rows.extend(dict(name=name,seed=seed,**{**r,'species':str(species[r['species']])}) for r in sp)
            saved={f'trajectory_{int(ref["ids"][i])}':trace for i,trace in traces}
            np.savez_compressed(out/'traces'/f'{tag}_{mode}.npz',**saved)
            for i,trace in traces:
                gas.TPY=trace[0,0],ref['pressures'][i],trace[0,1:];h0=gas.enthalpy_mass
                drift=[]
                for state in trace:
                    if state[1:].sum()>0 and np.all(np.isfinite(state)):
                        # Raw mixture enthalpy uses raw Y, without implicit normalization.
                        gas.TP=state[0],ref['pressures'][i]
                        hs=gas.standard_enthalpies_RT*ct.gas_constant*state[0]/gas.molecular_weights
                        drift.append(float(state[1:]@hs-h0))
                energy.append(dict(name=name,seed=seed,mode=mode,trajectory=int(ref['ids'][i]),max_abs_enthalpy_drift=float(np.max(np.abs(drift)))))
            model=PhysicalPredictor(p) if mode=='physics' else p
            began_one=time.perf_counter();pred=model.predict(x);one_seconds=time.perf_counter()-began_one
            np.savez_compressed(out/f'one_step_{tag}_{mode}.npz',pred=pred)
            one_step.append(dict(name=name,seed=seed,mode=mode,inference_seconds=one_seconds,**evaluate(y,pred,x,matrix)))
            one_bins.extend(dict(name=name,seed=seed,mode=mode,**b) for b in distributions(y,pred,species))
            # Equal fixed measurement subset and seed for bounded diagnostic cost.
            subset=x[::4];clean=model.predict(subset);rng=np.random.default_rng(100)
            for amplitude in (1e-16,1e-12,1e-8):
                noisy=subset.copy();noisy[:,1:9]=np.maximum(noisy[:,1:9]+rng.normal(size=(len(subset),8))*amplitude,0)
                noisy[:,9]=1-noisy[:,1:9].sum(axis=1)
                changed=model.predict(noisy);delta=changed[:,1:]-clean[:,1:]
                noise.append(dict(name=name,seed=seed,mode=mode,noise_std=amplitude,
                                  prediction_change_rmse=float(np.sqrt(np.nanmean(delta*delta))),
                                  invalid_fraction=float(np.mean(~np.all(np.isfinite(changed),axis=1)))))
            rr=pd.DataFrame(rows)
            result.update(median_ignition_relative_error_refined=float(rr.tau400_relative_error_refined.median()),
                          mean_ignition_relative_error_refined=float(rr.tau400_relative_error_refined.mean()),
                          median_peak_ignition_relative_error_refined=float(rr.tau_dTdt_relative_error_refined.median()))
            summary.append(dict(name=name,seed=seed,mode=mode,seconds=time.perf_counter()-began,**result))
            for filename,values in [('summary.csv',summary),('trajectories.csv',details),('species_profiles.csv',species_rows),
                                    ('one_step.csv',one_step),('concentration_bins.csv',one_bins),('noise.csv',noise),('enthalpy.csv',energy)]:
                pd.DataFrame(values).to_csv(out/filename,index=False)
            print('HELDOUT',name,seed,mode,result,flush=True)
        # Saved model reload must reproduce its validation-independent original prediction.
        fresh=load_candidate(entry)
        np.testing.assert_allclose(fresh.predict(x[:50]),p.predict(x[:50]),rtol=1e-10,atol=1e-12,equal_nan=True)
        reloads.append(dict(name=name,seed=seed,reload_passed=True))
    pd.DataFrame(reloads).to_csv(out/'reload_checks.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(status='completed',wall_seconds=time.perf_counter()-start,dt=dt,
        primary=primary,test_selection=False,teacher_forcing=False,models=len(entries),
        source_sha256=hashlib.sha256(Path('results/verified/dataset.npz').read_bytes()).hexdigest()),indent=2))
    print('HELDOUT COMPLETE',time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
