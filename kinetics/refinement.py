"""Iterative training-only augmentation; selection by validation 0-D ignition.

python -m kinetics.refinement --config configs/refinement.json
No existing dataset, model or report is overwritten. No test-conditioned tuning.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .augmentation import augment
from .data import samples
from .study import Predictor,tf,rollouts
from .physics import PhysicalPredictor
from .posteriori import reference,metrics,ignition
from .metrics import evaluate

SPECS=[dict(name='boxcox_g0.1',kind='boxcox',gamma=.1),
       dict(name='shifted',kind='shifted_boxcox',gamma=.3,epsilon=3.162277660168379e-8)]


def train_updates(spec,seed,updates,x,y,cfg,deadline):
    start=time.perf_counter();p=Predictor(spec,x,y,cfg,seed)
    features=tf.constant(p.train_features);target=tf.constant(p.train_target)
    @tf.function(input_signature=[tf.TensorSpec([None],tf.int32),tf.TensorSpec([],tf.float32)])
    def step(ids,lr):
        p.optimizer.learning_rate.assign(lr)
        with tf.GradientTape() as tape:
            residual=p.model(tf.gather(features,ids),training=True)-tf.gather(target,ids)
            loss=tf.reduce_mean(residual**2)
        grads=tape.gradient(loss,p.model.trainable_variables)
        grads,_=tf.clip_by_global_norm(grads,5.)
        p.optimizer.apply_gradients(zip(grads,p.model.trainable_variables))
        return loss
    rng=np.random.default_rng(seed);history=[]
    for k in range(updates):
        if k%100==0 and time.perf_counter()>deadline:raise TimeoutError('Refinement compute budget exhausted')
        lr=cfg['learning_rate']*(1 if k<updates*.6 else .3 if k<updates*.9 else .1)
        ids=rng.integers(0,len(x),cfg['batch_size'],dtype=np.int32)
        loss=float(step(ids,np.float32(lr)).numpy())
        if (k+1)%500==0:history.append(dict(update=k+1,loss=loss,learning_rate=lr))
    return p,time.perf_counter()-start,history


def posteriori(p,states,pressures,ids,matrix,dt,mode):
    data=dict(states=states,pressures=pressures,split=np.full(len(ids),'val'),element_matrix=matrix)
    model=PhysicalPredictor(p) if mode=='physics' else p
    _,traces=rollouts(model,data,'val',repair=False)
    rows=[];species_rows=[]
    for k,trace in traces:
        m,s=metrics(trace,states[k],dt,matrix,len(states[k])-1)
        htrue=ignition(states[k],dt,states[k,0,0]);hpred=ignition(trace,dt,states[k,0,0])
        truth_ign=np.isfinite(htrue['tau400']);pred_ign=np.isfinite(hpred['tau400'])
        # An incomplete no-ignition prediction is a failure, not a true negative.
        classified=(truth_ign==pred_ign) and (pred_ign or m['complete'])
        capped=abs(hpred['tau400']-htrue['tau400'])/htrue['tau400'] if truth_ign and pred_ign else 2. if truth_ign!=pred_ign else 0.
        m.update(trajectory=int(ids[k]),mode=mode,reference_ignites=truth_ign,prediction_ignites=pred_ign,
                 ignition_classification_correct=bool(classified),capped_ignition_error=min(capped,5.))
        rows.append(m)
        species_rows.extend(dict(trajectory=int(ids[k]),mode=mode,**v) for v in s)
    frame=pd.DataFrame(rows)
    complete=float(frame.complete.mean());physical=float(frame.physically_admissible_complete.mean())
    accuracy=float(frame.ignition_classification_correct.mean())
    paired=frame.loc[frame.reference_ignites&frame.prediction_ignites,'tau400_relative_error']
    # Includes penalties for failure and missed/false ignition; no survivor-only selection.
    temperature=np.where(frame.complete,np.minimum(frame.full_temperature_rmse.fillna(2000)/1000,5),2.)
    score=5*(1-complete)+3*(1-physical)+3*(1-accuracy)+float(frame.capped_ignition_error.mean())+float(np.mean(temperature))
    summary=dict(score=score,completion_fraction=complete,physical_fraction=physical,ignition_classification=accuracy,
                 ignition_pairs=len(paired),median_ignition_relative_error=float(paired.median()) if len(paired) else float('nan'),
                 mean_ignition_relative_error=float(paired.mean()) if len(paired) else float('nan'),
                 temperature_rmse_completed=float(frame.full_temperature_rmse.mean()),species_rmse_completed=float(frame.full_species_rmse.mean()))
    return summary,rows,species_rows,traces


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/refinement.json');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(cfg['output'])
    if out.exists():raise FileExistsError('Use a new refinement output directory')
    out.mkdir();(out/'models').mkdir();(out/'traces').mkdir()
    start=time.perf_counter();deadline=start+cfg['max_seconds']
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    source=np.load(cfg['source']);cached={k:source[k] for k in source.files}
    assert float(source['dt'])==cfg['dt']
    dt=cfg['dt'];matrix=cached['element_matrix']
    # Reference helper selects test; remap only original validation IDs for this call.
    val_source={**cached,'split':np.where(cached['split']=='val','test','other')}
    val_states,val_ids,_=reference(val_source,cfg['horizon'],out/'validation_reference.npz')
    val_pressures=cached['pressures'][val_ids]
    x0,y0,_=samples(source,'train');vx,vy,_=samples(source,'val')
    screening=[];all_rows=[];histories=[]
    def assess(p,name,seed,phase,train_seconds=0):
        for mode in ('raw','physics'):
            result,rows,sp,traces=posteriori(p,val_states,val_pressures,val_ids,matrix,dt,mode)
            common=dict(name=name,seed=seed,phase=phase,mode=mode)
            screening.append(dict(**common,**result,train_seconds=train_seconds))
            all_rows.extend(dict(name=name,seed=seed,phase=phase,**r) for r in rows)
            np.savez_compressed(out/'traces'/f'{phase}_{name}_seed{seed}_{mode}.npz',**{f'trajectory_{int(val_ids[i])}':t for i,t in traces})
            pd.DataFrame(screening).to_csv(out/'validation_summary.csv',index=False)
            pd.DataFrame(all_rows).to_csv(out/'validation_trajectories.csv',index=False)
            print('VALIDATION',common,result,flush=True)
    for spec,legacy in zip(SPECS,('boxcox_g0.1','r2_shifted_boxcox_global_3.16228')):
        p=Predictor.load(Path('results/verified/models')/f'{legacy}_seed11')
        assess(p,'legacy_'+spec['name'],11,'legacy')
    pools={'original':(x0,y0)};augmentation_meta=[]
    extra_x=[];extra_y=[]
    for level,count in zip(cfg['augmentation_levels'],cfg['augmentation_counts']):
        path=out/f'augmentation_{level}.npz'
        augmentation_meta.append(augment(cfg['source'],path,count,710+len(augmentation_meta),level))
        a=np.load(path);extra_x.append(a['x']);extra_y.append(a['y'])
        pools[level]=(np.concatenate([x0,*extra_x]),np.concatenate([y0,*extra_y]))
    candidates=[]
    for pool,(x,y) in pools.items():
        for spec in SPECS:
            name=pool+'_'+spec['name'];p,seconds,h=train_updates(spec,11,cfg['screen_updates'],x,y,cfg,deadline)
            p.save(out/'models'/f'screen_{name}_seed11')
            candidates.append(dict(name=name,pool=pool,spec=spec))
            histories.extend(dict(name=name,seed=11,phase='screen',**v) for v in h)
            assess(p,name,11,'screen',seconds)
    screen=pd.DataFrame(screening)
    eligible=screen[(screen.phase=='screen')&(screen['mode']=='physics')].sort_values('score')
    selected=[]
    for spec in SPECS:
        match=eligible[eligible.name.str.endswith('_'+spec['name'])].iloc[0]
        selected.append(next(c for c in candidates if c['name']==match['name']))
    (out/'selection_frozen.json').write_text(json.dumps(dict(selected=selected,criterion='validation self-fed 0-D score; no test access',seeds=cfg['seeds']),indent=2))
    final_models=[]
    for candidate in selected:
        x,y=pools[candidate['pool']]
        for seed in cfg['seeds']:
            p,seconds,h=train_updates(candidate['spec'],seed,cfg['final_updates'],x,y,cfg,deadline)
            name=candidate['name'];p.save(out/'models'/f'final_{name}_seed{seed}')
            histories.extend(dict(name=name,seed=seed,phase='final',**v) for v in h)
            assess(p,name,seed,'final',seconds)
            final_models.append(dict(name=name,seed=seed,path=str(out/'models'/f'final_{name}_seed{seed}')))
    pd.DataFrame(histories).to_csv(out/'history.csv',index=False)
    (out/'final_models.json').write_text(json.dumps(final_models,indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(status='validation_complete',wall_seconds=time.perf_counter()-start,
        augmentation=augmentation_meta,dt=dt,validation_ids=val_ids.tolist(),test_evaluated=False,
        source_sha256=hashlib.sha256(Path(cfg['source']).read_bytes()).hexdigest()),indent=2))
    print('VALIDATION COMPLETE',time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
