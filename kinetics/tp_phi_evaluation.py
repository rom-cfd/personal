"""Sequential ANN / T/P/phi / mixed teams with explicit, fixed IC phi."""
import gc
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .ann_deeponet_study import (check_deadline, physical_diagnostics, verified_evaluation,
                                write_json, one_step_metrics)
from .posteriori import metrics as trajectory_metrics
from .data import SPECIES, samples
from .study import Predictor, tf
from .mixed_ensemble import MixedMean, require_shared_preprocessing
from .deeponet_long_evaluation import TimedMean
from .p_only_training import sha256
from .tp_phi_deeponet import PhiDeepONet


class PhiMean(TimedMean):
    def __init__(self,members,families,mode='repaired'):
        super().__init__(members,families,mode)
        phi_members=[p for p in members if isinstance(p,PhiDeepONet)]
        if not phi_members:
            raise ValueError('At least one explicit-phi member required')
        self.phi_norm=phi_members[0].phi_norm
        for p in phi_members[1:]:
            np.testing.assert_array_equal(p.phi_norm.mean,self.phi_norm.mean)
            np.testing.assert_array_equal(p.phi_norm.scale,self.phi_norm.scale)
        @tf.function(input_signature=[tf.TensorSpec([None,12],tf.float32)])
        def forward(features):
            return tf.stack([p.model(features if isinstance(p,PhiDeepONet) else features[:,:11],training=False)
                             for p in self.members])
        self.forward=forward

    def proposals(self,x):
        PhiDeepONet.validate_inputs(x)
        started=time.perf_counter()
        common=Predictor.features(self.base,x)
        features=np.c_[common,self.phi_norm.forward(x[:,10:11])].astype('float32')
        self.timing['features']+=time.perf_counter()-started
        started=time.perf_counter()
        delta=self.forward(features).numpy().astype('float64')
        self.timing['neural']+=time.perf_counter()-started
        started=time.perf_counter()
        base=self.base
        z=base.norm.inverse(base.norm.forward(base.encode_state(x[:,:10]))[None,...]+delta)
        ys=z[...,1:]
        if self.mode=='repaired':ys=np.clip(ys,0,base.transform.forward(np.ones((1,9))))
        result=np.concatenate([z[...,:1]*1000,base.transform.inverse(ys)],axis=-1)
        self.timing['decode']+=time.perf_counter()-started
        return result


def rollout(predictor, initial_states, pressures, steps, dt, ids=None, deadline=float('inf'), *, phis):
    """Self-fed states only: future reference states are not accepted as arguments."""
    state = np.asarray(initial_states).copy()
    ids = np.arange(len(state)) if ids is None else np.asarray(ids)
    histories = [[s.copy()] for s in state]
    active = np.ones(len(state), bool)
    events = []
    for step in range(1, steps + 1):
        if time.perf_counter() >= deadline:
            raise TimeoutError('Rollout budget exhausted')
        live = np.flatnonzero(active)
        if not len(live):
            break
        x = np.c_[state[live], phis[live], pressures[live]]
        pred = predictor.predict(x)
        for j, k in enumerate(live):
            out = pred[j]
            accepted = bool(np.all(np.isfinite(out)) and 200 <= out[0] <= 6000
                            and np.all(out[1:] >= 0) and out[1:].sum() > 0)
            diag = predictor.last_diagnostics
            events.append(dict(trajectory=int(ids[k]), step=step, time=step * dt,
                               valid_count=int(diag['valid_count'][j]),
                               valid_ann=int(diag['valid_ann'][j]),
                               valid_deeponet=int(diag['valid_deeponet'][j]), accepted=accepted))
            if accepted:
                state[k] = out
                histories[k].append(out.copy())
            else:
                active[k] = False
    return [np.asarray(h) for h in histories], events


def one_steps(predictor, states, pressures, phis, matrix, chunk_size, deadline):
    x = np.concatenate([states[:, :-1], np.broadcast_to(pressures[:, None, None],
                       (*states[:, :-1].shape[:2], 1))], axis=-1).reshape(-1, 11)
    x = np.c_[x[:, :10], np.repeat(phis, states.shape[1]-1), x[:, -1]]
    target = states[:, 1:].reshape(-1, 10)
    predicted = np.full_like(target, np.nan)
    valid_ann = valid_deeponet = total_candidates = 0
    for start in range(0, len(x), chunk_size):
        check_deadline(deadline)
        end = start + chunk_size
        predicted[start:end] = predictor.predict(x[start:end])
        valid_ann += int(predictor.last_diagnostics['valid_ann'].sum())
        valid_deeponet += int(predictor.last_diagnostics['valid_deeponet'].sum())
        total_candidates += len(x[start:end]) * len(predictor.members)
    result = one_step_metrics(target, predicted, x, matrix)
    result.update(pair_count=len(x), valid_ann=valid_ann, valid_deeponet=valid_deeponet,
                  invalid_candidate_fraction=1 - (valid_ann + valid_deeponet) / total_candidates)
    finite = np.isfinite(predicted).all(axis=1)
    species = []
    for k, name in enumerate(SPECIES):
        error = predicted[finite, k + 1] - target[finite, k + 1]
        species.append(dict(species=name, finite_rows=int(finite.sum()), total_rows=len(x),
                            rmse=float(np.sqrt(np.mean(error ** 2))) if len(error) else np.nan))
    return result, species



def evaluate_arm(root,name,members,families,mode,split,data,base_cfg,fingerprint,deadline):
    parent=root/'evaluations'/f'{name}_{mode}_{split}'
    complete=sorted(parent.glob('*/complete.json'))
    if complete:
        return verified_evaluation(complete[-1].parent)
    ids=np.flatnonzero(data['split']==split)
    states,pressures=data['states'][ids],data['pressures'][ids]
    fine_info=fingerprint['reference_ignition'][split]
    if sha256(fine_info['path'])!=fine_info['sha256']:
        raise ValueError('Ignition reference changed')
    fine=pd.read_csv(fine_info['path']).set_index('trajectory')
    if set(fine.index)!=set(ids) or not np.allclose(fine.reference_dt,base_cfg['dt']/10):
        raise ValueError('Ignition reference ICs/resolution differ')
    check_deadline(deadline)
    started=time.perf_counter()
    p=PhiMean(members,families,mode)
    phis=data['initials'][ids,2]
    warm=time.perf_counter();p.predict(np.c_[states[:,0],phis,pressures])
    warmup_seconds=time.perf_counter()-warm;p.reset_timing()
    before=time.perf_counter()
    traces,events=rollout(p,states[:,0],pressures,states.shape[1]-1,base_cfg['dt'],ids,deadline,phis=phis)
    rollout_seconds=time.perf_counter()-before
    rollout_timing=dict(p.timing)
    before=time.perf_counter()
    rows,species_rows,physics=[],[],[]
    for i,truth,trace,pressure in zip(ids,states,traces,pressures):
        check_deadline(deadline)
        for horizon in (.002,base_cfg['horizon']):
            row,species=trajectory_metrics(trace,truth,base_cfg['dt'],data['element_matrix'],round(horizon/base_cfg['dt']))
            tau=float(fine.loc[i,'tau400']);tau=tau if tau<=horizon else np.nan
            pred_tau=row['predicted_tau400']
            row.update(trajectory=int(i),horizon=horizon,reference_tau400_refined=tau,
                tau400_relative_error_refined=abs(pred_tau-tau)/tau if np.isfinite(tau) and np.isfinite(pred_tau) else np.nan)
            row['reference_ignites']=bool(np.isfinite(row['reference_tau400']))
            row['prediction_ignites']=bool(np.isfinite(pred_tau))
            row['unresolved_before_ignition']=bool(not row['complete'] and not row['prediction_ignites'])
            row['ignition_classification_correct']=bool(row['reference_ignites']==row['prediction_ignites']
                and (row['prediction_ignites'] or row['complete']))
            rows.append(row)
            species_rows.extend(dict(trajectory=int(i),horizon=horizon,**{**r,'species':SPECIES[r['species']]}) for r in species)
        physics.append(dict(trajectory=int(i),**physical_diagnostics(trace,truth,pressure,
                           data['equilibrium'][i],p.physics,deadline)))
    diagnostics_seconds=time.perf_counter()-before
    p.reset_timing();before=time.perf_counter()
    one,one_species=one_steps(p,states,pressures,phis,data['element_matrix'],base_cfg['one_step_chunk_size'],deadline)
    one_seconds=time.perf_counter()-before
    destination=parent/f'attempt_{time.time_ns()}';destination.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(destination/'traces.npz',**{f'trajectory_{int(i)}':t for i,t in zip(ids,traces)})
    for filename,values in [('events',events),('trajectories',rows),('species_profiles',species_rows),
                            ('physics',physics),('one_step_species',one_species)]:
        pd.DataFrame(values).to_csv(destination/f'{filename}.csv',index=False)
    write_json(destination/'one_step.json',one)
    write_json(destination/'timing.json',dict(warmup_seconds=warmup_seconds,rollout_seconds=rollout_seconds,
        diagnostics_seconds=diagnostics_seconds,one_step_seconds=one_seconds,
        rollout_components=rollout_timing,one_step_components=dict(p.timing),total_arm_seconds=time.perf_counter()-started))
    info=dict(method=name,mode=mode,split=split,member_count=len(members),output_sha256={
        path.name:sha256(path) for path in destination.iterdir() if path.is_file()})
    write_json(destination/'complete.json',info)
    del p;gc.collect()
    return info



def endpoint_models(records):
    models=[]
    for record in records:
        cp=Path(record['checkpoint'])
        marker=json.loads((cp/'complete.json').read_text())
        for name,h in marker['output_sha256'].items():
            if Path(name).name!=name or sha256(cp/name)!=h:raise ValueError('Endpoint checkpoint changed')
        if sha256(record['path']+'.keras')!=record['sha256']:raise ValueError('Model changed')
        models.append(PhiDeepONet.load(record['path']))
    return models


def residual_losses(root,groups,pairs,data,deadline):
    destination=root/'loss_diagnostics'/f'attempt_{time.time_ns()}'
    destination.mkdir(parents=True,exist_ok=False)
    rows=[];started=time.perf_counter()
    for family,models in groups.items():
        base=models[0]
        width=12 if isinstance(base,PhiDeepONet) else 11
        @tf.function(input_signature=[tf.TensorSpec([None,width],tf.float32)])
        def forward(features):return tf.stack([p.model(features,training=False) for p in models])
        for split in ('train','val'):
            if split=='train':x,y=pairs['x'],pairs['y']
            else:
                x,y,ids=samples(data,'val')
                x=np.c_[x[:,:10],np.repeat(data['initials'][ids,2],data['states'].shape[1]-1),x[:,-1]]
            target=base.norm.forward(base.encode_state(y))-base.norm.forward(base.encode_state(x[:,:10]))
            features=base.features(x).astype('float32')
            squared=np.zeros(len(models))
            for start in range(0,len(x),512):
                check_deadline(deadline)
                output=forward(features[start:start+512]).numpy().astype('float64')
                squared+=np.sum((output-target[None,start:start+512])**2,axis=(1,2))
            for p,value in zip(models,squared/target.size):
                rows.append(dict(family=family,seed=p.seed_label,split=split,pairs=len(x),
                    transformed_residual_MSE=float(value)))
    pd.DataFrame(rows).to_csv(destination/'residual_losses.csv',index=False)
    write_json(destination/'timing.json',dict(seconds=time.perf_counter()-started))
    write_json(destination/'complete.json',dict(output_sha256={p.name:sha256(p) for p in destination.iterdir()}))


def evaluate(root,cfg,snapshot,pairs,data,deadline):
    from .tp_phi_study import event
    # All 10k and long endpoints are frozen before evaluation; test never selects them.
    evaluation_deadline=deadline-420
    started=time.perf_counter()
    ann=[Predictor.load(r['path']) for r in snapshot['fingerprint']['ann_members']]
    baseline=endpoint_models(json.loads((root/'matched_members.json').read_text()))
    long=endpoint_models(json.loads((root/'final_members.json').read_text()))
    require_shared_preprocessing(ann+baseline+long)
    for group in (ann,baseline,long):
        for p,seed in zip(group,snapshot['base_config']['seeds']):p.seed_label=seed
    write_json(root/'model_loading.json',dict(shared_30_model_loading_seconds=time.perf_counter()-started))
    residual_losses(root,dict(ann=ann,phi_10k=baseline,phi_long=long),pairs,data,evaluation_deadline)
    definitions=[('phi_10k10',baseline,['deeponet']*10),
        ('ann_phi_10k20',ann+baseline,['ann']*10+['deeponet']*10),
        ('phi_long10',long,['deeponet']*10),
        ('ann_phi_long20',ann+long,['ann']*10+['deeponet']*10)]
    # ANN has no phi input: reuse the unmodified, measured 11-input evaluator.
    from .deeponet_long_evaluation import evaluate_arm as ann_arm
    for split in cfg['evaluation_splits']:
        for mode in cfg['modes']:
            ann_arm(root,'ann10',ann,['ann']*10,mode,split,data,
                    snapshot['base_config'],snapshot['fingerprint'],evaluation_deadline)
            event(root,phase='evaluation_finished',method='ann10',mode=mode,split=split)
        for name,models,families in definitions:
            for mode in cfg['modes']:
                evaluate_arm(root,name,models,families,mode,split,data,
                             snapshot['base_config'],snapshot['fingerprint'],evaluation_deadline)
                event(root,phase='evaluation_finished',method=name,mode=mode,split=split)
