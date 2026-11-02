"""Five independent same-state mean teams; sequential dense and self-fed evaluation."""
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from .ann_deeponet_study import (check_deadline,physical_diagnostics,one_steps,
                                verified_evaluation,write_json)
from .posteriori import metrics as trajectory_metrics
from .data import SPECIES,samples
from .study import Predictor,tf
from .deeponet import ThermoDeepONet
from .p_only_deeponet import PressureDeepONet
from .mixed_ensemble import MixedMean,rollout,require_shared_preprocessing
from .p_only_training import sha256


class TimedMean(MixedMean):
    def reset_timing(self):
        super().reset_timing()
        self.timing.update(features=0.,neural=0.,decode=0.)

    def proposals(self,x):
        started=time.perf_counter()
        features=self.base.features(x).astype('float32')
        self.timing['features']+=time.perf_counter()-started
        started=time.perf_counter()
        delta=self.forward(features).numpy().astype('float64')
        self.timing['neural']+=time.perf_counter()-started
        started=time.perf_counter()
        base=self.base
        z=base.norm.inverse(base.norm.forward(base.encode_state(x[:,:10]))[None,...]+delta)
        ys=z[...,1:]
        if self.mode=='repaired':
            ys=np.clip(ys,0,base.transform.forward(np.ones((1,9))))
        result=np.concatenate([z[...,:1]*1000,base.transform.inverse(ys)],axis=-1)
        self.timing['decode']+=time.perf_counter()-started
        return result


def load_members(root,snapshot):
    records=json.loads((root/'final_members.json').read_text())
    if len(records)!=20 or any(r is None or r['updates']<10000 for r in records):
        raise ValueError('All twenty models need saved trained endpoints before comparison')
    members={'ann':[Predictor.load(r['path']) for r in snapshot['fingerprint']['ann_members']]}
    for family,loader in [('p_only',PressureDeepONet),('tp',ThermoDeepONet)]:
        group={r['seed']:r for r in records if r['family']==family}
        if set(group)!=set(snapshot['base_config']['seeds']):
            raise ValueError('Incomplete family endpoint set')
        models=[]
        for seed in snapshot['base_config']['seeds']:
            record=group[seed]
            marker=json.loads(Path(record['checkpoint'],'complete.json').read_text())
            for name,expected in marker['output_sha256'].items():
                if Path(name).name!=name or sha256(Path(record['checkpoint'],name))!=expected:
                    raise ValueError('Endpoint checkpoint hash mismatch')
            if sha256(record['path']+'.keras')!=record['sha256']:
                raise ValueError('Endpoint model hash mismatch')
            models.append(loader.load(record['path']))
        members[family]=models
    require_shared_preprocessing(members['ann']+members['tp']+members['p_only'])
    return members


def loss_diagnostics(root,members,pairs,data,seeds,deadline):
    parent=root/'loss_diagnostics'
    complete=sorted(parent.glob('*/complete.json'))
    if complete:
        verified_evaluation(complete[-1].parent);return
    destination=parent/f'attempt_{time.time_ns()}'
    destination.mkdir(parents=True,exist_ok=False)
    ordered=[(family,seed,p) for family in ('ann','tp','p_only')
             for seed,p in zip(seeds,members[family])]
    base=ordered[0][2]
    @tf.function(input_signature=[tf.TensorSpec([None,11],tf.float32)])
    def forward(features):
        return tf.stack([p.model(features,training=False) for _,_,p in ordered])
    rows=[];started=time.perf_counter()
    for split in ('train','val'):
        if split=='train':
            x,y=pairs['x'],pairs['y']
        else:
            x,y,_=samples(data,'val')
        target=base.norm.forward(base.encode_state(y))-base.norm.forward(base.encode_state(x[:,:10]))
        features=base.features(x).astype('float32')
        squared=np.zeros(len(ordered))
        for start in range(0,len(x),512):
            check_deadline(deadline)
            output=forward(features[start:start+512]).numpy().astype('float64')
            squared+=np.sum((output-target[None,start:start+512])**2,axis=(1,2))
        for (family,seed,_),value in zip(ordered,squared/target.size):
            rows.append(dict(family=family,seed=seed,split=split,pairs=len(x),transformed_residual_MSE=float(value)))
    pd.DataFrame(rows).to_csv(destination/'residual_losses.csv',index=False)
    write_json(destination/'timing.json',dict(seconds=time.perf_counter()-started,
        note='Train/validation diagnostics after frozen endpoints; no test-based selection'))
    write_json(destination/'complete.json',dict(output_sha256={
        p.name:sha256(p) for p in destination.iterdir() if p.is_file()}))


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
    p=TimedMean(members,families,mode)
    warm=time.perf_counter();p.predict(np.c_[states[:,0],pressures])
    warmup_seconds=time.perf_counter()-warm;p.reset_timing()
    before=time.perf_counter()
    traces,events=rollout(p,states[:,0],pressures,states.shape[1]-1,base_cfg['dt'],ids,deadline)
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
    one,one_species=one_steps(p,states,pressures,data['element_matrix'],base_cfg['one_step_chunk_size'],deadline)
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


def evaluate(root,cfg,snapshot,data,deadline):
    from .deeponet_long_study import event
    # Reporting is sequential too; reserve five minutes for immutable figures.
    evaluation_deadline=deadline-300
    check_deadline(evaluation_deadline)
    started=time.perf_counter();members=load_members(root,snapshot)
    if not (root/'model_loading.json').exists():
        write_json(root/'model_loading.json',dict(shared_30_model_loading_seconds=time.perf_counter()-started))
    with np.load(snapshot['base_config']['training_pairs']) as f:
        pairs={key:f[key] for key in f.files}
    loss_diagnostics(root,members,pairs,data,snapshot['base_config']['seeds'],evaluation_deadline)
    definitions=[('ann10',members['ann'],['ann']*10),('tp10',members['tp'],['deeponet']*10),
        ('p_only10',members['p_only'],['deeponet']*10),
        ('ann_tp20',members['ann']+members['tp'],['ann']*10+['deeponet']*10),
        ('ann_p_only20',members['ann']+members['p_only'],['ann']*10+['deeponet']*10)]
    for split in cfg['evaluation_splits']:
        for name,models,families in definitions:
            for mode in cfg['modes']:
                try:
                    info=evaluate_arm(root,name,models,families,mode,split,data,
                        snapshot['base_config'],snapshot['fingerprint'],evaluation_deadline)
                    event(root,phase='evaluation_finished',method=name,mode=mode,split=split)
                except TimeoutError:
                    event(root,phase='evaluation_time_limit',method=name,mode=mode,split=split)
                    return
