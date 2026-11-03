"""Frozen soft models, unchanged ensemble/repair rules and before/after traces."""
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from .ann_deeponet_study import check_deadline,verified_evaluation
from .data import samples
from .deeponet_long_evaluation import evaluate_arm as ordinary_arm,TimedMean
from .tp_phi_evaluation import evaluate_arm as phi_arm
from .mixed_ensemble import rollout,require_shared_preprocessing
from .p_only_training import sha256,write_json
from .soft_constraint_training import LOADERS,FAMILIES,SoftObjective
from .soft_constraint_study import event,inputs_for
from .study import tf

TEAMS=[('ann10','ann'),('p_only10','p_only'),('ann_p_only20','p_only'),
       ('tp10','tp'),('ann_tp20','tp'),('tp_phi10','tp_phi'),('ann_tp_phi20','tp_phi')]
BASELINE_KEYS={'ann10':'ann10','p_only10':'p_short','ann_p_only20':'ann_p_short',
    'tp10':'tp_short','ann_tp20':'ann_tp_short','tp_phi10':'phi_short','ann_tp_phi20':'ann_phi_short'}


def load_groups(root,snapshot):
    records=json.loads((root/'final_members.json').read_text())
    if len(records)!=40 or any(r['updates']!=10000 for r in records):raise ValueError('Need40 frozen10k members')
    groups={}
    for family in FAMILIES:
        by_seed={r['seed']:r for r in records if r['family']==family};members=[]
        for seed in snapshot['base_config']['seeds']:
            r=by_seed[seed];cp=Path(r['checkpoint']);marker=json.loads((cp/'complete.json').read_text())
            for name,h in marker['output_sha256'].items():
                if sha256(cp/name)!=h:raise ValueError('Soft model checkpoint changed')
            p=LOADERS[family].load(r['path']);p.seed_label=seed;members.append(p)
        groups[family]=members
    require_shared_preprocessing([p for group in groups.values() for p in group])
    return groups


def load_baseline(snapshot):
    groups={}
    for family,records in snapshot['baseline_members'].items():
        groups[family]=[LOADERS[family].load(r['path']) for r in records]
        for p,r in zip(groups[family],records):p.seed_label=r['seed']
    require_shared_preprocessing([p for group in groups.values() for p in group])
    return groups


def member_diagnostics(root,cfg,snapshot,groups,baseline,data,pairs,deadline):
    destination=root/'member_diagnostics'
    if (destination/'complete.json').exists():verified_evaluation(destination);return
    destination.mkdir(parents=True,exist_ok=False);rows=[]
    matrix=np.array(snapshot['element_matrix'])
    for stage,collection in [('baseline',baseline),('soft_mu10',groups)]:
        for family,models in collection.items():
            objectives=[SoftObjective(p,matrix,cfg['mu']) for p in models]
            width=12 if family=='tp_phi' else 11
            @tf.function(input_signature=[tf.TensorSpec([None,width],tf.float32)])
            def forward(features):return tf.stack([p.model(features,training=False) for p in models])
            for split in ('train','val'):
                if split=='train':x,y=inputs_for(family,pairs,data),pairs['y']
                else:
                    x,y,ids=samples(data,'val')
                    if family=='tp_phi':x=np.c_[x[:,:10],np.repeat(data['initials'][ids,2],data['states'].shape[1]-1),x[:,-1]]
                p=models[0];a=p.norm.forward(p.encode_state(x[:,:10]));b=p.norm.forward(p.encode_state(y))
                features=p.features(x).astype('float32');target=(b-a).astype('float32')
                totals=np.zeros((len(models),3));invalid=np.zeros(len(models),int)
                for start in range(0,len(x),512):
                    check_deadline(deadline);end=min(start+512,len(x));outputs=forward(features[start:end])
                    for k,(delta,objective) in enumerate(zip(outputs,objectives)):
                        _,values=objective.components(delta,tf.constant(target[start:end]),
                            tf.constant(a[start:end],tf.float64),tf.constant(y[start:end,1:],tf.float64))
                        totals[k]+=values.numpy()*(end-start)
                        z=p.norm.inverse(a[start:end]+delta.numpy().astype('float64'))[:,1:]
                        invalid[k]+=int(np.any(z<0,axis=1).sum())
                for p,(total,mse,element),bad in zip(models,totals/len(x),invalid):
                    rows.append(dict(stage=stage,family=family,seed=p.seed_label,split=split,pairs=len(x),
                        objective_at_mu10=float(total),data_MSE=float(mse),physical_element_L1=float(element),
                        invalid_inverse_row_fraction=float(bad/len(x))))
            event(root,phase='member_diagnostics_finished',stage=stage,family=family)
    pd.DataFrame(rows).to_csv(destination/'loss_components.csv',index=False)
    write_json(destination/'complete.json',dict(output_sha256={p.name:sha256(p) for p in destination.iterdir()}))


def baseline_trace_map(root,cfg,snapshot,data,baseline,deadline):
    grid=Path(cfg['baseline_grid']);run=json.loads((grid/'run.json').read_text())
    result={}
    for team,_ in TEAMS:
        key=BASELINE_KEYS[team]
        if key in run['cached_traces']:path=Path(run['cached_traces'][key])
        else:path=grid/'generated_rollouts'/key/'traces.npz'
        if key in run['cached_traces']:expected=run['source_sha256'][str(path)]
        else:expected=json.loads((path.parent/'complete.json').read_text())['output_sha256']['traces.npz']
        if sha256(path)!=expected:raise ValueError('Repaired baseline trace changed')
        result[f'{team}_repaired']=dict(path=str(path),sha256=sha256(path))
    # Reuse the already frozen raw ANN/phi results. Generate only missing P/T/P
    # baseline raw test rollouts; no model is retrained for a control.
    cached={'ann10':('results/tp_phi_study/run_001','ann10'),
        'tp_phi10':('results/tp_phi_study/run_001','phi_10k10'),
        'ann_tp_phi20':('results/tp_phi_study/run_001','ann_phi_10k20')}
    ids=np.flatnonzero(data['split']=='test');states=data['states'][ids];pressures=data['pressures'][ids]
    for team,family in TEAMS:
        if team in cached:
            base,method=cached[team];marker=sorted(Path(base,'evaluations',f'{method}_raw_test').glob('*/complete.json'))[-1]
            info=json.loads(marker.read_text());path=marker.parent/'traces.npz'
            if sha256(path)!=info['output_sha256']['traces.npz']:raise ValueError('Raw baseline changed')
        else:
            destination=root/'baseline_raw'/team;marker=destination/'complete.json'
            if marker.exists():verified_evaluation(destination)
            else:
                models=baseline[family]
                labels=['deeponet']*10
                if team.startswith('ann_'):models=baseline['ann']+models;labels=['ann']*10+labels
                predictor=TimedMean(models,labels,'raw');before=time.perf_counter()
                predictor.predict(np.c_[states[:,0],pressures]);warmup=time.perf_counter()-before
                predictor.reset_timing();before=time.perf_counter()
                traces,events=rollout(predictor,states[:,0],pressures,5000,2e-6,ids,deadline)
                elapsed=time.perf_counter()-before
                destination.mkdir(parents=True,exist_ok=False)
                np.savez_compressed(destination/'traces.npz',**{f'trajectory_{int(i)}':t for i,t in zip(ids,traces)})
                pd.DataFrame(events).to_csv(destination/'events.csv',index=False)
                write_json(destination/'timing.json',dict(rollout_seconds=elapsed,warmup_seconds=warmup,
                    rollout_components=predictor.timing,completed=sum(len(t)==5001 for t in traces)))
                write_json(marker,dict(output_sha256={p.name:sha256(p) for p in destination.iterdir()}))
                event(root,phase='baseline_raw_finished',method=team,seconds=elapsed)
                del predictor;gc.collect()
            path=destination/'traces.npz'
        result[f'{team}_raw']=dict(path=str(path),sha256=sha256(path))
    write_json(root/'baseline_traces.json',result)


def evaluate(root,cfg,snapshot,data,pairs,deadline):
    started=time.perf_counter();groups=load_groups(root,snapshot);baseline=load_baseline(snapshot)
    write_json(root/'model_loading.json',dict(seconds=time.perf_counter()-started,
        soft_members=40,baseline_members=40))
    member_diagnostics(root,cfg,snapshot,groups,baseline,data,pairs,deadline)
    baseline_trace_map(root,cfg,snapshot,data,baseline,deadline)
    for team,family in TEAMS:
        models=groups[family];labels=['ann']*10 if family=='ann' else ['deeponet']*10
        if team.startswith('ann_'):models=groups['ann']+models;labels=['ann']*10+labels
        runner=phi_arm if family=='tp_phi' else ordinary_arm
        for split in cfg['evaluation_splits']:
            for mode in cfg['modes']:
                runner(root,team,models,labels,mode,split,data,snapshot['base_config'],snapshot['fingerprint'],deadline)
                event(root,phase='evaluation_finished',method=team,mode=mode,split=split)
