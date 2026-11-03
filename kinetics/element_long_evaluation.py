"""Frozen long soft-mu10 members; original inference and test arms unchanged."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .study import tf
from .ann_deeponet_study import check_deadline,verified_evaluation
from .p_only_training import sha256,write_json
from .data import samples
from .soft_constraint_training import LOADERS,FAMILIES,SoftObjective
from .soft_constraint_study import inputs_for
from .soft_constraint_evaluation import TEAMS
from .deeponet_long_evaluation import evaluate_arm as ordinary_arm
from .tp_phi_evaluation import evaluate_arm as phi_arm
from .mixed_ensemble import require_shared_preprocessing
from .element_long_study import event


def load_groups(root,snapshot):
    records=json.loads((root/'final_members.json').read_text())
    if len(records)!=40 or any(r['updates']<10000 for r in records):raise ValueError('Need40 frozen continued members')
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


def member_diagnostics(root,cfg,snapshot,groups,baseline,data,pairs,deadline):
    destination=root/'member_diagnostics'
    if (destination/'complete.json').exists():verified_evaluation(destination);return
    destination.mkdir(parents=True,exist_ok=False);rows=[]
    matrix=np.array(snapshot['element_matrix'])
    for stage,collection in [('soft_long',groups)]:
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



def evaluate(root,cfg,snapshot,data,pairs,deadline):
    started=time.perf_counter();groups=load_groups(root,snapshot)
    write_json(root/'model_loading.json',dict(seconds=time.perf_counter()-started,members=40))
    member_diagnostics(root,cfg,snapshot,groups,None,data,pairs,deadline)
    for team,family in TEAMS:
        models=groups[family];labels=['ann']*10 if family=='ann' else ['deeponet']*10
        if team.startswith('ann_'):models=groups['ann']+models;labels=['ann']*10+labels
        runner=phi_arm if family=='tp_phi' else ordinary_arm
        for split in cfg['evaluation_splits']:
            for mode in cfg['modes']:
                runner(root,team,models,labels,mode,split,data,snapshot['base_config'],snapshot['fingerprint'],deadline)
                event(root,phase='evaluation_finished',method=team,mode=mode,split=split)
