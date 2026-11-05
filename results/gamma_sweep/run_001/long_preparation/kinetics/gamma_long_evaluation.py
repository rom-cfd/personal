"""Paired rebuilt short origins and frozen long endpoint evaluation."""
import gc
import json
from pathlib import Path

import numpy as np
from .gamma_sweep_training import gamma_tag
from .gamma_sweep_evaluation import trace_sources as original_sources,validation_choices
from .soft_constraint_training import FAMILIES,LOADERS
from .soft_constraint_evaluation import TEAMS
from .soft_constraint_study import event
from .p_only_training import sha256,write_json
from .mixed_ensemble import require_shared_preprocessing
from .deeponet_long_evaluation import evaluate_arm as ordinary_arm
from .tp_phi_evaluation import evaluate_arm as phi_arm
from .study import tf

REBUILT_TEAMS={'ann10','tp10','ann_p_only20','ann_tp20','ann_tp_phi20'}


def groups_for(records,gamma,seeds):
    tf.keras.backend.clear_session();groups={}
    for family in FAMILIES:
        lookup={r['seed']:r for r in records if r['gamma']==gamma and r['family']==family}
        if set(lookup)!=set(seeds):raise ValueError('Incomplete family endpoint set')
        group=[]
        for seed in seeds:
            r=lookup[seed]
            if sha256(r['path']+'.keras')!=r['sha256']:raise ValueError('Model changed')
            p=LOADERS[family].load(r['path']);p.seed_label=seed
            if p.transform.gamma!=gamma:raise ValueError('Wrong preprocessing exponent')
            group.append(p)
        groups[family]=group
    require_shared_preprocessing([p for group in groups.values() for p in group])
    return groups


def source_for_arm(parent,team,mode,split):
    markers=list((parent/'evaluations'/f'{team}_{mode}_{split}').glob('*/complete.json'))
    if len(markers)!=1:raise ValueError('Require one complete frozen evaluation per arm')
    path=markers[0].parent/'traces.npz';expected=json.loads(markers[0].read_text())['output_sha256']['traces.npz']
    if sha256(path)!=expected:raise ValueError('Trace changed')
    return dict(path=str(path),sha256=expected)


def profiles_from_sources(sources,data,split):
    ids=np.flatnonzero(data['split']==split).astype(int).tolist();profiles={}
    for method,info in sources.items():
        if sha256(info['path'])!=info['sha256']:raise ValueError('Frozen source trace changed')
        with np.load(info['path']) as f:profiles[method]={i:f[f'trajectory_{i}'] for i in ids}
        for i,trace in profiles[method].items():
            if trace.ndim!=2 or trace.shape[1]!=10 or not 1<=len(trace)<=len(data['states'][i]) or not np.isfinite(trace).all():raise ValueError('Invalid trace')
            np.testing.assert_array_equal(trace[0],data['states'][i,0])
    return profiles


def paired_short_sources(root,cfg,snapshot,split,mode):
    old=snapshot['short_snapshot']
    sources=original_sources(Path(cfg['origin']),old['config'],old,split,mode)
    for team in REBUILT_TEAMS:
        sources[f'gamma_0.1_{team}']=source_for_arm(root/'bootstrap_evaluations',team,mode,split)
    return sources


def bootstrap_evaluations(root,cfg,snapshot,short_records,data,deadline):
    if (root/'paired_short_traces.json').exists():return
    groups=groups_for(short_records,.1,snapshot['base_config']['seeds']);all_sources={}
    for split in cfg['evaluation_splits']:
        if split=='test' and not (root/'short_validation_selection.json').exists():raise ValueError('Freeze paired-short validation selection first')
        for team,family in TEAMS:
            if team not in REBUILT_TEAMS:continue
            models=groups[family];labels=['ann']*10 if family=='ann' else ['deeponet']*10
            if team.startswith('ann_'):models=groups['ann']+models;labels=['ann']*10+labels
            runner=phi_arm if family=='tp_phi' else ordinary_arm
            for mode in cfg['modes']:
                runner(root/'bootstrap_evaluations',team,models,labels,mode,split,data,snapshot['base_config'],snapshot['fingerprint'],deadline)
                event(root,phase='bootstrap_evaluation_finished',method=team,mode=mode,split=split)
        for mode in cfg['modes']:all_sources[f'{split}_{mode}']=paired_short_sources(root,cfg,snapshot,split,mode)
        if split=='val':
            profiles=profiles_from_sources(all_sources['val_repaired'],data,'val')
            selected=validation_choices(data,profiles,cfg['gammas']);path=root/'short_validation_selection.json'
            if path.exists() and json.loads(path.read_text())!=selected:raise ValueError('Paired short selection changed')
            if not path.exists():write_json(path,selected)
            del profiles
    write_json(root/'paired_short_traces.json',all_sources)
    del groups,models;gc.collect();event(root,phase='paired_short_evaluations_frozen')


def long_sources(root,cfg,split,mode):
    return {f'{gamma_tag(g)}_{team}':source_for_arm(root/gamma_tag(g),team,mode,split) for g in cfg['gammas'] for team,_ in TEAMS}


def stage_profiles(root,cfg,snapshot,data,stage,split,mode):
    sources=json.loads((root/'paired_short_traces.json').read_text())[f'{split}_{mode}'] if stage=='short' else long_sources(root,cfg,split,mode)
    return profiles_from_sources(sources,data,split),sources


def evaluate(root,cfg,snapshot,data,deadline):
    records=json.loads((root/'final_members.json').read_text())
    if len(records)!=240:raise ValueError('Freeze all240 long endpoints first')
    for split in cfg['evaluation_splits']:
        if split=='test' and not (root/'long_validation_selection.json').exists():raise ValueError('Freeze long validation choices before test')
        for gamma in cfg['gammas']:
            groups=groups_for(records,gamma,snapshot['base_config']['seeds'])
            for team,family in TEAMS:
                models=groups[family];labels=['ann']*10 if family=='ann' else ['deeponet']*10
                if team.startswith('ann_'):models=groups['ann']+models;labels=['ann']*10+labels
                runner=phi_arm if family=='tp_phi' else ordinary_arm
                for mode in cfg['modes']:
                    runner(root/gamma_tag(gamma),team,models,labels,mode,split,data,snapshot['base_config'],snapshot['fingerprint'],deadline)
                    event(root,phase='evaluation_finished',gamma=gamma,method=team,mode=mode,split=split)
            del groups,models;gc.collect()
        if split=='val':
            profiles=profiles_from_sources(long_sources(root,cfg,'val','repaired'),data,'val')
            selected=validation_choices(data,profiles,cfg['gammas']);path=root/'long_validation_selection.json'
            if path.exists() and json.loads(path.read_text())!=selected:raise ValueError('Frozen long choice changed')
            if not path.exists():write_json(path,selected)
            event(root,phase='long_validation_selection_frozen',choices=selected['choices']);del profiles
