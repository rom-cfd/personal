"""Prepare a traceable research narrative from immutable saved evidence."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT=Path(__file__).parent
REPO=Path('/workspace/personal')
REV='ad8c537edc6f0084d07a052bf3ab0c3ec9ceb2e8'
MAIN='e89c656eeeac0e93bafb4322b9482c7dfec70168'
TEAMS=['ann10','p_only10','ann_p_only20','tp10','ann_tp20','tp_phi10','ann_tp_phi20']
LABELS=['ANN10','P-only10','ANN+P20','T/P10','ANN+T/P20','T/P/phi10','ANN+T/P/phi20']
REFS={}


def git_tree(rev):
    data=subprocess.check_output(['git','ls-tree','-r',rev],cwd=REPO,text=True)
    return {v.split('\t',1)[1]:v.split('\t',1)[0].split()[2] for v in data.splitlines()}


TREE=git_tree(REV)
MAIN_TREE=git_tree(MAIN)


def source(path,alias=None):
    candidates=[ROOT/'sources/evidence'/path,REPO/path,Path('/workspace/raw_history_audit/run_001/sources')/path]
    for candidate in candidates:
        if candidate.is_file():
            raw=candidate.read_bytes()
            blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            if blob==TREE.get(path):
                REFS[path]=dict(path=path,git_blob_sha=blob,sha256=hashlib.sha256(raw).hexdigest(),
                    url='https://github.com/rom-cfd/personal/blob/'+REV+'/'+path,local=str(candidate))
                return candidate
    if alias and Path(alias).is_file():
        raw=Path(alias).read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob==TREE.get(path):
            REFS[path]=dict(path=path,git_blob_sha=blob,sha256=hashlib.sha256(raw).hexdigest(),url='https://github.com/rom-cfd/personal/blob/'+REV+'/'+path,local=str(alias))
            return Path(alias)
    archive=ROOT/'sources/evidence.zip'
    if archive.is_file():
        with zipfile.ZipFile(archive) as z:
            if path in z.namelist():
                raw=z.read(path)
                blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
                if blob!=TREE.get(path):raise ValueError('Archived source hash mismatch: '+path)
                cached=Path('/tmp/research-story-evidence')/REV/path
                cached.parent.mkdir(parents=True,exist_ok=True);cached.write_bytes(raw)
                REFS[path]=dict(path=path,git_blob_sha=blob,sha256=hashlib.sha256(raw).hexdigest(),
                    url='https://github.com/rom-cfd/personal/blob/'+REV+'/'+path,local=str(cached))
                return cached
    raise FileNotFoundError('Missing verified source: '+path)


def csv(path):return pd.read_csv(source(path))
def js(path):return json.loads(source(path).read_text())


def nrmse_stats(path):
    d=csv(path);out=[]
    for (method,ic),g in d.groupby(['method','trajectory']):
        assert len(g)==10 and g.variable.nunique()==10
        complete=bool(g.complete.all());v=g[g.variable!='N2'].nrmse.to_numpy()
        if complete:assert np.isfinite(v).all()
        first=g.iloc[0]
        row=dict(method=method,trajectory=int(ic),complete=complete,passed=bool(complete and max(v)<=.1),worst9=float(max(v)) if complete else None)
        for key in ['stage','gamma','lambda_value','team','training','model']:
            if key in first.index:row[key]=first[key].item() if hasattr(first[key],'item') else first[key]
        out.append(row)
    return pd.DataFrame(out)


def main():
    facts={'source_revision':REV,'main_revision':MAIN,'data_hash':'8d41f19f2145e12f9e89ee3eaa88fb0b1b93a84dcaaee65dcaa2e8b264077563'}
    main=csv('results/committee100_long10ms/summary.csv');main=main[(main.split=='test')&(main.horizon==.01)]
    facts['main_baseline']=main[['method','completed','physical_completed','temperature_profile_mean','species_profile_mean','ignition_classification']].to_dict('records')
    facts['series']={}
    for family in ['temporal_pilot','temporal_pilot_long','ann_multistep','ann_multistep_long']:
        root=f'results/{family}/run_001'
        d=csv(root+'/report/test_summary.csv')
        facts[family+'_all_predictors']=d.to_dict('records')
        d=d[d.predictor.str.endswith('mean3')]
        facts['series'][family]=d.to_dict('records')
        facts[family+'_training']=csv(root+'/training.csv').to_dict('records')
        js(root+'/verification.json');js(root+'/complete.json')
        facts[family+'_selection']=js(root+'/selection_frozen.json')
    base='results/penalty_weight_sweep/run_001/reports/report_1793729103103453738'
    lambdas=[]
    for mode in ['raw','repaired']:
        d=nrmse_stats(base+'/'+mode+'/per_IC_NRMSE.csv')
        for (weight,team),g in d.groupby(['lambda_value','team']):
            lambdas.append(dict(weight=float(weight),team=team,mode=mode,complete=int(g.complete.sum()),passed=int(g.passed.sum())))
    facts['lambda_short']=lambdas
    gamma=[]
    base='results/gamma_long/run_001/reports/report_1793852690561933910/test'
    for mode in ['raw','repaired']:
        d=nrmse_stats(base+'/'+mode+'/per_IC_NRMSE.csv')
        for (stage,value,team),g in d.groupby(['stage','gamma','team']):
            gamma.append(dict(stage=stage,gamma=float(value),team=team,mode=mode,complete=int(g.complete.sum()),passed=int(g.passed.sum())))
    facts['gamma']=gamma
    facts['gamma_validation_selection']=js('results/gamma_long/run_001/FINAL_ARTIFACTS.json')['validation_selection']
    soft=[]
    base='results/element_soft_long/run_001/reports/report_1793705344040514166'
    for mode in ['raw','repaired']:
        d=nrmse_stats(base+'/'+mode+'/per_IC_NRMSE.csv')
        for (training,method),g in d.groupby(['training','method']):
            soft.append(dict(training=training,team=method.split('_',1)[1],mode=mode,complete=int(g.complete.sum()),passed=int(g.passed.sum())))
    facts['soft10']=soft
    facts['raw_audit']=js('results/raw_history_audit/run_001/summary.json')
    facts['raw_passes']=csv('results/raw_history_audit/run_001/passing_configurations.csv').to_dict('records')
    facts['pass_variables']=csv('results/raw_history_audit/run_001/passing_variable_NRMSE.csv').to_dict('records')
    assert not any(x['joint9_le_0p1'] for x in facts['series']['ann_multistep_long'])
    facts['raw_audit_including_latest_ANN']={'configuration_IC_passes':3,'unique_ICs':[39],
        'reason':'The two new ordinary-ANN series have zero raw accuracy passes; the previous audit success set is unchanged.'}
    reference=source('results/long10ms_data/trajectories.npz')
    with np.load(reference) as d:
        truths=d['states'];species=d['species'].tolist();dt=float(d['dt']);initials=d['initials']
    selected={'IC7_reference':truths[7],'IC39_reference':truths[39],'IC74_reference':truths[74]}
    sources={}
    def extract(path,key,case,alias=None):
        p=source(path,alias=alias)
        with np.load(p) as d:
            name='IC_'+str(case) if 'IC_'+str(case) in d.files else 'trajectory_'+str(case)
            selected[key]=d[name]
        sources[key]={'path':path,'sha256':REFS[path]['sha256'],'IC':case}
    special='results/gamma_long/run_001/gamma_0.2/evaluations/ann_p_only20_repaired_test/attempt_1793851598429520706/traces.npz'
    extract(special,'IC7_gamma02_ANNP_repaired',7,alias='/workspace/gamma_case_search/run_001/sources/ann_p_only20_repaired_gamma02_long_traces.npz')
    for family in ['ann_multistep','ann_multistep_long','temporal_pilot_long']:
        for method in ['single_step','multistep']:
            for mode in ['raw','repaired']:
                path=f'results/{family}/run_001/evaluations/test/{method}_mean3_{mode}/traces.npz'
                extract(path,f'IC39_{family}_{method}_{mode}',39)
    old_path='results/deeponet_long_study/run_001/evaluations/tp10_raw_test/attempt_1793628655000147755/traces.npz'
    choices=[p for p in TREE if p.startswith('results/deeponet_long_study/run_001/evaluations/tp10_raw_test/') and p.endswith('traces.npz')]
    old_path=choices[0]
    extract(old_path,'IC39_legacy_TP_raw',39)
    gamma_path=next(p for p in TREE if p.startswith('results/gamma_long/run_001/gamma_0.2/evaluations/p_only10_raw_test/') and p.endswith('traces.npz'))
    extract(gamma_path,'IC39_gamma_P_raw',39)
    np.savez_compressed(ROOT/'sources/selected_traces.npz',**selected)
    (ROOT/'sources/selected_trace_provenance.json').write_text(json.dumps({'dt':dt,'species':species,'sources':sources},indent=2))
    predicted=selected['IC7_gamma02_ANNP_repaired'];ref=truths[7]
    ratios=np.linalg.norm(ref-predicted,axis=0)/np.linalg.norm(ref,axis=0)
    facts['IC7_trap']={'IC':7,'initial_T':float(ref[0,0]),'reference_delta_T':float(ref[-1,0]-ref[0,0]),
        'NN_delta_T':float(predicted[-1,0]-predicted[0,0]),'T_NRMSE':float(ratios[0]),'H2O_NRMSE':float(ratios[6]),'OH_NRMSE':float(ratios[5])}
    facts['IC39_initial']={'T_K':float(initials[39,0]),'P_atm':float(initials[39,1]/101325),'phi':float(initials[39,2])}
    sampling=csv('results/long10ms_data/sampling_summary.csv')
    facts['sampling']={'dense_pairs':int(sampling.total_candidates.sum()),'selected_pairs':int(sampling.selected.sum()),
        'dense_equilibrium':int(sampling.available_equilibrium.sum()),'selected_equilibrium':int(sampling.selected_equilibrium.sum())}
    source('results/long10ms_data/selected_inputs.csv')
    for root in ['deeponet_long_study','tp_phi_study','element_soft_long','weight_long_study','gamma_long']:
        facts[root+'_artifacts']=js(f'results/{root}/run_001/FINAL_ARTIFACTS.json')
    history=json.loads((ROOT/'sources/commit_history_complete.json').read_text())
    for c in history:c['date_KST']=datetime.fromisoformat(c['date'].replace('Z','+00:00')).astimezone(ZoneInfo('Asia/Seoul')).isoformat()
    facts['history']=history
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value):return None
        return value
    (ROOT/'sources/facts.json').write_text(json.dumps(clean(facts),indent=2,ensure_ascii=False,allow_nan=False))
    (ROOT/'sources/used_source_manifest.json').write_text(json.dumps(list(REFS.values()),indent=2))
    # Index all original PNG/PDF/SVG files without moving or downloading them.
    rows=[]
    for path,blob in TREE.items():
        if not path.startswith('results/') or not path.lower().endswith(('.png','.pdf','.svg')) or 'handoff_audit_20261102' in path:continue
        old=MAIN_TREE.get(path);scope='main 기준 자료' if old==blob else 'bw 추가/변경 자료'
        mode='raw' if '_raw' in path or '/raw/' in path else 'repaired' if 'repaired' in path or 'physics' in path else '원문 확인'
        rows.append(dict(experiment=path.split('/')[1],scope=scope,mode=mode,path=path,git_blob_sha=blob,
            url='https://github.com/rom-cfd/personal/blob/'+REV+'/'+path,
            raw_url='https://raw.githubusercontent.com/rom-cfd/personal/'+REV+'/'+path))
    pd.DataFrame(rows).to_csv(ROOT/'all_figures_index.csv',index=False)
    (ROOT/'sources/figure_inventory.json').write_text(json.dumps(rows,ensure_ascii=False))
    print('Facts prepared; source files',len(REFS),'indexed figures/PDFs',len(rows),'selected traces',len(selected))
    print('Metric trap',facts['IC7_trap'])


if __name__=='__main__':main()
