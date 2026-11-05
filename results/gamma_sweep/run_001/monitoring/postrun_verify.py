"""Independent saved-data checks; run only after the production process exits."""
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import zipfile

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='2'
import numpy as np

ROOT=Path('results/gamma_sweep/run_001')
started=time.time()
final=json.loads((ROOT/'FINAL_ARTIFACTS.json').read_text()); report=Path(final['report'])
snapshot=json.loads((ROOT/'run.json').read_text());cfg=snapshot['config']
attempt=sorted((ROOT/'process_launches').glob('attempt_*'))[-1]
exit_info=json.loads((attempt/'exit.json').read_text())
assert exit_info['returncode']==0 and not exit_info['timed_out']
assert json.loads((ROOT/'environment.json').read_text())['git_head']==json.loads((attempt/'start.json').read_text())['published_commit']
TEAMS=['ann10','p_only10','ann_p_only20','tp10','ann_tp20','tp_phi10','ann_tp_phi20']
VARIABLES=['T','H2','H','O','O2','OH','H2O','HO2','H2O2','N2']
GAMMAS=cfg['gammas']


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def close(actual,expected):
    if math.isnan(float(expected)):
        assert actual=='' or math.isnan(float(actual)),(actual,expected)
    elif math.isinf(float(expected)):
        assert float(actual)==float(expected),(actual,expected)
    else:
        assert actual!='' and math.isclose(float(actual),float(expected),rel_tol=3e-12,abs_tol=2e-14),(actual,expected)


tree=ast.parse(Path('references/user_nrmse/util.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_nrmse_1d')
namespace={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'references/user_nrmse/util.py','exec'),namespace)
user_nrmse=namespace['_nrmse_1d']
with np.load(snapshot['base_config']['source']) as f:
    truth=np.asarray(f['states'],dtype=np.float64);splits=f['split'];initials=f['initials']
with np.load(snapshot['base_config']['training_pairs']) as f:
    train_x=f['x'];train_y=f['y'];parent=f['parent']
frames={};scores=0;incomplete=0;undefined=0;summary_rows=0;cohorts={}


def verify_summaries(folder,rows):
    global summary_rows
    methods=list(dict.fromkeys(r['method'] for r in rows))
    complete={m:{int(r['trajectory']) for r in rows if r['method']==m and r['complete']=='True'} for m in methods}
    common=sorted(set.intersection(*(complete[m] for m in methods)))
    teams={r['method']:r['team'] for r in rows}
    within={t:sorted(set.intersection(*(complete[m] for m in methods if teams[m]==t))) for t in TEAMS}
    lookup={}
    for r in rows:lookup.setdefault((r['method'],r['variable']),[]).append(r)
    for name,local in [('common_IC_NRMSE_summary.csv',False),('within_team_NRMSE_summary.csv',True)]:
        data=read_csv(folder/name);assert len(data)==420
        for r in data:
            ids=within[r['team']] if local else common
            assert json.loads(r['common_ICs'])==ids and int(r['common_IC_count'])==len(ids)
            part=lookup[(r['method'],r['variable'])]
            values=np.array([float(x['nrmse']) for x in part if int(x['trajectory']) in ids and x['nrmse']!=''])
            assert int(r['valid_nrmse_IC_count'])==len(values)
            assert int(r['completed_all_ICs'])==sum(x['complete']=='True' for x in part)
            for field,value in [('median_nrmse',np.median(values) if len(values) else np.nan),('mean_nrmse',np.mean(values) if len(values) else np.nan),('std_nrmse',np.std(values) if len(values) else np.nan),('q25_nrmse',np.quantile(values,.25) if len(values) else np.nan),('q75_nrmse',np.quantile(values,.75) if len(values) else np.nan)]:close(r[field],value)
            summary_rows+=1
    return dict(common_ICs=common,within_team_common_ICs=within)


for split in ('val','test'):
    ids=np.flatnonzero(splits==split).astype(int).tolist();cohorts[split]={}
    for mode in ('repaired','raw'):
        folder=report/split/mode;rows=read_csv(folder/'per_IC_NRMSE.csv');assert len(rows)==7560
        frames[(split,mode)]=rows
        lookup={(r['method'],int(r['trajectory']),r['variable']):r for r in rows}
        for gamma in GAMMAS:
            for team in TEAMS:
                method=f'gamma_{gamma:g}_{team}'
                if gamma==.1 and split=='test':path=Path(snapshot['baseline_test_traces'][f'{team}_{mode}']['path'])
                else:
                    markers=list((ROOT/f'gamma_{gamma:g}'/'evaluations'/f'{team}_{mode}_{split}').glob('*/complete.json'));assert len(markers)==1
                    path=markers[0].parent/'traces.npz'
                with np.load(path) as f:
                    for i in ids:
                        prediction=np.asarray(f[f'trajectory_{i}'],dtype=np.float64)
                        np.testing.assert_array_equal(prediction[0],truth[i,0])
                        complete=len(prediction)==len(truth[i])
                        for k,variable in enumerate(VARIABLES):
                            r=lookup[(method,i,variable)];ref=truth[i,:,k]
                            assert int(r['points'])==len(prediction) and (r['complete']=='True')==complete
                            close(r['reference_rms'],np.sqrt(np.mean(ref**2)))
                            close(r['prefix_rmse'],np.sqrt(np.mean((ref[:len(prediction)]-prediction[:,k])**2)))
                            if not complete:
                                assert r['nrmse']=='' and r['full_rmse']=='';incomplete+=1;continue
                            close(r['full_rmse'],np.sqrt(np.mean((ref-prediction[:,k])**2)))
                            try:expected=user_nrmse(ref,prediction[:,k])
                            except ValueError:assert r['nrmse']=='';undefined+=1
                            else:close(r['nrmse'],expected);scores+=1
        cohorts[split][mode]=verify_summaries(folder,rows)

selection=json.loads((ROOT/'validation_selection.json').read_text())
assert selection['split']=='val' and selection['mode']=='repaired' and not selection['test_used']
val=[r for r in frames[('val','repaired')] if r['variable']=='T']
for team in TEAMS:
    completed={g:{int(r['trajectory']) for r in val if r['team']==team and float(r['gamma'])==g and r['complete']=='True'} for g in GAMMAS}
    maximum=max(map(len,completed.values()));eligible=[g for g in GAMMAS if len(completed[g])==maximum]
    common=sorted(set.intersection(*(completed[g] for g in eligible)))
    values={str(g):float(np.median([float(r['nrmse']) for r in val if r['team']==team and float(r['gamma'])==g and int(r['trajectory']) in common])) for g in eligible} if common else {}
    chosen=min(eligible,key=lambda g:(values[str(g)],g)) if values else None
    s=selection['choices'][team]
    assert s['chosen_gamma']==chosen and s['common_ICs']==common and s['maximum_completed_ICs']==maximum
    assert s['completion_counts']=={str(g):len(completed[g]) for g in GAMMAS}
    for key,value in values.items():close(s['temperature_NRMSE'][key],value)
for mode in ('repaired','raw'):
    rows=frames[('test',mode)]
    for r in read_csv(report/'test'/mode/'validation_selected_vs_reference.csv'):
        team=r['team'];chosen=selection['choices'][team]['chosen_gamma'];assert float(r['selected_gamma'])==chosen
        parts=[[p for p in rows if p['team']==team and float(p['gamma'])==g and p['variable']==r['variable']] for g in (.1,chosen)]
        common=sorted(set.intersection(*({int(p['trajectory']) for p in part if p['complete']=='True'} for part in parts)))
        assert json.loads(r['common_ICs'])==common and int(r['common_IC_count'])==len(common)
        for name,part in zip(('reference','selected'),parts):
            v=[float(p['nrmse']) for p in part if int(p['trajectory']) in common and p['nrmse']!='']
            close(r[name+'_median_NRMSE'],np.median(v) if v else np.nan)
            assert int(r[name+'_completed'])==sum(p['complete']=='True' for p in part)
            summary_rows+=1

records=json.loads((ROOT/'final_members.json').read_text());assert len(records)==240
expected_scales={}
for gamma in GAMMAS:
    a=np.c_[train_x[:,0]/1000,train_x[:,1:10]**gamma/gamma]
    b=np.c_[train_y[:,0]/1000,train_y[:,1:10]**gamma/gamma]
    both=np.r_[a,b];mean=both.mean(axis=0);scale=both.std(axis=0);scale=np.where(scale>1e-14,scale,1.)
    features=np.c_[(a-mean)/scale,np.log(train_x[:,-1])]
    im=features.mean(axis=0);isc=features.std(axis=0);isc=np.where(isc>1e-14,isc,1.)
    expected_scales[gamma]=dict(output_mean=mean,output_scale=scale,input_mean=im,input_scale=isc)
for r in records:
    info=json.loads(Path(r['path']+'.json').read_text());assert info['spec']['gamma']==r['gamma']
    for name,expected in expected_scales[r['gamma']].items():np.testing.assert_array_equal(info[name],expected)
    if r['family']=='tp_phi':
        physical_rows=np.c_[train_x[:,:10],initials[parent[:,0],2],train_x[:,-1]]
        values=physical_rows[:,10:11];np.testing.assert_array_equal(info['phi_mean'],values.mean(axis=0))
        scale=values.std(axis=0);np.testing.assert_array_equal(info['phi_scale'],np.where(scale>1e-14,scale,1.))
    if not r.get('reused'):
        cp=Path(r['checkpoint']);state=json.loads((cp/'state.json').read_text())
        with np.load(cp/'adam.npz') as f:assert int(f['v0'])==r['updates']==state['optimizer_iteration']
        assert r['member_budget']['seconds']==150 and r['member_budget']['expires_unix']-r['member_budget']['started_unix']==150
        assert r['reason']==('fixed_update_budget' if r['updates']==10000 else 'member_time_limit')
        assert state['signature']['loss']=='transformed_residual_MSE_only'

manifest=json.loads((report/'complete.json').read_text())['output_sha256']
assert digest(report/'complete.json')==final['report_complete_sha256']
for relative,expected in manifest.items():assert digest(report/relative)==expected,relative
pdfs={}
for p in report.rglob('*.pdf'):
    count=len(re.findall(rb'/Type\s*/Page\b',p.read_bytes()))
    expected=36 if p.name=='profiles_and_NRMSE.pdf' else 7 if p.name=='within_team_NRMSE.pdf' else 1
    assert count==expected,(str(p),count);pdfs[str(p.relative_to(report))]=count
archives={}
for p in report.glob('*.zip'):
    with zipfile.ZipFile(p) as z:assert z.testzip() is None;archives[p.name]=dict(bytes=p.stat().st_size,files=len(z.namelist()))
assert len(archives)==4
events=[json.loads(line) for line in (ROOT/'execution.jsonl').read_text().splitlines()]
freeze=next(i for i,r in enumerate(events) if r['phase']=='validation_selection_frozen')
first_test=next(i for i,r in enumerate(events) if r['phase']=='evaluation_finished' and r['split']=='test')
assert freeze<first_test
result=dict(exact_user_NRMSE_scores=scores,incomplete_variable_rows=incomplete,undefined_variable_rows=undefined,
    summary_values_checked=summary_rows,actual_Adam_iterations_verified=200,train_only_preprocessors_verified=240,
    validation_choices_independently_verified=7,validation_frozen_before_test=True,report_hashes_verified=len(manifest),
    pdf_page_objects=pdfs,zip_CRC_checked=archives,cohorts=cohorts,seconds=time.time()-started)
(ROOT/'postrun_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
analysis=dict(validation_selection=selection,selected_test_comparisons={mode:read_csv(report/'test'/mode/'validation_selected_vs_reference.csv') for mode in ('repaired','raw')},
    phase_seconds={r['phase']:r['seconds'] for r in events if r['phase'] in ('training_wall_finished','evaluation_wall_finished','report_finished')},
    pipeline_seconds=final['pipeline_seconds'],new_member_updates={str(g):dict(min=min(r['updates'] for r in records if r['gamma']==g),max=max(r['updates'] for r in records if r['gamma']==g)) for g in GAMMAS if g!=.1})
(ROOT/'analysis_summary.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
inventory={str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='all_artifacts_sha256.json'}
(ROOT/'all_artifacts_sha256.json').write_text(json.dumps(dict(count=len(inventory),files=inventory),indent=2)+'\n')
print(json.dumps(dict(status='passed',NRMSE_scores=scores,incomplete_rows=incomplete,summary_values=summary_rows,PDFs=len(pdfs),ZIPs=len(archives),inventory_files=len(inventory),seconds=time.time()-started)))
