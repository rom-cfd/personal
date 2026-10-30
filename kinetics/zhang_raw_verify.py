"""Audit saved raw Zhang models, training data, end times and replayed transitions."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from .zhang_model import ZhangCommittee,ZhangPredictor


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('results/zhang_multiscale_raw'))
    args=parser.parse_args();root=args.root
    manifest=json.loads((root/'manifest.json').read_text());cfg=json.loads((root/'config.json').read_text())
    assert manifest['status']=='completed' and manifest['physics'] is False
    for name,digest in manifest['input_hashes'].items():assert sha(name)==digest,name
    renderer=json.loads((root/'report_manifest.json').read_text())
    assert sha(root/'manifest.json')==renderer['simulation_manifest_sha256']
    assert sha('kinetics/zhang_raw_report.py')==renderer['renderer_sha256']
    for name,digest in manifest['source_hashes'].items():
        if name!='kinetics/zhang_raw_report.py':assert sha(name)==digest,name
    frozen=json.loads((root/'all_selections_frozen.json').read_text())
    assert len(frozen)==4
    count=0
    for selection in frozen:
        out=root/selection['label'];records=json.loads((out/'members.json').read_text())
        assert len(records)==10 and len({r['seed'] for r in records})==10
        history=pd.read_csv(out/'history.csv')
        assert (history.groupby('seed')['update'].max()==cfg['updates_per_member']).all()
        rank=pd.read_csv(out/'validation_member_ranking.csv').sort_values(['selection_score','seed'])
        assert selection['selected_seed']==int(rank.iloc[0].seed) and (rank.samples==90000).all()
        assert not selection['test_used'] and not selection['rollout_used']
        for record in records:
            assert sha(record['path']+'.keras')==record['sha256']
            assert sha(record['path']+'.json')==record['metadata_sha256']
            info=json.loads(Path(record['path']+'.json').read_text())
            assert info['model_type']=='zhang_full_transformed_increment_raw' and info['physics'] is False
            assert info['spec']['gamma']==selection['gamma']
            with zipfile.ZipFile(record['path']+'.keras') as z:layers=json.loads(z.read('config.json'))['config']['layers']
            dense=[l['config'] for l in layers if l['class_name']=='Dense']
            assert [l['units'] for l in dense]==[64,64,64,11]
            assert [l['activation'] for l in dense]==['tanh','tanh','tanh','linear']
            count+=1
        regime=next(r for r in cfg['regimes'] if r['name']==selection['regime'])
        with np.load(regime['training_pairs']) as d:x,y=d['x'],d['y']
        p=ZhangPredictor.load(records[0]['path'])
        encoded=p.encode(x);delta=p.encode(np.c_[y,x[:,-1]])-encoded
        np.testing.assert_allclose(p.input_norm.mean,encoded.mean(axis=0),rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(p.output_norm.mean,delta.mean(axis=0),rtol=1e-12,atol=1e-14)
        # Pressure is an 11th NN output: target is zero, no runtime pinning.
        assert np.max(np.abs(delta[:,1]))==0
    detail=pd.read_csv(root/'trajectories.csv');summary=pd.read_csv(root/'summary.csv')
    checks=pd.read_csv(root/'reload_checks.csv')
    assert len(detail)==864 and len(summary)==48 and len(checks)==24
    assert checks[['reload_match','first_self_fed_step_match']].all().all() and not checks.physics.any()
    with np.load(cfg['source']) as d:split=d['split'];matrix=d['element_matrix']
    with np.load(root/'reference20ms.npz') as d:
        refs={int(i):r for i,r in zip(d['ids'],d['states'])}
        assert d['states'].shape==(36,10001,10)
    replays=0;prefix_metrics=0
    for selection in frozen:
        out=root/selection['label']
        for method in ['single_best','mean10','medoid10']:
            p=ZhangCommittee.load(out/'methods'/f'{method}.json')
            for part in ['val','test']:
                ids=np.flatnonzero(split==part)
                with np.load(out/'traces'/f'{method}_{part}.npz') as d:traces={int(i):d[f'trajectory_{int(i)}'] for i in ids}
                events=pd.read_csv(out/f'events_{method}_{part}.csv.gz').set_index(['trajectory','step'])
                for step in [1,100,500,5000,10000]:
                    active=[int(i) for i in ids if len(traces[int(i)])>=step and (int(i),step) in events.index]
                    if not active:continue
                    current=np.array([traces[i][step-1] for i in active]);pred=p.predict_full(current)
                    logged=events.loc[[(i,step) for i in active]]
                    np.testing.assert_array_equal(logged.selected_seed,p.last_diagnostics['selected_seed'])
                    np.testing.assert_array_equal(logged.finite_members,p.last_diagnostics['valid_count'])
                    for i,y,e in zip(active,pred,logged.itertuples()):
                        finite=np.isfinite(y).all()
                        huge=bool(finite and (abs(y[0])>1e6 or np.max(np.abs(y[1:10]))>1e3 or abs(y[-1])>1e15))
                        assert bool(e.accepted)==bool(finite and not huge)
                        if e.accepted:np.testing.assert_allclose(y,traces[i][step],rtol=1e-8,atol=1e-8)
                        replays+=1
                for i in ids:
                    a=traces[int(i)];truth=refs[int(i)]
                    for horizon in cfg['horizons']:
                        rows=detail[(detail.regime==selection['regime'])&np.isclose(detail.gamma,selection['gamma'])&
                            (detail.method==method)&(detail.split==part)&(detail.trajectory==i)&np.isclose(detail.horizon,horizon)]
                        assert len(rows)==1;r=rows.iloc[0];n=min(len(a),round(horizon/cfg['dt'])+1)
                        assert r.completed_steps==n-1 and bool(r.complete)==(n==round(horizon/cfg['dt'])+1)
                        np.testing.assert_allclose(r.mass_sum_max,np.max(np.abs(a[:n,1:10].sum(axis=1)-1)),rtol=1e-9,atol=1e-12)
                        if r.complete:
                            rmse=np.sqrt(np.trapezoid((a[:n,0]-truth[:n,0])**2,dx=cfg['dt'])/horizon)
                            np.testing.assert_allclose(r.full_temperature_rmse,rmse,rtol=1e-9,atol=1e-9)
                        else:assert np.isnan(r.full_temperature_rmse)
                        prefix_metrics+=1
    suites=list(ET.parse(root/'pytest.xml').getroot().iter('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    assert counts['tests']>0 and counts['failures']==counts['errors']==counts['skipped']==0
    result=dict(fresh_models_verified=count,output_dimension=11,hidden_layers=[64,64,64],activation='tanh',physics=False,
        input_hashes_unchanged=True,simulation_source_hashes_match=True,renderer_source_hash_matches=True,
        train_only_normalizers_verified=True,validation_only_selections_verified=4,
        replayed_transitions=replays,independently_checked_horizon_rows=prefix_metrics,
        reference_shape=[36,10001,10],horizons=cfg['horizons'],
        **{'pytest_'+k:v for k,v in counts.items()},verifier_sha256=sha(__file__))
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
