"""Verify completed gamma-ablation models, selection, provenance and pytest results."""
import argparse, hashlib, json, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/preprocess_tanh10'))
    root=parser.parse_args().root
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='completed'
    config=json.loads((root/'config.json').read_text())
    for key, digest in manifest['input_hashes'].items():
        assert hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()==digest
    for name, digest in manifest['source_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, name
    count=0
    ranks=[]
    for gamma in config['gammas']:
        out=root/f'g{gamma:g}'
        records=json.loads((out/'members.json').read_text())
        assert len(records)==10 and len({r['seed'] for r in records})==10
        history=pd.read_csv(out/'history.csv')
        assert (history.groupby('seed')['update'].max()==10000).all()
        for record in records:
            path=Path(record['path']+'.keras')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
            metadata=Path(record['path']+'.json')
            assert hashlib.sha256(metadata.read_bytes()).hexdigest()==record['metadata_sha256']
            assert json.loads(metadata.read_text())['spec']['gamma']==gamma
            assert record['updates']==10000
            with zipfile.ZipFile(path) as archive:
                layers=json.loads(archive.read('config.json'))['config']['layers']
            dense=[layer['config'] for layer in layers if layer['class_name']=='Dense']
            assert [d['units'] for d in dense]==[64,64,64,10]
            assert [d['activation'] for d in dense]==['tanh','tanh','tanh','linear']
            count+=1
        rank=pd.read_csv(out/'validation_member_ranking.csv')
        assert len(rank)==10 and (rank.samples==90000).all()
        chosen=json.loads((out/'selection_frozen.json').read_text())
        assert chosen['selected_seed']==int(rank.sort_values(['selection_score','seed']).iloc[0].seed)
        ranks.extend(dict(gamma=gamma,seed=int(r.seed),score=r.selection_score) for r in rank.itertuples())
    common=json.loads((root/'global_single_frozen.json').read_text())
    assert (common['gamma'],common['selected_seed'])==min(((r['gamma'],r['seed']) for r in ranks),key=lambda k:next(r['score'] for r in ranks if (r['gamma'],r['seed'])==k))
    summary=pd.read_csv(root/'summary.csv')
    checks=pd.read_csv(root/'reload_checks.csv')
    assert len(summary)==len(checks)==12
    assert checks[['reload_match','first_step_match','singleton_seed_fixed']].all().all()
    assert (summary.trajectories==18).all()
    one=pd.read_csv(root/'one_step.csv')
    assert len(one)==12 and (one.samples==90000).all()
    for gamma in config['gammas']:
        val=one[(one.gamma==gamma)&(one.method=='single_best')&(one.split=='val')].iloc[0]
        selected=json.loads((root/f'g{gamma:g}'/'selection_frozen.json').read_text())
        np.testing.assert_allclose(val.selection_score,selected['validation_score'],rtol=1e-7,atol=1e-10)
    suites=list(ET.parse(root/'pytest.xml').getroot().iter('testsuite'))
    counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
    assert counts['tests']>0 and counts['failures']==counts['errors']==counts['skipped']==0
    result=dict(verified_fresh_models=count,architecture='64/64/64 tanh',updates_per_member=10000,
                input_hashes_unchanged=True,source_hashes_match=True,selection_validation_only=True,
                common_control_gamma=common['gamma'],common_control_seed=common['selected_seed'],
                verified_method_split_pairs=len(checks),full_one_step_pairs_per_split=90000,
                **{'pytest_'+key:value for key,value in counts.items()})
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
