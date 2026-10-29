"""Verify immutable data/model hashes and serialized 10/100-member decisions."""
import gc,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from .committee import Committee


def main():
    root=Path('results/committee100_long10ms');cfg=json.loads((root/'config.json').read_text())
    records=json.loads((root/'members.json').read_text())
    assert len(records)==100 and len({r['seed'] for r in records})==100
    assert len({r['path'] for r in records})==100
    for record in records:
        assert record['updates']==10000
        assert hashlib.sha256(Path(record['path']+'.keras').read_bytes()).hexdigest()==record['sha256']
    data=np.load(cfg['source']);pairs=np.load(cfg['training_pairs']);manifest=json.loads((root/'manifest.json').read_text())
    states=data['states'];pressures=data['pressures']
    assert hashlib.sha256(Path(cfg['source']).read_bytes()).hexdigest()==manifest['dense_sha256']
    assert hashlib.sha256(Path(cfg['training_pairs']).read_bytes()).hexdigest()==manifest['train_pair_sha256']
    assert float(data['dt'])==float(pairs['dt'])==2e-6
    train=set(np.flatnonzero(data['split']=='train'))
    assert set(pairs['parent'][:,0])<=train
    for x,y,(i,j) in zip(pairs['x'],pairs['y'],pairs['parent']):
        np.testing.assert_array_equal(x[:10],states[i,j])
        np.testing.assert_array_equal(y,states[i,j+1])
    checks=[]
    ids=np.flatnonzero(data['split']=='test');x=np.c_[states[ids,0],pressures[ids]]
    for method in ['mean10','medoid10','mean100','medoid100']:
        p=Committee.load(root/'methods'/f'{method}.json')
        predicted=p.predict(x);chosen=p.last_diagnostics['selected_seed'].copy()
        trace=np.load(root/'traces'/f'{method}_test.npz')
        expected=np.array([trace[f'trajectory_{int(i)}'][1] for i in ids])
        np.testing.assert_allclose(predicted,expected,rtol=1e-9,atol=1e-10,equal_nan=True)
        events=pd.read_csv(root/f'selection_{method}_test.csv')
        expected_seeds=events[events.step==1].sort_values('trajectory').selected_seed.to_numpy()
        np.testing.assert_array_equal(chosen,expected_seeds)
        if method.startswith('medoid'):
            candidates=p.repair(p.proposals(x),x)
            lookup={int(s):k for k,s in enumerate(p.seeds)}
            actual=np.array([candidates[lookup[int(seed)],j] for j,seed in enumerate(chosen)])
            np.testing.assert_allclose(predicted,actual,rtol=1e-9,atol=1e-10)
        checks.append(dict(method=method,prediction_reload=True,representative_reload=True,
                           actual_candidate_verified=method.startswith('medoid')))
        print('VERIFIED',method,flush=True);del p;gc.collect()
    pd.DataFrame(checks).to_csv(root/'reload_checks.csv',index=False)
    print('All hashes, fixed-dt training pairs and committee reloads verified.')


if __name__=='__main__':main()
