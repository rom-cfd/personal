"""Reload raw increment committees and verify recorded first-step/seed identity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .delta_model import DeltaCommittee


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/delta100_raw'))
    root=parser.parse_args().root;out=root/'evaluation'
    manifest=json.loads((out/'manifest.json').read_text());assert manifest['status']=='completed'
    cfg=json.loads((root/'config.json').read_text())
    assert hashlib.sha256(Path(cfg['source']).read_bytes()).hexdigest()==manifest['source_sha256']
    assert hashlib.sha256(Path(cfg['training_pairs']).read_bytes()).hexdigest()==manifest['train_pair_sha256']
    with np.load(cfg['source']) as data:
        ids=np.flatnonzero(data['split']=='test')
        x=np.c_[data['states'][ids,0],data['pressures'][ids]]
        matrix=data['element_matrix']
    # Check discrete crossing admissibility independently from interpolated IDT.
    details=pd.read_csv(out/'trajectories.csv')
    for name in ('single_best','mean100','medoid100'):
        for partition in ('val','test'):
            d=details[(details.method==name)&(details.split==partition)].set_index('trajectory')
            with np.load(out/'traces'/f'{name}_{partition}.npz') as archive:
                for key in archive.files:
                    i=int(key.split('_')[-1]);trace=archive[key];y=trace[:,1:]
                    bad=(np.any(y < -1e-12,axis=1)|np.any(y>1+1e-12,axis=1)
                        |(trace[:,0]<200)|(trace[:,0]>6000)|(np.abs(y.sum(axis=1)-1)>.01)
                        |(np.max(np.abs((y-y[0])@matrix.T),axis=1)>.01))
                    failure=np.flatnonzero(bad);cross=np.flatnonzero(trace[:,0]>=trace[0,0]+400)
                    admissible=bool(len(cross) and (not len(failure) or cross[0]<failure[0]))
                    assert admissible==bool(d.loc[i,'ignition_physically_admissible'])
    p=DeltaCommittee.load(out/'methods/mean100.json');checks=[]
    for name,rule in [('mean100','mean'),('medoid100','medoid'),('single_best','single')]:
        desc=json.loads((out/'methods'/f'{name}.json').read_text());assert desc['rule']==rule
        if rule=='single':
            current=DeltaCommittee.load(out/'methods/single_best.json')
        else:
            current=p;current.rule=rule
            assert desc['paths']==p.paths and desc['metric']==p.metric
        actual=current.predict(x)
        with np.load(out/'traces'/f'{name}_test.npz') as archive:
            for j,i in enumerate(ids):
                trace=archive[f'trajectory_{int(i)}']
                if len(trace)>1:
                    np.testing.assert_allclose(actual[j],trace[1],atol=1e-8,rtol=1e-10)
                else:
                    assert not np.all(np.isfinite(actual[j])) or abs(actual[j,0])>1e6 or np.max(np.abs(actual[j,1:]))>1e3
        if rule in ('single','medoid'):
            chosen=current.last_diagnostics['selected_seed'].copy()
            events=pd.read_csv(out/f'selection_{name}_test.csv')
            np.testing.assert_array_equal(chosen,events[events.step==1].set_index('trajectory').loc[ids].selected_seed)
            candidates=current.candidates(x)
            for j,s in enumerate(chosen):
                if s>=0:
                    np.testing.assert_array_equal(actual[j],candidates[list(current.seeds).index(s),j])
        checks.append(dict(method=name,reload_first_step=True,
                           raw_candidate_identity=True if rule!='mean' else 'not_applicable'))
    pd.DataFrame(checks).to_csv(out/'reload_checks.csv',index=False)
    (out/'verified_source_hashes.json').write_text(json.dumps(
        {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('kinetics').glob('*.py'))},indent=2))
    print(pd.DataFrame(checks).to_string(index=False))


if __name__=='__main__':main()
