"""Validation-only ensemble assessment after all three fits are complete."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .ensemble import MeanPredictor
from .refinement import posteriori


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='results/refinement');args=parser.parse_args()
    root=Path(args.root);out=root/'ensemble_validation'
    if out.exists():raise FileExistsError('Ensemble assessment already exists')
    models=json.loads((root/'final_models.json').read_text())
    out.mkdir();ref=np.load(root/'validation_reference.npz');d=np.load('results/verified/dataset.npz')
    rows=[];details=[];entries=[];start=time.perf_counter()
    for name in sorted(set(m['name'] for m in models)):
        members=[m['path'] for m in models if m['name']==name]
        assert len(members)==3
        net=MeanPredictor(members)
        net.save(out/f'{name}.json')
        entries.append(dict(name=name+'_ensemble',seed=0,ensemble=True,members=members))
        for mode in ('raw','physics'):
            summary,rr,sp,traces=posteriori(net,ref['states'],ref['pressures'],ref['ids'],d['element_matrix'],float(d['dt']),mode)
            rows.append(dict(name=name+'_ensemble',seed=0,phase='ensemble',mode=mode,**summary))
            details.extend(dict(name=name+'_ensemble',seed=0,**r) for r in rr)
            np.savez_compressed(out/f'{name}_{mode}.npz',**{f'trajectory_{int(ref["ids"][i])}':t for i,t in traces})
            print('ENSEMBLE',name,mode,summary,flush=True)
    pd.DataFrame(rows).to_csv(out/'validation_summary.csv',index=False)
    pd.DataFrame(details).to_csv(out/'validation_trajectories.csv',index=False)
    (out/'models.json').write_text(json.dumps(entries,indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,test_accessed=False,architecture='3 independently trained 64x64x64 DNNs; physical-output mean'),indent=2))


if __name__=='__main__':main()
