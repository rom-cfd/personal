"""Recompute physical ignition labels from saved traces, without new inference.

A crossing whose endpoint is already invalid is not certified physically valid,
even if its interpolated crossing time precedes that discrete endpoint. Preserve
pre-audit labels and record all changed labels; no state/profile/IDT is changed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/delta100_raw'))
    root=parser.parse_args().root;out=root/'evaluation'
    cfg=json.loads((root/'config.json').read_text())
    with np.load(cfg['source']) as data:matrix=data['element_matrix']
    path=out/'trajectories.csv';details=pd.read_csv(path);corrected=details.copy();changes=[]
    for (name,partition),group in details.groupby(['method','split']):
        with np.load(out/'traces'/f'{name}_{partition}.npz') as archive:
            for index,row in group.iterrows():
                trace=archive[f'trajectory_{int(row.trajectory)}'];y=trace[:,1:]
                bad=(np.any(y < -1e-12,axis=1)|np.any(y>1+1e-12,axis=1)
                     |(trace[:,0]<200)|(trace[:,0]>6000)|(np.abs(y.sum(axis=1)-1)>.01)
                     |(np.max(np.abs((y-y[0])@matrix.T),axis=1)>.01))
                failure=np.flatnonzero(bad);cross=np.flatnonzero(trace[:,0]>=trace[0,0]+400)
                admissible=bool(len(cross) and (not len(failure) or cross[0]<failure[0]))
                if admissible!=bool(row.ignition_physically_admissible):
                    changes.append(dict(method=name,split=partition,trajectory=int(row.trajectory),
                        old=bool(row.ignition_physically_admissible),new=admissible,
                        first_cross_step=int(cross[0]) if len(cross) else None,
                        first_invalid_step=int(failure[0]) if len(failure) else None))
                corrected.loc[index,'ignition_physically_admissible']=admissible
                corrected.loc[index,'physical_tau400_relative_error']=row.tau400_relative_error if admissible else np.nan
    if changes:
        before=out/'trajectories_before_admissibility_audit.csv'
        if before.exists():raise FileExistsError('Pre-audit provenance already exists')
        before.write_bytes(path.read_bytes());corrected.to_csv(path,index=False)
        pd.DataFrame(changes).to_csv(out/'admissibility_label_changes.csv',index=False)
    manifest_path=out/'manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['physical_ignition_admissibility']='crossing index must precede first invalid index; audited from saved raw traces'
    manifest['inference_changed_by_audit']=False
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print('Admissibility label changes:',len(changes),changes)


if __name__=='__main__':main()
