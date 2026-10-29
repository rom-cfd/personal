"""Refine only reference ignition measurement; ANN/data timestep stays 2 us."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .posteriori import reference, ignition


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',default='results/verified');args=parser.parse_args()
    root=Path(args.study)/'posteriori'
    source=np.load(Path(args.study)/'dataset.npz')
    copied={k:source[k] for k in source.files}
    original_dt=float(copied['dt']);copied['dt']=original_dt/10
    states,ids,seconds=reference(copied,.002,root/'cantera_fine_reference.npz')
    pd.DataFrame([dict(trajectory=int(i),**ignition(states[k],original_dt/10,states[k,0,0])) for k,i in enumerate(ids)]).to_csv(root/'reference_ignition_refined.csv',index=False)
    (root/'reference_resolution.json').write_text(json.dumps(dict(ann_and_dataset_dt=original_dt,reference_diagnostic_dt=original_dt/10,reference_wall_seconds=seconds,
        purpose='Resolve ignition timing measurement only; no additional training/test samples or model tuning'),indent=2))
    print('Reference ignition measurement refined',seconds)


if __name__=='__main__':main()
