"""Evaluation cases for the T >= 1000 K pipeline.

Same sampler, seed (20261029), split and P/phi draws as the original 90 cases; only
T0 ~ U(1000, 1600) K instead of U(900, 1600) K. Then the 10-ms long trajectories
(long_data.generate) and the 20-ms references with refined tau400 for val+test
(zhang_raw_study.build_references), all Cantera CVODE rtol 1e-10 / atol 1e-20.
"""
from pathlib import Path
import numpy as np
from .data import generate
from .long_data import generate as long_generate
from .zhang_raw_study import build_references

ROOT=Path('results/t1000_cases')


def main():
    cases=ROOT/'dataset.npz';long=Path('results/t1000_long10ms_data');ref=Path('results/t1000_reference')
    if not cases.exists():print('CASES',generate(cases,temperature_range=(1000,1600)),flush=True)
    if not (long/'manifest.json').exists():long_generate(cases,long)
    if not (ref/'reference20ms.npz').exists():
        ref.mkdir(parents=True,exist_ok=True)
        with np.load(long/'trajectories.npz') as d:data={k:d[k] for k in d.files}
        ids=np.flatnonzero(data['split']!='train')
        build_references(data,ids,float(data['dt']),.02,ref,lambda:None)
    with np.load(long/'trajectories.npz') as d:
        print('T0_RANGE',float(d['states'][:,0,0].min()),float(d['states'][:,0,0].max()),flush=True)
    print('CASES_DONE',flush=True)


if __name__=='__main__':main()
