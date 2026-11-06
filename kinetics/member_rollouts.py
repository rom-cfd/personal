"""10/20-ms rollouts of one committee member alone (no selection/correction).

--physics wang (default): Wang 2025 hard element correction + enthalpy temperature every step.
--physics none: raw network output self-fed (T, P, Y from the ANN; no projection, clipping or closure).
Writes <root>/evaluation_{wang,raw}_members/seed<S>/ with the same files and assessment as
large_zhang_evaluate.py, so every seed can be compared case by case.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .multiscale_large_data import atomic_json,sha
from .zhang_model import ZhangCommittee
from .zhang_wang import WangHardCorrection
from .zhang_raw_study import raw_rollouts,assess
from .data import samples
from .metrics import evaluate
from .long_committee_study import horizon_summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('results/zhang_5p3m_gelu_ens/g0.1'))
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--source',type=Path,default=Path('results/long10ms_data/trajectories.npz'))
    ap.add_argument('--reference',type=Path,default=Path('results/zhang_multiscale_raw/reference20ms.npz'))
    ap.add_argument('--physics',choices=['wang','none'],default='wang')
    a=ap.parse_args();root=a.root;out=root/('evaluation_wang_members' if a.physics=='wang' else 'evaluation_raw_members')/f'seed{a.seed}'
    if out.exists():raise FileExistsError(out)
    folder=root/f'seed{a.seed}';r=json.loads((folder/'completed.json').read_text())
    assert r['status']=='completed' and r['model_sha256']==sha(folder/'model.keras')
    metric=json.loads((root/'preprocessing/metric.json').read_text())['metric']
    with np.load(a.source) as d:data={k:d[k] for k in d.files}
    with np.load(a.reference) as d:
        refs={int(i):x for i,x in zip(d['ids'],d['states'])};taus={int(i):t for i,t in zip(d['ids'],d['refined_tau400'])}
        assert float(d['dt'])==float(data['dt'])
    dt=float(data['dt']);out.mkdir(parents=True);(out/'traces').mkdir()
    p=ZhangCommittee([str(folder/'model')],[a.seed],metric,'single')
    if a.physics=='wang':p=WangHardCorrection(p)
    p.save(out/'method.json')
    atomic_json(out/'protocol.json',dict(seed=a.seed,physics=a.physics,correction='Wang 2025 Eqs. 5-9 + enthalpy T, decoder guard on' if a.physics=='wang' else 'none: raw ANN T, P and Y self-fed',selection=None,
        horizons=[.01,.02],model_sha256=r['model_sha256'],reference_sha256=sha(a.reference),source_sha256=sha(a.source)))
    name=f'seed{a.seed}';rows,summaries,one,timings=[],[],[],[]
    for split in ('test','val'):
        x,y,ids=samples(data,split);prediction=np.empty_like(y)
        for i in range(0,len(x),4096):prediction[i:i+4096]=p.predict(x[i:i+4096])
        with np.errstate(all='ignore'):one.append(dict(method=name,split=split,**evaluate(y,prediction,x,data['element_matrix'])))
        truth=np.asarray([refs[int(i)] for i in ids]);pressures=data['pressures'][ids];start=time.perf_counter()
        traces,statuses,events=raw_rollouts(p,truth,pressures,ids,dt,lambda:None)
        timings.append(dict(method=name,split=split,seconds=time.perf_counter()-start))
        np.savez_compressed(out/'traces'/f'{name}_{split}.npz',**{f'trajectory_{int(i)}':t for i,t in zip(ids,traces)})
        for horizon in [.01,.02]:
            rr=[]
            for i,t,ref,pressure,status in zip(ids,traces,truth,pressures,statuses):
                q,_=assess(t,ref,pressure,dt,data['element_matrix'],horizon,int(i),status,taus[int(i)])
                rr.append(q);rows.append(dict(method=name,split=split,horizon=horizon,**q))
            sm=horizon_summary(rr);summaries.append(dict(method=name,split=split,horizon=horizon,**sm))
            print('RESULT',name,split,horizon,{k:sm[k] for k in ('completed','physical_completed','ignition_classification','temperature_profile_mean')},flush=True)
    for fn,v in [('summary',summaries),('trajectories',rows),('one_step',one),('timing',timings)]:
        pd.DataFrame(v).to_csv(out/f'{fn}.csv',index=False)
    atomic_json(out/'manifest.json',dict(status='completed',seed=a.seed,evaluator_sha256=sha(__file__)))


if __name__=='__main__':main()
