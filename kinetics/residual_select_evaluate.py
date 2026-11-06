"""10/20-ms rollouts of the implicit-residual committee selection (Wang correction on).

Same reference, assessment and horizon summary as large_zhang_evaluate.py, so the
results compare directly with evaluation_wang_seeds*/summary.csv of the same root.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .multiscale_large_data import atomic_json,sha
from .zhang_model import ZhangCommittee
from .zhang_wang import WangHardCorrection
from .residual_select import ResidualSelect
from .zhang_raw_study import raw_rollouts,assess
from .data import samples
from .metrics import evaluate
from .long_committee_study import horizon_summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('results/zhang_5p3m_gelu_ens/g0.1'))
    ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55,66,77,88,99,110])
    ap.add_argument('--source',type=Path,default=Path('results/long10ms_data/trajectories.npz'))
    ap.add_argument('--reference',type=Path,default=Path('results/zhang_multiscale_raw/reference20ms.npz'))
    ap.add_argument('--splits',nargs='+',default=['test','val'])
    ap.add_argument('--induction-correction',action='store_true')
    ap.add_argument('--scaling',choices=['jacobian','diag','raw'],default='jacobian')
    ap.add_argument('--period',type=int,default=1,help='select by residual every N steps, hold the member in between')
    ap.add_argument('--output',type=Path)
    a=ap.parse_args();seeds=sorted(a.seeds);root=a.root
    tag='residual_select'+(f'_{a.scaling}' if a.scaling!='jacobian' else '')+(f'_every{a.period}' if a.period>1 else '')+('_induction_newton' if a.induction_correction else '')
    out=a.output or root/(f'evaluation_wang_{tag}_seeds'+'_'.join(map(str,seeds)))
    if out.exists():raise FileExistsError(out)
    metric=json.loads((root/'preprocessing/metric.json').read_text())['metric']
    paths=[str(root/f'seed{s}/model') for s in seeds]
    for s in seeds:
        r=json.loads((root/f'seed{s}/completed.json').read_text());assert r['status']=='completed' and r['model_sha256']==sha(root/f'seed{s}/model.keras')
    with np.load(a.source) as d:data={k:d[k] for k in d.files}
    with np.load(a.reference) as d:
        refs={int(i):r for i,r in zip(d['ids'],d['states'])};taus={int(i):t for i,t in zip(d['ids'],d['refined_tau400'])}
        assert float(d['dt'])==float(data['dt'])
    dt=float(data['dt']);out.mkdir(parents=True);(out/'traces').mkdir()
    p=ResidualSelect(WangHardCorrection(ZhangCommittee(paths,seeds,metric,'single' if len(seeds)==1 else 'mean')),dt,
        induction_correction=a.induction_correction,period=a.period,scaling=a.scaling);p.save(out/'method.json')
    atomic_json(out/'protocol.json',dict(selection='argmin scaled rms of one simplified-Newton trapezoidal step',
        induction_correction=dict(enabled=a.induction_correction,gate='T(x)<1300 K and dt*dT/dt(x)<0.5 K',
            newton='up to 3 simplified-Newton trapezoidal iterations, J(x) fixed; negative Y rejected; T from enthalpy'),
        correction='Wang 2025 Eqs. 5-9 + enthalpy T, decoder guard on',seeds=seeds,horizons=[.01,.02],
        reference_sha256=sha(a.reference),source_sha256=sha(a.source),metric=metric,test_used_for_design=False))
    began=time.perf_counter();rows,summaries,one,timings,shares=[],[],[],[],[]
    name=tag
    for split in a.splits:
        x,y,ids=samples(data,split);prediction=np.empty_like(y)
        if a.period>1:
            # Same held-choice rule along each trajectory's true states: one call per time step.
            nt=len(ids);xs=x.reshape(nt,-1,11);ps=prediction.reshape(nt,-1,10);p.reset()
            for t in range(xs.shape[1]):ps[:,t]=p.predict(xs[:,t])
        else:
            for i in range(0,len(x),2048):prediction[i:i+2048]=p.predict(x[i:i+2048])
        with np.errstate(all='ignore'):one.append(dict(method=name,split=split,**evaluate(y,prediction,x,data['element_matrix'])))
        truth=np.asarray([refs[int(i)] for i in ids]);pressures=data['pressures'][ids];start=time.perf_counter()
        p.reset();corr_flags=[];resel=[];base_predict=p.predict_full
        def tracked(z):
            r=base_predict(z);corr_flags.extend(p.last_diagnostics['corrected'].tolist())
            if 'reselected' in p.last_diagnostics:resel.extend(p.last_diagnostics['reselected'].tolist())
            return r
        p.predict_full=tracked
        traces,statuses,events=raw_rollouts(p,truth,pressures,ids,dt,lambda:None)
        p.predict_full=base_predict
        timings.append(dict(method=name,split=split,seconds=time.perf_counter()-start))
        np.savez_compressed(out/'traces'/f'{name}_{split}.npz',**{f'trajectory_{int(i)}':t for i,t in zip(ids,traces)})
        ev=pd.DataFrame(events)
        if a.period>1:atomic_json(out/f'reselection_{split}.json',dict(period=a.period,fraction_of_steps_with_residual=float(np.mean(resel)) if resel else 0.,
            member_switches_per_trajectory=float((ev.sort_values(['trajectory','step']).groupby('trajectory').selected_seed.apply(lambda q:(q.diff().fillna(0)!=0).sum())).mean())))
        if a.induction_correction:atomic_json(out/f'correction_share_{split}.json',dict(fraction_of_steps=float(np.mean(corr_flags)) if corr_flags else 0.))
        ev.to_csv(out/f'events_{name}_{split}.csv.gz',index=False)
        shares.append(dict(split=split,**{str(k):float(v) for k,v in ev.selected_seed.value_counts(normalize=True).sort_index().items()}))
        for horizon in [.01,.02]:
            rr=[]
            for i,t,ref,pressure,status in zip(ids,traces,truth,pressures,statuses):
                r,_=assess(t,ref,pressure,dt,data['element_matrix'],horizon,int(i),status,taus[int(i)])
                rr.append(r);rows.append(dict(method=name,split=split,horizon=horizon,**r))
            sm=horizon_summary(rr);summaries.append(dict(method=name,split=split,horizon=horizon,**sm))
            print('RESULT',name,split,horizon,sm,flush=True)
        for fn,v in [('summary',summaries),('trajectories',rows),('one_step',one),('timing',timings),('selection_share',shares)]:
            pd.DataFrame(v).to_csv(out/f'{fn}.csv',index=False)
    atomic_json(out/'manifest.json',dict(status='completed',execution_seconds=time.perf_counter()-began,evaluator_sha256=sha(__file__),
        selector_sha256=sha(Path(__file__).with_name('residual_select.py'))))


if __name__=='__main__':main()
