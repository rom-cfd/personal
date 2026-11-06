"""A priori test: does a smaller Wang correction mean a more accurate candidate?

For every one-step pair, each committee member proposes a next state; each is
Wang-corrected (decoder guard, Eqs. 5-9, enthalpy T). We compare the size of
the correction with the corrected candidate's true error against Cantera and
score selection rules. All distance scales are the stored train-only medoid
metric. No test data are used to define anything; test is reported only as a
second split.

Correction sizes (fixed before looking at results):
  species  sqrt(mean_j ((Y_corr-Y_raw)/sY_j)^2)                (Wang Y change only)
  full     sqrt(((T_corr-T_ann)/sT)^2 + sum_j((Y_corr-Y_raw)/sY_j)^2)/sqrt(10)
           (the MIN_CORRECTION_REPORT.md criterion)
Error: same scaled T/Y distance between corrected candidate and truth.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .zhang_model import ZhangCommittee
from .zhang_wang import WangHardCorrection
from .data import samples
from .multiscale_large_data import atomic_json


def scaled(a,metric):
    return np.concatenate([a[...,:1]/metric['temperature_scale'],a[...,1:10]/np.asarray(metric['species_scale'])],axis=-1)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('results/zhang_5p3m_gelu_ens/g0.1'))
    ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55,66,77,88,99,110])
    ap.add_argument('--source',type=Path,default=Path('results/long10ms_data/trajectories.npz'))
    ap.add_argument('--single',type=int,default=99,help='validation-frozen single seed (selection_frozen.json)')
    ap.add_argument('--output',type=Path)
    a=ap.parse_args();out=a.output or a.root/'min_correction_apriori'
    if out.exists():raise FileExistsError(out)
    metric=json.loads((a.root/'preprocessing/metric.json').read_text())['metric']
    paths=[str(a.root/f'seed{s}/model') for s in a.seeds]
    p=WangHardCorrection(ZhangCommittee(paths,a.seeds,metric,'mean'))
    with np.load(a.source) as d:data={k:d[k] for k in d.files}
    out.mkdir(parents=True);rows=[];bins=[];hits=[]
    for split in ('val','test'):
        x,y,_=samples(data,split);n=len(x);M=len(a.seeds)
        raw=np.empty((M,n,11));cor=np.empty((M,n,11))
        for i in range(0,n,2048):
            r=p.raw_candidates(x[i:i+2048]);raw[:,i:i+2048]=r
            cor[:,i:i+2048]=np.concatenate([p.thermo.repair(r[...,:10],x[i:i+2048]),r[...,10:]],axis=-1)
        ok=np.isfinite(cor[...,:10]).all(-1)
        zs=scaled(cor,metric);zr=scaled(raw,metric);zt=scaled(y,metric)[None];zx=scaled(x,metric)[None]
        c_sp=np.sqrt(np.mean((zs[...,1:]-zr[...,1:])**2,-1))
        c_full=np.sqrt(np.sum((zs-zr)**2,-1)/10)
        err=np.sqrt(np.mean((zs-zt)**2,-1))
        move_pred=np.sqrt(np.mean((zs-zx)**2,-1));move_true=np.sqrt(np.mean((zt-zx)**2,-1))[0]
        for A in (c_sp,c_full,err):A[~ok]=np.nan
        # Selection rules
        pick={}
        for name,c in (('min_species',c_sp),('min_full',c_full),('oracle',err)):
            pick[name]=np.nanargmin(np.where(ok,c,np.inf),axis=0)
        pick['single']=np.full(n,a.seeds.index(a.single))
        idx=np.arange(n)
        state={k:cor[v,idx,:10] for k,v in pick.items()}
        mean=np.where(ok[...,None],cor[...,:10],0).sum(0)/ok.sum(0)[:,None]
        hs,_=p.thermo.species_h_cp(x[:,0]);h=np.sum(x[:,1:10]/x[:,1:10].sum(1)[:,None]*hs,1)
        mean[:,0]=p.thermo.temperatures(mean[:,1:10],x[:,0],h,x[:,-1]);state['mean10']=mean
        from .committee import choose_medoid
        t=cor[...,:10].copy();t[~ok]=np.nan;state['medoid10'],_,_=choose_medoid(t,metric,'scaled')
        for k,s in state.items():
            e=np.sqrt(np.mean((scaled(s,metric)-zt[0])**2,-1));mv=np.sqrt(np.mean((scaled(s,metric)-zx[0])**2,-1))
            rows.append(dict(split=split,rule=k,scaled_error_mean=float(np.mean(e)),scaled_error_p95=float(np.quantile(e,.95)),
                species_rmse=float(np.sqrt(np.mean((s[:,1:10]-y[:,1:10])**2))),temperature_rmse=float(np.sqrt(np.mean((s[:,0]-y[:,0])**2))),
                predicted_to_true_move_median=float(np.median(mv[move_true>0]/move_true[move_true>0])),
                oracle_hit_rate=float(np.mean(pick[k]==pick['oracle'])) if k in pick else np.nan))
        # Does smaller correction mean smaller error?  Pooled and within-state (the selection-relevant one).
        for name,c in (('species',c_sp),('full',c_full)):
            good=ok.all(0);cc=c[:,good];ee=err[:,good]
            within=[spearmanr(cc[:,j],ee[:,j]).correlation for j in range(0,cc.shape[1],7)]
            hits.append(dict(split=split,correction=name,pooled_spearman=float(spearmanr(cc.ravel(),ee.ravel()).correlation),
                within_state_spearman_mean=float(np.nanmean(within)),within_state_spearman_median=float(np.nanmedian(within)),
                oracle_hit_rate=float(np.mean(pick['min_'+name]==pick['oracle'])),chance=1/M,
                min_pick_error_rank_mean=float(np.mean(np.argsort(np.argsort(ee,0),0)[np.argmin(cc,0),np.arange(cc.shape[1])]+1))))
        # Regime check: by true one-step change magnitude (quartiles fitted on this split's truth only, descriptive).
        q=np.quantile(move_true,[.25,.5,.75]);b=np.digitize(move_true,q)
        for k in ('single','min_species','min_full','mean10','oracle'):
            s=state[k];mv=np.sqrt(np.mean((scaled(s,metric)-zx[0])**2,-1));e=np.sqrt(np.mean((scaled(s,metric)-zt[0])**2,-1))
            for j in range(4):
                m=b==j
                bins.append(dict(split=split,rule=k,true_move_quartile=j+1,samples=int(m.sum()),scaled_error_mean=float(e[m].mean()),
                    predicted_to_true_move_median=float(np.median(mv[m&(move_true>0)]/move_true[m&(move_true>0)]))))
        sel=pd.Series(np.asarray(a.seeds)[pick['min_species']]).value_counts(normalize=True).sort_index()
        atomic_json(out/f'selection_share_{split}.json',{str(k):float(v) for k,v in sel.items()})
        print('DONE',split,flush=True)
    pd.DataFrame(rows).to_csv(out/'rules.csv',index=False);pd.DataFrame(hits).to_csv(out/'correlation.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'by_regime.csv',index=False)
    atomic_json(out/'manifest.json',dict(status='completed',root=str(a.root),seeds=a.seeds,single=a.single,metric='stored train-only medoid metric',
        guard=True,correction='Wang 2025 Eqs. 5-9 + enthalpy T',note='test split reported for reference only; rules fixed before evaluation'))


if __name__=='__main__':main()
