"""Summaries + page data for results/t1000_disagreement (see kinetics.t1000_disagreement)."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr

OUT=Path('results/t1000_disagreement');SEEDS=[11,22,33,44,55];TH=0.3;RISE=50.
d=np.load(OUT/'disagreement.npz');idx=d['idx'];t=idx*2e-6
C={c['id']:c for c in json.loads((OUT/'cases.json').read_text())['cases']}
tr=pd.read_csv('results/t1000_report/trajectories_all.csv');tr=tr[tr.approach.str.startswith('5')&(tr.horizon==.02)]
# plot grid: ~300 log-spaced sampled indices in (0, 20 ms]
targets=np.geomspace(4e-6,2e-2,320);pick=np.unique([np.argmin(np.abs(t-x)) for x in targets]);pick=pick[t[pick]>0]


def online(i,k):
    """Max over the pre-ignition part of the rollout (own T < T0+50 K) of R, 10-point rolling median."""
    T=d[f'T_{i}'][k,idx];R=d[f'roll_R_{i}'][k];m=(T<T[0]+RISE)&(t>0)
    if not m.any():return 0.,None
    Rs=pd.Series(R[m]).rolling(10,min_periods=1).median().to_numpy()
    return float(Rs.max()),(float(t[m][np.argmax(Rs>TH)]) if (Rs>TH).any() else None)


runs,cases,series=[],[],{}
for i,c in sorted(C.items(),key=lambda kv:kv[1]['tau_ref']):
    tau=c['tau_ref'];pre=t<tau;T=d[f'T_{i}'];Tref=d[f'Tref_{i}']
    sd=d[f'ref_sigma_D_{i}'];R=d[f'ref_R_{i}'];md=np.maximum(sd/R-1e-3,0)
    for k,s in enumerate(SEEDS):
        r=tr[(tr.trajectory==i)&(tr.seed==s)].iloc[0];mx,tf=online(i,k);err=np.abs(T[k]-Tref)
        te=float(np.argmax(err>RISE)*2e-6) if (err>RISE).any() else None
        runs.append(dict(id=i,split=c['split'],seed=s,maxR=mx,t_flag=tf,t_err50=te,
            R_pre_rollout=float(np.median(d[f'roll_R_{i}'][k][pre])),tau_err=None if np.isnan(r.tau400_relative_error_refined) else float(r.tau400_relative_error_refined),
            T_rmse=float(r.full_temperature_rmse),correct=bool(r.ignition_classification_correct),pred_tau=None if np.isnan(r.predicted_tau400) else float(r.predicted_tau400)))
        runs[-1]['bad']=(not runs[-1]['correct']) or (runs[-1]['tau_err'] or 0)>.1
    spread=T.std(0);
    cases.append(dict(id=i,split=c['split'],T0=c['T0'],P_atm=c['P_atm'],tau_ref=tau,
        R_pre_ref=float(np.median(R[pre])),noise=float(np.median(sd[pre])),signal=float(np.median(md[pre])),
        spread_max=float(spread.max()),err_max=float(np.abs(T-Tref).mean(0).max())))
    rr=d[f'roll_R_{i}'];ign=[(float(np.argmax(T[k]>T[k,0]+RISE)*2e-6) if (T[k]>T[k,0]+RISE).any() else None) for k in range(5)]
    f=lambda a,n=4:[float(f'{v:.{n}g}') if np.isfinite(v) else None for v in a]
    series[i]=dict(Tref=f(Tref[idx[pick]],5),T=[f(T[k,idx[pick]],5) for k in range(5)],R=[f(rr[k][pick],3) for k in range(5)],
        Rref=f(R[pick],3),spread=f(spread[idx[pick]],3),err=f(np.abs(T[:,idx[pick]]-Tref[idx[pick]]).mean(0),3),ign=ign)
R=pd.DataFrame(runs);K=pd.DataFrame(cases);R.to_csv(OUT/'runs.csv',index=False);K.to_csv(OUT/'cases_summary.csv',index=False)
val=R[R.split=='val'];te=R[R.split=='test']
stats=dict(threshold=TH,
    val_good_max=float(val[~val.bad].maxR.max()),val_bad_min=float(val[val.bad].maxR.min()),
    test_flag=int((te.maxR>TH).sum()),test_bad=int(te.bad.sum()),test_tp=int(((te.maxR>TH)&te.bad).sum()),test_fp=int(((te.maxR>TH)&~te.bad).sum()),
    all_flag=int((R.maxR>TH).sum()),all_bad=int(R.bad.sum()),all_tp=int(((R.maxR>TH)&R.bad).sum()),
    rho_Rpre_Trmse=float(spearmanr(R.R_pre_rollout,R.T_rmse)[0]),rho_maxR_Trmse=float(spearmanr(R.maxR,R.T_rmse)[0]),
    rho_within=float(np.nanmean([spearmanr(g.maxR,g.T_rmse)[0] for _,g in R[R.id.isin([52,69,78])].groupby('id')])),
    slow_min=float(R[R.id.isin([7,9,52,69,78])].groupby('id').maxR.median().min()),
    other_max=float(R[~R.id.isin([7,9,52,69,78])].groupby('id').maxR.median().max()))
json.dump(dict(time=f(t[pick],4),runs=runs,cases=cases,series={str(k):v for k,v in series.items()},stats=stats),
    open(OUT/'report_data.json','w'),separators=(',',':'))
print(json.dumps(stats,indent=1));print(R[R.id.isin([69,78])][['id','seed','maxR','t_flag','t_err50','tau_err','T_rmse']].round(5).to_string())
