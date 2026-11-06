"""Page data for the disagreement-gated hybrid prototype (results/t1000_gated_hybrid)."""
import json,warnings
from pathlib import Path
import numpy as np,pandas as pd
warnings.filterwarnings('ignore')
OUT=Path('results/t1000_gated_hybrid');V=['gate_off','gate0.3_handover50K','gate0.3_handover200K']
SLOW=[78,69,52,7,9];dt=2e-6
tr={v:pd.read_csv(OUT/v/'trajectories.csv') for v in V};sm={v:pd.read_csv(OUT/v/'summary.csv') for v in V}
man={v:json.loads((OUT/v/'manifest.json').read_text()) for v in V};cost=json.loads((OUT/'ode_cost.json').read_text())
nn=lambda x:None if x is None or (isinstance(x,float) and not np.isfinite(x)) else (round(float(x),6) if isinstance(x,(float,np.floating)) else x)
summary=[]
for v in V:
    for _,r in sm[v].iterrows():
        g=tr[v][(tr[v].split==r.split)&(tr[v].horizon==r.horizon)]
        summary.append({k:nn(x) for k,x in dict(variant=v,split=r.split,horizon=r.horizon,completed=int(r.completed),physical=int(r.physical_completed),
            classification=r.ignition_classification,tau_median=r.median_ignition_error,tau_mean=r.mean_ignition_error,tau_max=float(g.tau400_relative_error_refined.max()),
            T_mean=r.temperature_profile_mean,T_median=float(g.groupby('trajectory').full_temperature_rmse.mean().median()),
            ode_fraction=r.ode_step_fraction,gated=int(r.gated_runs)).items()})
a=tr['gate_off'];b=tr['gate0.3_handover50K']
m=a.merge(b,on=['trajectory','seed','horizon'],suffixes=('_off','_on'));m=m[m.horizon==.02]
with np.load('results/t1000_reference/reference20ms.npz') as d:refs={int(i):s for i,s in zip(d['ids'],d['states'])};taus={int(i):t for i,t in zip(d['ids'],d['refined_tau400'])}
cases=[]
for i,g in m.groupby('trajectory'):
    cases.append({k:nn(x) for k,x in dict(id=int(i),split=g.split_off.iloc[0],T0=float(refs[i][0,0]),tau_ref=float(taus[i]),gated=int(g.gate_on_s_on.notna().sum()),
        gate_on_us=float(g.gate_on_s_on.median()*1e6) if g.gate_on_s_on.notna().any() else None,
        gate_off_ms=float(g.gate_off_s_on.median()*1e3) if g.gate_off_s_on.notna().any() else None,ode_steps=float(g.ode_steps_on.median()),
        tau_off=float(g.tau400_relative_error_refined_off.median()) if g.tau400_relative_error_refined_off.notna().any() else None,
        tau_on=float(g.tau400_relative_error_refined_on.median()),ok_off=int(g.ignition_classification_correct_off.sum()),ok_on=int(g.ignition_classification_correct_on.sum()),
        T_off=float(g.full_temperature_rmse_off.mean()),T_on=float(g.full_temperature_rmse_on.mean()),
        seeds=[dict(seed=int(r.seed),tau_off=nn(r.tau400_relative_error_refined_off),tau_on=nn(r.tau400_relative_error_refined_on),T_off=nn(r.full_temperature_rmse_off),T_on=nn(r.full_temperature_rmse_on),
            on_us=nn(r.gate_on_s_on*1e6),off_ms=nn(r.gate_off_s_on*1e3)) for _,r in g.sort_values('seed').iterrows()]).items()})
# traces for gated cases on a log grid
t=np.arange(10001)*dt;grid=np.unique(np.r_[np.round(np.geomspace(1,10000,360)).astype(int)])
TO=np.load(OUT/'gate_off/traces.npz');TN=np.load(OUT/'gate0.3_handover50K/traces.npz')
series={}
for i in SLOW:
    sel=np.flatnonzero(TN['case']==i);tau=taus[i]
    lin=np.unique(np.clip(np.round(np.linspace(.5*tau,1.5*tau,240)/dt).astype(int),1,10000))
    pts=np.unique(np.r_[grid,lin])
    series[str(i)]=dict(idx=pts.tolist(),Tref=np.round(refs[i][pts,0],2).tolist(),
        on=[np.round(TN['T'][n][pts],2).tolist() for n in sel],off=[np.round(TO['T'][n][pts],2).tolist() for n in sel],
        ode=[[int(np.flatnonzero(TN['on_ode'][n]).min()),int(np.flatnonzero(TN['on_ode'][n]).max())] if TN['on_ode'][n].any() else None for n in sel])
on=man['gate0.3_handover50K']
meta=dict(threshold=.3,window_us=20,detect_rise=50,ode_calls=on['ode_calls'],ode_ms_per_step=on['ode_seconds_per_step']*1e3,
    ode_continuous_total_s=cost['total_seconds'],ode_continuous_us_per_step=cost['total_seconds']/(36*10000)*1e6,
    wall={v:man[v]['wall_seconds'] for v in V},net={v:man[v]['network_seconds'] for v in V},
    max_trace_diff_off_vs_saved_K=0.000122)
json.dump(dict(summary=summary,cases=cases,series=series,meta=meta),open(OUT/'report_data.json','w'),separators=(',',':'))
print(pd.DataFrame(summary).round(4).to_string());print(json.dumps(meta,indent=1))
