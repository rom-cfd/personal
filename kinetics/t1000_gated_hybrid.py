"""Disagreement-gated hybrid rollouts: strategy 5 network + Cantera ODE in the untrusted regime.

Each run = (case, seed). The network step is that seed's Wang-corrected prediction (identical to the strategy-5
member rollouts in evaluation_wang_members). At every step the five best-validation members are queried on the
current state (one batched call; the run's own seed is one of them) and the relative disagreement
R = ||std_k D_k|| / (||mean_k D_k|| + 1e-3) is formed as in kinetics.t1000_disagreement.

Gate (fixed before looking at test): while the run is pre-ignition (T < T0 + 50 K) and the rolling median of R over
the last 10 steps (20 us) exceeds --threshold (0.3, chosen from validation), the run latches onto Cantera
(same mechanism/reactor/tolerances as the reference) and advances dt per step from its current state. It hands
back to the network for good once T >= T0 + --handover K. --threshold inf reproduces strategy 5 (sanity check).
Writes results/t1000_gated_hybrid/<variant>/.
"""
import argparse,json,time
from pathlib import Path
import cantera as ct
import numpy as np
import pandas as pd
from .multiscale_large_data import atomic_json,sha
from .zhang_model import ZhangCommittee
from .zhang_wang import WangHardCorrection
from .zhang_raw_study import assess
from .long_committee_study import horizon_summary
from .data import mechanism

ROOT=Path('results/t1000_ur10_phys_best/g0.1');SEEDS=(11,22,33,44,55)
SRC=Path('results/t1000_long10ms_data/trajectories.npz');REF=Path('results/t1000_reference/reference20ms.npz')
OUT=Path('results/t1000_gated_hybrid');EPS=1e-3;WINDOW=10;DETECT_RISE=50.


def encode(s,gamma):
    with np.errstate(invalid='ignore',divide='ignore'):
        return np.concatenate([s[...,:1],(np.maximum(s[...,1:10],0)**gamma-1)/gamma],axis=-1)


class ODE:
    def __init__(self,dt):self.gas=mechanism();self.dt=dt;self.calls=0;self.seconds=0.
    def step(self,s):
        t0=time.perf_counter();g=self.gas;g.TPY=s[0],s[-1],np.maximum(s[1:10],0)
        r=ct.IdealGasConstPressureReactor(g,clone=True);net=ct.ReactorNet([r]);net.rtol=1e-10;net.atol=1e-20
        net.advance(self.dt);out=np.r_[r.T,np.maximum(r.phase.Y,0),s[-1]]
        self.calls+=1;self.seconds+=time.perf_counter()-t0;return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--threshold',type=float,default=.3);ap.add_argument('--handover',type=float,default=50.)
    ap.add_argument('--name');ap.add_argument('--steps',type=int,default=10000,help='debug only');a=ap.parse_args()
    name=a.name or ('gate_off' if not np.isfinite(a.threshold) else f'gate{a.threshold:g}_handover{a.handover:g}K')
    out=OUT/name
    if (out/'manifest.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    metric=json.loads((ROOT/'preprocessing/metric.json').read_text())['metric']
    for s in SEEDS:
        r=json.loads((ROOT/f'seed{s}/completed.json').read_text());assert r['model_sha256']==sha(ROOT/f'seed{s}/model.keras')
    w=WangHardCorrection(ZhangCommittee([str(ROOT/f'seed{s}/model') for s in SEEDS],list(SEEDS),metric,'mean'))
    base=w.predictor.base;gamma=base.gamma;scale=np.r_[base.output_norm.scale[0],base.output_norm.scale[2:]]
    with np.load(SRC) as d:split=d['split'];pressures=d['pressures'];matrix=d['element_matrix'];dt=float(d['dt'])
    with np.load(REF) as d:refs={int(i):x for i,x in zip(d['ids'],d['states'])};taus={int(i):t for i,t in zip(d['ids'],d['refined_tau400'])}
    ids=sorted(refs);runs=[(i,k) for i in ids for k in range(len(SEEDS))];N=len(runs);steps=a.steps
    case=np.array([i for i,_ in runs]);member=np.array([k for _,k in runs])
    state=np.stack([np.r_[refs[i][0],pressures[i]] for i in case]);T0=state[:,0].copy()
    hist=np.full((N,steps+1,11),np.nan);hist[:,0]=state;Rhist=np.full((N,steps+1),np.nan)
    on_ode=np.zeros((N,steps+1),bool);Rbuf=np.full((N,WINDOW),np.nan)
    latched=np.zeros(N,bool);done_gate=np.zeros(N,bool);active=np.ones(N,bool);status=['completed']*N
    gate_on=np.full(N,np.nan);gate_off=np.full(N,np.nan);ode=ODE(dt);net_seconds=0.;t_start=time.perf_counter()
    for j in range(1,steps+1):
        live=np.flatnonzero(active)
        if not len(live):break
        x=state[live];t0=time.perf_counter()
        c=w.candidates(x)                                    # (5,n,11) corrected next states, one per member
        ok=np.isfinite(c).all(-1);D=(encode(c[...,:10],gamma)-encode(x[:,:10],gamma)[None])/scale
        D[~ok]=np.nan
        with np.errstate(all='ignore'):
            R=np.sqrt(np.nansum(np.nanvar(D,axis=0),-1))/(np.linalg.norm(np.nanmean(D,axis=0),axis=-1)+EPS)
        net_seconds+=time.perf_counter()-t0
        Rhist[live,j-1]=R;Rbuf[live,(j-1)%WINDOW]=R
        with np.errstate(all='ignore'):Rs=np.nanmedian(Rbuf[live],axis=1)
        pre=x[:,0]<T0[live]+DETECT_RISE
        fire=(~latched[live])&(~done_gate[live])&pre&(Rs>a.threshold)
        latched[live[fire]]=True;gate_on[live[fire]]=(j-1)*dt
        for q,n in enumerate(live):
            if latched[n] and state[n,0]>=T0[n]+a.handover:
                latched[n]=False;done_gate[n]=True;gate_off[n]=(j-1)*dt
            if latched[n]:
                y=ode.step(state[n]);on_ode[n,j]=True
            else:
                y=c[member[n],q]
                finite=np.isfinite(y).all()
                huge=bool(finite and (abs(y[0])>1e6 or np.max(np.abs(y[1:10]))>1e3 or abs(y[-1])>1e15))
                if not finite or huge:
                    active[n]=False;status[n]='numeric_magnitude_guard' if huge else 'undefined_BCT_inverse_or_nonfinite';continue
            state[n]=y;hist[n,j]=y
        if j%1000==0:print('STEP',j,'ode_runs',int(latched.sum()),'ode_calls',ode.calls,f'{time.perf_counter()-t_start:.0f}s',flush=True)
    wall=time.perf_counter()-t_start
    rows,summ=[],[]
    for n,(i,k) in enumerate(runs):
        tr=hist[n][np.isfinite(hist[n]).all(-1)]
        for h in (.01,.02):
            q,_=assess(tr,refs[i],pressures[i],dt,matrix,h,i,status[n],taus[i])
            rows.append(dict(variant=name,split=str(split[i]),horizon=h,seed=SEEDS[k],**q,
                gate_on_s=gate_on[n],gate_off_s=gate_off[n],ode_steps=int(on_ode[n,1:round(h/dt)+1].sum()),
                max_R_smoothed_pre=float(np.nanmax(pd.Series(Rhist[n]).rolling(WINDOW,min_periods=1).median()[hist[n,:,0]<T0[n]+DETECT_RISE])) if (hist[n,:,0]<T0[n]+DETECT_RISE).any() else np.nan))
    df=pd.DataFrame(rows);df.to_csv(out/'trajectories.csv',index=False)
    for (sp,h),g in df.groupby(['split','horizon']):
        sm=horizon_summary(g.rename(columns={'tau400_relative_error':'_unused','tau400_relative_error_refined':'tau400_relative_error'}).to_dict('records'))
        sm.update(ode_step_fraction=float(g.ode_steps.sum()/(len(g)*round(h/dt))),gated_runs=int(g.gate_on_s.notna().sum()))
        summ.append(dict(variant=name,split=sp,horizon=h,**sm));print('RESULT',name,sp,h,{k:sm[k] for k in ('completed','physical_completed','ignition_classification','median_ignition_error','temperature_profile_mean','ode_step_fraction','gated_runs')},flush=True)
    pd.DataFrame(summ).to_csv(out/'summary.csv',index=False)
    np.savez_compressed(out/'traces.npz',case=case,seed=np.array(SEEDS)[member],T=hist[...,0].astype('float32'),R=Rhist.astype('float32'),on_ode=on_ode)
    atomic_json(out/'manifest.json',dict(status='completed',variant=name,threshold=a.threshold,handover_K=a.handover,detect_rise_K=DETECT_RISE,
        window_steps=WINDOW,eps=EPS,models=str(ROOT),seeds=list(SEEDS),wall_seconds=wall,network_seconds=net_seconds,
        ode_calls=ode.calls,ode_seconds=ode.seconds,ode_seconds_per_step=ode.seconds/max(ode.calls,1),
        reference_sha256=sha(REF),evaluator_sha256=sha(__file__)))
    print('DONE',name,f'{wall:.0f}s','ode',ode.calls,f'{ode.seconds:.1f}s',flush=True)


if __name__=='__main__':main()
