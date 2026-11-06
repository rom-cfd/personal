"""Committee disagreement along the a-posteriori rollouts of strategy 5 (UR10 physics-in-loop + physics).

Read-only on all existing results. For every test/val case and every seed's saved 20-ms rollout
(results/t1000_ur10_phys_best/g0.1/evaluation_wang_members/seed*/traces), and for the Cantera reference
trajectory, the five best-validation members are queried on the SAME state and their Wang-corrected
next states are compared:

  dT_k      = T_k(next) - T(state)               (enthalpy-recovered temperature, K/step)
  D_k       = [encode(next_k) - encode(state)] / output_scale   (T + 9 BCT(Y) channels; P excluded)
  sigma_T   = std_k dT_k                          (K/step)
  R         = ||std_k D_k|| / (||mean_k D_k|| + eps)            relative one-step disagreement

plus the trajectory-level spread of the five independent rollouts, std_k T_k(t), against their error.
Writes results/t1000_disagreement/.
"""
import json,time
from pathlib import Path
import numpy as np
from .multiscale_large_data import atomic_json
from .zhang_model import ZhangCommittee
from .zhang_wang import WangHardCorrection

ROOT=Path('results/t1000_ur10_phys_best/g0.1');EVAL=ROOT/'evaluation_wang_members'
SEEDS=(11,22,33,44,55);OUT=Path('results/t1000_disagreement')
SRC=Path('results/t1000_long10ms_data/trajectories.npz');REF=Path('results/t1000_reference/reference20ms.npz')
EPS=1e-3   # floor on ||mean D|| (units of training increment std), keeps R finite at equilibrium
IDX=np.r_[np.arange(0,1000,2),np.arange(1000,10000,10)]   # 0-2 ms every 4 us, 2-20 ms every 20 us


def encode(s,gamma):
    with np.errstate(invalid='ignore',divide='ignore'):
        return np.c_[s[...,0],(np.maximum(s[...,1:10],0)**gamma-1)/gamma] if s.ndim==2 else \
            np.concatenate([s[...,:1],(np.maximum(s[...,1:10],0)**gamma-1)/gamma],axis=-1)


def query(w,x,scale,gamma,batch=8192):
    sT,mT,R,sD,valid=[],[],[],[],[]
    for i in range(0,len(x),batch):
        xb=x[i:i+batch];c=w.candidates(xb)                      # (5,n,11) corrected next states
        ok=np.isfinite(c).all(-1);valid.append(ok.sum(0))
        dT=np.where(ok,c[...,0]-xb[None,:,0],np.nan)
        D=(encode(c[...,:10],gamma)-encode(xb[:,:10],gamma)[None])/scale[None,None]
        D[~ok]=np.nan
        sd=np.sqrt(np.nansum(np.nanvar(D,axis=0),axis=-1));md=np.linalg.norm(np.nanmean(D,axis=0),axis=-1)
        sT.append(np.nanstd(dT,axis=0));mT.append(np.nanmean(dT,axis=0));sD.append(sd);R.append(sd/(md+EPS))
    return {k:np.concatenate(v) for k,v in dict(sigma_dT=sT,mean_dT=mT,sigma_D=sD,R=R,valid=valid).items()}


def main():
    OUT.mkdir(parents=True,exist_ok=True);t0=time.perf_counter()
    metric=json.loads((ROOT/'preprocessing/metric.json').read_text())['metric']
    com=ZhangCommittee([str(ROOT/f'seed{s}/model') for s in SEEDS],list(SEEDS),metric,'mean');w=WangHardCorrection(com)
    base=com.base;gamma=base.gamma
    scale=np.r_[base.output_norm.scale[0],base.output_norm.scale[2:]]   # network order T,P,BCT -> drop P
    with np.load(SRC) as d:split=d['split'];pressures=d['pressures'];initials=d['initials']
    with np.load(REF) as d:refs={int(i):s for i,s in zip(d['ids'],d['states'])};taus={int(i):t for i,t in zip(d['ids'],d['refined_tau400'])}
    store={};cases=[]
    for sp in ('test','val'):
        traces={s:np.load(EVAL/f'seed{s}/traces/seed{s}_{sp}.npz') for s in SEEDS}
        for cid in sorted(int(k.split('_')[1]) for k in traces[SEEDS[0]].files):
            ref=refs[cid];P=float(pressures[cid]);rollouts=[traces[s][f'trajectory_{cid}'] for s in SEEDS]
            assert all(len(r)==10001 for r in rollouts),cid
            T=np.stack([r[:,0] for r in rollouts]);Tref=ref[:,0]
            # a-priori: members on the reference states; a-posteriori: members on each seed's own rollout states
            xr=np.c_[ref[IDX],np.full(len(IDX),P)];q_ref=query(w,xr,scale,gamma)
            q_roll=[query(w,r[IDX],scale,gamma) for r in rollouts]
            store[cid]=dict(T=T,Tref=Tref,q_ref=q_ref,q_roll=q_roll)
            cases.append(dict(id=cid,split=sp,T0=float(ref[0,0]),P_atm=P/101325.,tau_ref=float(taus[cid])))
            print('CASE',sp,cid,f'{time.perf_counter()-t0:.0f}s',flush=True)
    np.savez_compressed(OUT/'disagreement.npz',idx=IDX,
        **{f'{k}_{cid}':v for cid,s in store.items() for k,v in
           [('T',s['T']),('Tref',s['Tref'])]+[(f'ref_{m}',s['q_ref'][m]) for m in s['q_ref']]+
           [(f'roll_{m}',np.stack([q[m] for q in s['q_roll']])) for m in s['q_roll'][0]]})
    atomic_json(OUT/'cases.json',dict(cases=cases,seeds=list(SEEDS),root=str(ROOT),eps=EPS,
        idx='0-2 ms every 2 steps, 2-20 ms every 10 steps',seconds=time.perf_counter()-t0))
    print('DONE',time.perf_counter()-t0,flush=True)


if __name__=='__main__':main()
