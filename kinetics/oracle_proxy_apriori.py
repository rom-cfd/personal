"""A priori test of two truth-free proxies for the per-step committee oracle.

Candidates: the ten 1000-epoch members, each Wang-corrected (guard, Eqs. 5-9,
enthalpy T), exactly as in min_correction_apriori.py.

Proxy 1, implicit-step residual (no learning).  With f the adiabatic fixed-P
source term of the same 28-reaction mechanism (Cantera, normalized-Y convention):
    R(c)  = c - x - dt/2 (f(x) + f(c))          trapezoidal residual in [T, Y9]
    delta = (I - dt/2 J(x))^-1 R(c)             one simplified-Newton step, J by FD at x
    score = scaled rms(delta) with the stored train-only metric; pick argmin.
Also reported: the same delta for the Cantera truth (how far the true step is
from satisfying the trapezoidal rule) and 'polished' states c - delta with T
recomputed from enthalpy (one Newton step; not a selection rule).

Proxy 2, nearest-neighbour oracle lookup.  Library = validation pairs with their
per-member true errors.  For a query state, take its k nearest library states
in the model's own standardized input features and pick the member with the
lowest mean log error there.  k is chosen by leave-one-trajectory-out CV on
validation; test is scored once with the full validation library.

Persistence: how often the oracle winner at step t also wins at t+1 within a
trajectory, against the chance rate sum_m p_m^2.
"""
import argparse,json,os
from pathlib import Path
import numpy as np

SPECIES=['H2','H','O','O2','OH','H2O','HO2','H2O2','N2']
_GAS=None


def _rhs_chunk(s):
    """s: (n,11) = [T, Y9, P].  Returns (n,10) = [dT/dt, dY9/dt]."""
    global _GAS
    if _GAS is None:
        from .data import mechanism
        _GAS=mechanism()
    g=_GAS;W=g.molecular_weights;out=np.full((len(s),10),np.nan)
    for i,r in enumerate(s):
        if not np.isfinite(r).all() or r[0]<=0 or (r[1:10]<0).any():continue
        try:g.TPY=r[0],r[10],r[1:10]
        except Exception:continue
        w=g.net_production_rates;rho=g.density
        out[i,0]=-np.dot(g.partial_molar_enthalpies,w)/(rho*g.cp_mass);out[i,1:]=w*W/rho
    return out


def rhs(states,pool,chunk=4000):
    parts=pool.map(_rhs_chunk,[states[i:i+chunk] for i in range(0,len(states),chunk)])
    return np.concatenate(parts) if parts else np.empty((0,10))


def scaled(a,metric):
    return np.concatenate([a[...,:1]/metric['temperature_scale'],a[...,1:10]/np.asarray(metric['species_scale'])],axis=-1)


def srms(a,metric):return np.sqrt(np.mean(scaled(a,metric)**2,-1))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('results/zhang_5p3m_gelu_ens/g0.1'))
    ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55,66,77,88,99,110])
    ap.add_argument('--source',type=Path,default=Path('results/long10ms_data/trajectories.npz'))
    ap.add_argument('--single',type=int,default=99)
    ap.add_argument('--ks',type=int,nargs='+',default=[1,4,16,64,256])
    ap.add_argument('--workers',type=int,default=int(os.environ.get('SLURM_CPUS_PER_TASK',8)))
    ap.add_argument('--output',type=Path)
    a=ap.parse_args();out=a.output or a.root/'oracle_proxy_apriori'
    if out.exists():raise FileExistsError(out)
    import multiprocessing as mp
    pool=mp.get_context('spawn').Pool(a.workers)   # workers never import TensorFlow
    import pandas as pd
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    from .zhang_model import ZhangCommittee
    from .zhang_wang import WangHardCorrection
    from .data import samples
    from .multiscale_large_data import atomic_json
    metric=json.loads((a.root/'preprocessing/metric.json').read_text())['metric']
    p=WangHardCorrection(ZhangCommittee([str(a.root/f'seed{s}/model') for s in a.seeds],a.seeds,metric,'mean'))
    with np.load(a.source) as d:data={k:d[k] for k in d.files}
    dt=float(data['dt']);M=len(a.seeds);S=a.seeds.index(a.single);RAD=[2,5,7]
    out.mkdir(parents=True)
    rows,corr,knn_rows,ind_rows=[],[],[],[];extra={}
    D={}
    for split in ('val','test'):
        x,y,ids=samples(data,split);n=len(x);P=x[:,10]
        steps=data['states'].shape[1]-1;group=np.repeat(np.arange(len(ids)),steps)
        cor=np.empty((M,n,10))
        for i in range(0,n,2048):
            r=p.raw_candidates(x[i:i+2048]);cor[:,i:i+2048]=p.thermo.repair(r[...,:10],x[i:i+2048])
        ok=np.isfinite(cor).all(-1)
        err=np.sqrt(np.mean((scaled(cor,metric)-scaled(y,metric)[None])**2,-1));err[~ok]=np.inf
        # --- source terms and FD Jacobian at x
        fx=rhs(x,pool)
        base=x[:,:10];hcol=1.5e-8*np.concatenate([base[:,:1],np.maximum(np.abs(base[:,1:]),1e-4)],1)
        pert=np.repeat(x[None],10,0)
        for j in range(10):pert[j,:,j]+=hcol[:,j]
        fp=rhs(pert.reshape(-1,11),pool).reshape(10,n,10)
        J=np.transpose((fp-fx[None])/hcol.T[:,:,None],(1,2,0))       # J[n, i, j] = df_i/dz_j
        A=np.eye(10)[None]-0.5*dt*J
        fc=rhs(np.concatenate([cor,np.broadcast_to(P[None,:,None],(M,n,1))],-1).reshape(-1,11),pool).reshape(M,n,10)
        fy=rhs(np.concatenate([y,P[:,None]],1),pool)
        def newton(c,fcc):
            R=c-base-0.5*dt*(fx+fcc)
            good=np.isfinite(R).all(-1)&np.isfinite(A).all((-1,-2))
            dl=np.full_like(R,np.nan);dl[good]=np.linalg.solve(A[good],R[good][...,None])[...,0];return dl
        delta=np.stack([newton(cor[m],fc[m]) for m in range(M)])
        trap=srms(delta,metric);trap[~np.isfinite(trap)]=np.inf
        dtruth=srms(newton(y,fy),metric)
        # --- polished candidates: c - delta, T from enthalpy; rejected (kept unpolished) if any Y<0
        hs,_=p.thermo.species_h_cp(x[:,0]);h=np.sum(x[:,1:10]/x[:,1:10].sum(1)[:,None]*hs,1)
        def polish(m_idx):
            idx=np.arange(n);c=cor[m_idx,idx];d=delta[m_idx,idx];q=c-d
            bad=~np.isfinite(q).all(-1)|(q[:,1:]<0).any(-1);q[bad]=c[bad]
            q[~bad,0]=p.thermo.temperatures(q[~bad,1:10],x[~bad,0],h[~bad],P[~bad]);return q,float(bad.mean())
        # --- oracle and persistence
        orc=np.argmin(err,0);w=orc.reshape(len(ids),steps)
        share=np.bincount(orc,minlength=M)/n
        extra[f'{split}_persistence']=dict(same_as_next=float(np.mean(w[:,1:]==w[:,:-1])),chance=float(np.sum(share**2)),
            oracle_share={str(s):float(v) for s,v in zip(a.seeds,share)})
        # --- kNN oracle lookup
        F=p.predictor.base.features(x);F=np.asarray(F,dtype='float64')
        L=np.log(np.where(np.isfinite(err),err,np.nanmax(err[np.isfinite(err)])*10))
        D[split]=dict(F=F,L=L,group=group)
        picks={'single':np.full(n,S),'trap_min':np.argmin(trap,0),'oracle':orc}
        if split=='val':
            for k in a.ks:
                pk=np.empty(n,int)
                for gi in np.unique(group):
                    q=group==gi;tr=cKDTree(F[~q]);_,nb=tr.query(F[q],k=k);nb=nb.reshape(q.sum(),k)
                    pk[q]=np.argmin(L[~q][:,None] if False else L[:,~q][:,nb].mean(-1),0)
                e=err[pk,np.arange(n)]
                knn_rows.append(dict(split='val(LOTO-CV)',k=k,scaled_error_mean=float(e.mean()),oracle_hit_rate=float(np.mean(pk==orc))))
                picks[f'knn_k{k}']=pk
            kbest=min(knn_rows,key=lambda r:r['scaled_error_mean'])['k'];extra['knn_k_selected_on_val']=int(kbest)
            picks['knn']=picks[f'knn_k{kbest}']
            for k in a.ks:del picks[f'knn_k{k}']
        else:
            V=D['val'];tr=cKDTree(V['F'])
            for k in a.ks:
                _,nb=tr.query(F,k=k);nb=nb.reshape(n,k);pk=np.argmin(V['L'][:,nb].mean(-1),0)
                e=err[pk,np.arange(n)]
                knn_rows.append(dict(split='test(val library)',k=k,scaled_error_mean=float(e.mean()),oracle_hit_rate=float(np.mean(pk==orc))))
                if k==extra['knn_k_selected_on_val']:picks['knn']=pk
        idx=np.arange(n)
        state={k:cor[v,idx] for k,v in picks.items()}
        mean=np.where(ok[...,None],cor,0).sum(0)/ok.sum(0)[:,None]
        mean[:,0]=p.thermo.temperatures(mean[:,1:10],x[:,0],h,P);state['mean10']=mean
        state['single+polish'],extra[f'{split}_polish_rejected_single']=polish(picks['single'])
        state['trap_min+polish'],extra[f'{split}_polish_rejected_trap']=polish(picks['trap_min'])
        for k,s in state.items():
            e=np.sqrt(np.mean((scaled(s,metric)-scaled(y,metric))**2,-1))
            rows.append(dict(split=split,rule=k,scaled_error_mean=float(np.mean(e)),scaled_error_p95=float(np.quantile(e,.95)),
                temperature_rmse=float(np.sqrt(np.mean((s[:,0]-y[:,0])**2))),species_rmse=float(np.sqrt(np.mean((s[:,1:]-y[:,1:])**2))),
                oracle_hit_rate=float(np.mean(picks[k]==orc)) if k in picks else np.nan))
        # --- does the trap score rank members by true error?
        good=np.isfinite(trap).all(0)&np.isfinite(err).all(0);cc=trap[:,good];ee=err[:,good]
        within=[spearmanr(cc[:,j],ee[:,j]).correlation for j in range(0,cc.shape[1],7)]
        corr.append(dict(split=split,score='trap_newton',pooled_spearman=float(spearmanr(cc.ravel(),ee.ravel()).correlation),
            within_state_spearman_mean=float(np.nanmean(within)),oracle_hit_rate=float(np.mean(np.argmin(cc,0)==np.argmin(ee,0))),chance=1/M,
            pick_error_rank_mean=float(np.mean(np.argsort(np.argsort(ee,0),0)[np.argmin(cc,0),np.arange(cc.shape[1])]+1))))
        extra[f'{split}_truth_residual']=dict(truth_delta_median=float(np.median(dtruth)),truth_delta_p95=float(np.quantile(dtruth,.95)),
            member_error_median=float(np.median(err[np.isfinite(err)])),oracle_error_median=float(np.median(err[orc,idx])),
            member_delta_median=float(np.median(trap[np.isfinite(trap)])))
        # --- induction: radical-pool growth of the chosen state
        px=x[:,RAD].sum(1);py=y[:,RAD].sum(1)
        ind=(px>1e-14)&(y[:,0]-x[:,0]<0.5)&(x[:,0]<1300)&(py>px*(1+1e-6))
        gt=np.log(py[ind]/px[ind]);gm=np.log(np.maximum(cor[:,ind][...,RAD].sum(-1),1e-300)/px[ind]);rank=np.argsort(np.argsort(-gm,0),0)
        for k,s in state.items():
            g=np.log(np.maximum(s[ind][:,RAD].sum(1),1e-300)/px[ind])
            ind_rows.append(dict(split=split,rule=k,pairs=int(ind.sum()),under_predict_share=float(np.mean(g<gt)),
                growth_rank_mean=float(np.mean(rank[picks[k][ind],np.arange(ind.sum())])+1) if k in picks else np.nan,
                scaled_error_mean=float(np.sqrt(np.mean((scaled(s[ind],metric)-scaled(y[ind],metric))**2,-1)).mean())))
        print('DONE',split,flush=True)
    pool.close()
    pd.DataFrame(rows).to_csv(out/'rules.csv',index=False);pd.DataFrame(corr).to_csv(out/'correlation.csv',index=False)
    pd.DataFrame(knn_rows).to_csv(out/'knn.csv',index=False);pd.DataFrame(ind_rows).to_csv(out/'induction.csv',index=False)
    atomic_json(out/'diagnostics.json',extra)
    atomic_json(out/'manifest.json',dict(status='completed',root=str(a.root),seeds=a.seeds,single=a.single,dt=dt,
        metric='stored train-only medoid metric',correction='Wang 2025 Eqs. 5-9 + enthalpy T, guard on',
        trap='trapezoidal residual, one simplified-Newton step with FD Jacobian at x',
        knn='library = val; k chosen by leave-one-trajectory-out on val; test scored once',
        note='rules fixed before evaluation; test reported for reference'))


if __name__=='__main__':main()
