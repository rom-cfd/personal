"""Committee selection by implicit-step residual (truth-free per-step oracle proxy).

Each member's Wang-corrected candidate c is scored by one simplified-Newton step of
the trapezoidal rule for the adiabatic fixed-P 28-reaction source term f:
    R(c) = c - x - dt/2 (f(x) + f(c)),   delta = (I - dt/2 J(x))^-1 R(c)
with J(x) by forward differences. The candidate with the smallest train-metric
scaled rms(delta) is returned (selection only, by default).

Optional induction correction (induction_correction=True): when the current state
is pre-ignition by a truth-free gate, T(x) < t_max and dt*dT/dt(x) < dT_max, the
selected candidate is replaced by up to `iterations` simplified-Newton iterations
of the same trapezoidal system (J(x) fixed), Y kept as is otherwise, T recomputed
from the current enthalpy. Corrections that produce any negative/non-finite Y are
rejected (candidate kept). Element totals are preserved because f conserves atoms.
Optional period > 1: select by residual only every `period` steps per trajectory and
hold that member in between (stateful; call reset() before each rollout).
A priori validation: kinetics/oracle_proxy_apriori.py.
"""
import json
from pathlib import Path
import numpy as np
from .oracle_proxy_apriori import _rhs_chunk,srms
from . import oracle_proxy_apriori as _opa


def _rhs_lambda_chunk(s):
    """[f (n,10), lambda (n,9)] where lambda_k = destruction rate / concentration (1/s) at s."""
    if _opa._GAS is None:_rhs_chunk(s[:1])
    g=_opa._GAS;f=_rhs_chunk(s);lam=np.zeros((len(s),9))
    for i,r in enumerate(s):
        if not np.isfinite(f[i]).all():lam[i]=np.nan;continue
        g.TPY=r[0],r[10],r[1:10];c=g.concentrations;d=g.destruction_rates
        lam[i]=np.where(c>0,d/np.maximum(c,1e-300),0.)
    return f,lam


def jacobian(x,fx):
    base=x[:,:10];h=1.5e-8*np.concatenate([base[:,:1],np.maximum(np.abs(base[:,1:]),1e-4)],1)
    pert=np.repeat(x[None],10,0)
    for j in range(10):pert[j,:,j]+=h[:,j]
    fp=_rhs_chunk(pert.reshape(-1,11)).reshape(10,len(x),10)
    return np.transpose((fp-fx[None])/h.T[:,:,None],(1,2,0))


class ResidualSelect:
    def __init__(self,wang,dt,induction_correction=False,t_max=1300.,dT_max=.5,iterations=3,tol=1e-12,period=1,scaling='jacobian'):
        self.wang=wang;self.predictor=wang.predictor;self.dt=float(dt);self.metric=self.predictor.metric
        self.induction_correction=induction_correction;self.t_max=t_max;self.dT_max=dT_max;self.iterations=iterations;self.tol=tol
        self.period=int(period);self.last_diagnostics={};self.reset()
        if scaling not in ('jacobian','diag','raw'):raise ValueError(scaling)
        if induction_correction and scaling!='jacobian':raise ValueError('induction correction needs the Jacobian')
        self.scaling=scaling

    def reset(self):
        """Forget the held per-row choice (call before each rollout batch when period > 1)."""
        self._step=0;self._choice=None

    def residual_scores(self,x,c):
        if self.scaling!='jacobian':return self.cheap_scores(x,c)
        M,n=c.shape[:2];fx=_rhs_chunk(x);A=np.eye(10)[None]-0.5*self.dt*jacobian(x,fx);self._fx,self._A=fx,A
        fc=_rhs_chunk(np.concatenate([c[...,:10],np.broadcast_to(x[None,:,10:],(M,n,1))],-1).reshape(-1,11)).reshape(M,n,10)
        R=c[...,:10]-x[None,:,:10]-0.5*self.dt*(fx[None]+fc)
        good=np.isfinite(R).all(-1)&np.isfinite(A).all((-1,-2))[None]
        d=np.full_like(R,np.nan)
        for m in range(M):
            g=good[m]
            if g.any():d[m,g]=np.linalg.solve(A[g],R[m,g][...,None])[...,0]
        s=srms(d,self.metric);s[~np.isfinite(s)]=np.inf;return s

    def cheap_scores(self,x,c):
        """Jacobian-free: raw trapezoidal residual, optionally divided per species by
        (1 + dt/2 lambda_k(x)), lambda_k = destruction rate / concentration (diagonal stiffness)."""
        M,n=c.shape[:2]
        if self.scaling=='diag':fx,lam=_rhs_lambda_chunk(x)
        else:fx=_rhs_chunk(x)
        fc=_rhs_chunk(np.concatenate([c[...,:10],np.broadcast_to(x[None,:,10:],(M,n,1))],-1).reshape(-1,11)).reshape(M,n,10)
        R=c[...,:10]-x[None,:,:10]-0.5*self.dt*(fx[None]+fc)
        if self.scaling=='diag':R=np.concatenate([R[...,:1],R[...,1:]/(1+0.5*self.dt*lam[None])],-1)
        s=srms(R,self.metric);s[~np.isfinite(s)]=np.inf;return s

    def predict_full(self,x):
        c=self.wang.candidates(x);finite=np.isfinite(c).all(-1);n=len(x);idx=np.arange(n)
        if self.period>1:
            # Stateful: rows are the same trajectories call after call. Select by residual every
            # `period` calls (or if the batch changes / the held member's candidate is invalid),
            # otherwise keep the held member without computing any residual.
            if self._choice is None or len(self._choice)!=n:self._step=0
            redo=np.ones(n,bool) if self._step%self.period==0 else ~finite[self._choice,idx]
            choice=np.zeros(n,int) if self._step%self.period==0 else self._choice.copy()
            s=np.zeros(n)
            if redo.any():
                xs,cs=x[redo],c[:,redo];sc=self.residual_scores(xs,cs);sc[~finite[:,redo]]=np.inf
                choice[redo]=np.argmin(sc,0);s[redo]=sc[choice[redo],np.arange(redo.sum())]
            live=finite[choice,idx];self._choice=choice;self._step+=1
            out=np.full(c.shape[1:],np.nan);out[live]=c[choice[live],np.flatnonzero(live)]
            seed=np.full(n,-1);seed[live]=self.predictor.seeds[choice[live]]
            self.last_diagnostics=dict(selected_seed=seed,valid_count=finite.sum(0),score=s,corrected=np.zeros(n,bool),reselected=redo)
            return out
        s=self.residual_scores(x,c);s[~finite]=np.inf
        choice=np.argmin(s,0);live=np.isfinite(s[choice,np.arange(len(x))])
        out=np.full(c.shape[1:],np.nan);out[live]=c[choice[live],np.flatnonzero(live)]
        seed=np.full(len(x),-1);seed[live]=self.predictor.seeds[choice[live]]
        corrected=np.zeros(len(x),bool)
        if self.induction_correction:
            gate=live&(x[:,0]<self.t_max)&(self.dt*self._fx[:,0]<self.dT_max)
            if gate.any():
                out[gate,:10],corrected[gate]=self.newton(x[gate],out[gate,:10],self._fx[gate],self._A[gate])
        self.last_diagnostics=dict(selected_seed=seed,valid_count=finite.sum(0),score=s[choice,np.arange(len(x))],corrected=corrected)
        return out

    def newton(self,x,c,fx,A):
        q=c.copy()
        for _ in range(self.iterations):
            fq=_rhs_chunk(np.c_[q,x[:,10:]]);R=q-x[:,:10]-0.5*self.dt*(fx+fq)
            ok=np.isfinite(R).all(-1);d=np.zeros_like(R)
            if ok.any():d[ok]=np.linalg.solve(A[ok],R[ok][...,None])[...,0]
            q=q-d
            if np.all(srms(d,self.metric)<self.tol):break
        bad=~np.isfinite(q).all(-1)|(q[:,1:]<0).any(-1);q[bad]=c[bad]
        if (~bad).any():
            th=self.wang.thermo;hs,_=th.species_h_cp(x[~bad,0]);yn=x[~bad,1:10]/x[~bad,1:10].sum(1)[:,None]
            q[~bad,0]=th.temperatures(q[~bad,1:10],x[~bad,0],np.sum(yn*hs,1),x[~bad,10])
        return q,~bad

    def predict(self,x):return self.predict_full(x)[:,:10]

    def save(self,path):
        self.predictor.save(path);path=Path(path);info=json.loads(path.read_text())
        info.update(temperature_mode='wang',selection='implicit_trapezoidal_residual',dt=self.dt,
            period=self.period,scaling=self.scaling,induction_correction=self.induction_correction,t_max=self.t_max,dT_max=self.dT_max,iterations=self.iterations)
        path.write_text(json.dumps(info,indent=2))
