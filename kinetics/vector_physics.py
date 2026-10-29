"""Batched versions of the same element projection and ideal-gas HPY closure.

NASA-7 thermodynamics come from the loaded Cantera mechanism, not fitted data.
Ill-conditioned/ambiguous QPs and nonconverged thermal roots use the original
scalar projection/Cantera thermodynamics. No chemical ODE is used at inference.
"""
import numpy as np
from .physics import project_species,element_matrix
from .data import mechanism


def project_batch(proposals,current,matrix):
    v=np.asarray(proposals,dtype=float);current=np.asarray(current,dtype=float)
    n=len(v);scale=np.minimum(np.maximum(np.maximum(np.abs(v),current),1e-8),1.)
    variance=scale**2;b=current@matrix.T
    active=np.zeros_like(v,dtype=bool);pending=np.arange(n);out=np.full_like(v,np.nan)
    for _ in range(20):
        if not len(pending):break
        ii=pending;free=~active[ii];weighted=variance[ii]*free
        gram=np.einsum('ei,ni,fi->nef',matrix,weighted,matrix)
        rhs=(v[ii]*free)@matrix.T-b[ii]
        lam=np.einsum('nij,nj->ni',np.linalg.pinv(gram,rcond=1e-14),rhs)
        yy=np.where(free,v[ii]-variance[ii]*(lam@matrix),0.)
        negative=free&(yy < -1e-14)
        has_negative=negative.any(axis=1)
        if has_negative.any():
            jj=np.argmin(np.where(negative,yy/scale[ii],np.inf),axis=1)
            active[ii[has_negative],jj[has_negative]]=True
        dual=-v[ii]/variance[ii]+lam@matrix
        release=active[ii]&(dual < -1e-7)
        has_release=(~has_negative)&release.any(axis=1)
        if has_release.any():
            jj=np.argmin(np.where(release,dual,np.inf),axis=1)
            active[ii[has_release],jj[has_release]]=False
        done=~has_negative&~has_release
        yy=np.maximum(yy,0.)
        residual=np.max(np.abs(yy@matrix.T-b[ii]),axis=1)
        accepted=done&(residual<1e-10)
        out[ii[accepted]]=yy[accepted]
        retry=has_negative|has_release
        pending=ii[retry]
    fallback=np.flatnonzero(~np.all(np.isfinite(out),axis=1))
    for k in fallback:
        try:out[k]=project_species(v[k],current[k],matrix)
        except ValueError:pass
    return out,len(fallback)


class VectorThermo:
    def __init__(self):
        self.gas=mechanism();self.matrix=element_matrix(self.gas)
        if self.gas.thermo_model!='ideal-gas':raise ValueError('ideal gas required')
        coefficients=[]
        for species in self.gas.species():
            if species.thermo.__class__.__name__!='NasaPoly2':raise ValueError('NASA-7 polynomial required')
            coefficients.append(species.thermo.coeffs)
        coefficients=np.asarray(coefficients)
        self.mid=coefficients[:,0];self.high=coefficients[:,1:8];self.low=coefficients[:,8:15]
        import cantera as ct
        self.r_over_w=ct.gas_constant/self.gas.molecular_weights
        self.qp_fallbacks=0;self.thermal_fallbacks=0
        self.hmin=self.species_h_cp(np.array([200.]))[0][0]
        self.hmax=self.species_h_cp(np.array([6000.]))[0][0]

    def species_h_cp(self,temperatures):
        t=np.asarray(temperatures)[:,None]
        a=np.where((t<=self.mid)[...,None],self.low[None,:,:],self.high[None,:,:])
        cp=((((a[:,:,4]*t+a[:,:,3])*t+a[:,:,2])*t+a[:,:,1])*t+a[:,:,0])*self.r_over_w
        hrt=a[:,:,0]+t*(a[:,:,1]/2+t*(a[:,:,2]/3+t*(a[:,:,3]/4+t*a[:,:,4]/5)))+a[:,:,5]/t
        return hrt*t*self.r_over_w,cp

    def temperatures(self,y,initial_t,target_h,pressure):
        y=y/y.sum(axis=1)[:,None]
        lo=np.full(len(y),200.);hi=np.full(len(y),6000.)
        feasible=(y@self.hmin<=target_h+1e-7)&(y@self.hmax>=target_h-1e-7)
        t=np.clip(np.asarray(initial_t,dtype=float),lo,hi)
        done=~feasible
        for _ in range(30):
            live=np.flatnonzero(~done)
            if not len(live):break
            hs,cp=self.species_h_cp(t[live]);err=np.sum(y[live]*hs,axis=1)-target_h[live]
            converged=np.abs(err)<=1e-5;done[live[converged]]=True
            remaining=live[~converged]
            if not len(remaining):continue
            e=err[~converged];cap=np.sum(y[remaining]*cp[~converged],axis=1)
            hot=e>0;hi[remaining[hot]]=t[remaining[hot]];lo[remaining[~hot]]=t[remaining[~hot]]
            proposed=t[remaining]-e/cap
            valid=(proposed>lo[remaining])&(proposed<hi[remaining])&np.isfinite(proposed)
            t[remaining]=np.where(valid,proposed,.5*(lo[remaining]+hi[remaining]))
        t[~feasible]=np.nan
        fallback=np.flatnonzero(feasible&~done)
        self.thermal_fallbacks+=len(fallback)
        for k in fallback:
            try:
                self.gas.TP=initial_t[k],pressure[k];self.gas.HPY=target_h[k],pressure[k],y[k]
                t[k]=self.gas.T if 200<=self.gas.T<=6000 else np.nan
            except (ValueError,RuntimeError):t[k]=np.nan
        return t

    def repair(self,proposals,x):
        shape=proposals.shape;members,batch,_=shape
        # Member-major order matches Committee.proposals and returned candidates.
        raw=proposals.reshape(-1,10);current=np.tile(x[:,:10],(members,1));p=np.tile(x[:,-1],members)
        out=np.full_like(raw,np.nan);valid=np.all(np.isfinite(raw),axis=1)
        live=np.flatnonzero(valid)
        if not len(live):return out.reshape(shape)
        y,falls=project_batch(raw[live,1:],current[live,1:],self.matrix);self.qp_fallbacks+=falls
        good=np.all(np.isfinite(y),axis=1)
        live=live[good];y=y[good]
        if not len(live):return out.reshape(shape)
        hs,_=self.species_h_cp(x[:,0]);normalized=x[:,1:10]/x[:,1:10].sum(axis=1)[:,None]
        h0=np.sum(normalized*hs,axis=1)
        target=np.tile(h0,members)[live]
        t=self.temperatures(y,current[live,0],target,p[live])
        out[live]=np.c_[t,y]
        return out.reshape(shape)
