"""Explicit optional positivity/element/enthalpy repair, never a kinetics solver.

Weighted Euclidean projection onto A Y = A Y_current, Y >= 0. Active-set
constraints protect trace species by using squared abundance as correction scale.
This changes the integrator and must be compared separately from augmentation.
"""
import numpy as np
from scipy.optimize import minimize
from .data import mechanism


def element_matrix(gas):
    return np.array([[gas.n_atoms(k,e)*gas.atomic_weight(e)/gas.molecular_weights[k]
                     for k in range(gas.n_species)] for e in ('H','O','N')])


def project_species(proposal,current,matrix):
    v=np.asarray(proposal,dtype=float);current=np.asarray(current,dtype=float)
    if not np.all(np.isfinite(v)):raise ValueError('nonfinite composition')
    b=matrix@current
    scale=np.maximum(np.maximum(np.abs(v),current),1e-8)
    scale=np.minimum(scale,1.)
    variance=scale**2
    active=np.zeros(len(v),bool)
    # Drop negative free coordinates, re-solve, and check the KKT conditions.
    for _ in range(20):
        free=~active
        a=matrix[:,free];d=variance[free]
        lam=np.linalg.lstsq((a*d)@a.T,a@v[free]-b,rcond=1e-14)[0]
        y=np.zeros_like(v);y[free]=v[free]-d*(a.T@lam)
        negatives=free&(y < -1e-14)
        if negatives.any():
            active[np.flatnonzero(negatives)[np.argmin(y[negatives]/scale[negatives])]]=True
            continue
        # Objective derivative at constrained zero minus affine multiplier.
        dual=-v/variance+matrix.T@lam
        release=active&(dual < -1e-7)
        if release.any():
            active[np.flatnonzero(release)[np.argmin(dual[release])]]=False
            continue
        y=np.maximum(y,0)
        if np.max(np.abs(matrix@y-b))<1e-10:return y
        break
    # Solve in scaled displacement coordinates to avoid ill-conditioned weights.
    c=matrix*scale;row_scale=np.maximum(np.linalg.norm(c,axis=1),1e-20)
    c=c/row_scale[:,None];rhs=(b-matrix@v)/row_scale
    fit=minimize(lambda u:.5*np.sum(u*u),(current-v)/scale,
                 jac=lambda u:u,bounds=[(-float(q/s),None) for q,s in zip(v,scale)],
                 constraints={'type':'eq','fun':lambda u:c@u-rhs,'jac':lambda u:c},
                 method='SLSQP',options={'ftol':1e-13,'maxiter':100})
    result=v+scale*fit.x
    if not fit.success or np.max(np.abs(matrix@result-b))>1e-9 or result.min() < -1e-12:
        raise ValueError('composition projection failed: '+fit.message)
    return np.maximum(result,0)


class PhysicalPredictor:
    """Common wrapper: element projection and adiabatic enthalpy recovery.

    The ANN predicts next composition; temperature is recovered from current
    mixture enthalpy with Cantera HPY. No reaction integration or net production
    rates are used during prediction. Therefore energy conservation is imposed,
    and must not be reported as independently learned by the ANN.
    """
    def __init__(self,predictor):
        self.predictor=predictor;self.gas=mechanism();self.matrix=element_matrix(self.gas)
        self.projection_changes=[]

    def predict(self,x):
        if hasattr(self.predictor,'predict_guarded'):
            raw=self.predictor.predict_guarded(x)
        elif hasattr(self.predictor,'transform'):
            # Explicit physical decoder guard: a mass fraction lies in [0,1].
            # Applied identically to all transformations, only in repaired mode.
            net=self.predictor
            delta=net.model(net.features(x).astype('float32'),training=False).numpy().astype('float64')
            z=net.norm.inverse(net.norm.forward(net.encode_state(x[:,:10]))+delta)
            upper=net.transform.forward(np.ones((1,9)))
            y=net.transform.inverse(np.clip(z[:,1:],0,upper))
            raw=np.c_[x[:,0],y]
        else:
            raw=self.predictor.predict(x)
        out=np.full_like(raw,np.nan)
        for k,(a,p) in enumerate(zip(x,raw)):
            try:
                if not np.all(np.isfinite(p)):continue
                y=project_species(p[1:],a[1:10],self.matrix)
                self.gas.TPY=a[0],a[-1],a[1:10]
                h=self.gas.enthalpy_mass
                self.gas.HPY=h,a[-1],y
                out[k]=np.r_[self.gas.T,y]
            except (ValueError,RuntimeError):
                continue
        return out
