"""Wang 2025 ANN-hard correction applied to frozen Zhang full-state models.

Per candidate: optional decoder guard (clip next BCT(Y) to the image of [0,1]),
Wang Eqs. (5)-(9) element projection in Y space, then adiabatic fixed-P
temperature recovery from the current enthalpy. The ANN temperature output is
discarded; the ANN pressure output is kept. Negative or inconsistent corrected
candidates are rejected (NaN), never clipped or renormalized afterwards.
This is the same correction protocol as WANG_HARD_REPORT.md.
"""
import json
from pathlib import Path
import numpy as np
from .wang_physics import WangThermo


class WangHardCorrection:
    def __init__(self,predictor,guard=True):
        self.predictor=predictor;self.guard=guard
        self.thermo=WangThermo(gamma=predictor.base.gamma)
        if predictor.rule not in ('single','mean','medoid'):raise ValueError('unknown committee rule')

    @property
    def last_diagnostics(self):return self.predictor.last_diagnostics

    def raw_candidates(self,x):
        p=self.predictor;base=p.base
        delta=p.forward(base.features(x).astype('float32')).numpy().astype('float64')
        z=base.encode(x)+base.output_norm.inverse(delta)
        if self.guard:z[...,2:]=np.clip(z[...,2:],-1/base.gamma,0.)
        with np.errstate(invalid='ignore',over='ignore',divide='ignore'):
            b=1+base.gamma*z[...,2:];y=np.where(b>=0,b**(1/base.gamma),np.nan)
        return np.concatenate([z[...,:1],y,z[...,1:2]*101325.],axis=-1)

    def candidates(self,x):
        raw=self.raw_candidates(x)
        return np.concatenate([self.thermo.repair(raw[...,:10],x),raw[...,10:]],axis=-1)

    def predict_full(self,x):
        c=self.candidates(x);finite=np.isfinite(c).all(axis=-1);count=finite.sum(axis=0)
        if self.predictor.rule=='single':out=c[0];choice=np.where(finite[0],0,-1)
        elif self.predictor.rule=='medoid':
            # Medoid of the corrected candidates on the train-scaled T/Y distance; carry its P.
            from .committee import choose_medoid
            t=c[:,:,:10].copy();t[~finite]=np.nan
            _,choice,_=choose_medoid(t,self.predictor.metric,'scaled')
            out=np.full(c.shape[1:],np.nan);live=np.flatnonzero(choice>=0);out[live]=c[choice[live],live]
        else:
            # Mean of corrected candidates is still element-consistent (constraints are linear).
            out=np.where(finite[...,None],c,0).sum(axis=0)/np.maximum(count,1)[:,None]
            out[count==0]=np.nan;choice=np.full(len(x),-1)
            live=count>0
            if live.any():out[live,0]=self.thermo.temperatures(out[live,1:10],x[live,0],self._enthalpy(x[live]),out[live,-1])
        seed=np.full(len(x),-1);live=np.flatnonzero(choice>=0);seed[live]=self.predictor.seeds[choice[live]]
        self.predictor.last_diagnostics=dict(selected_seed=seed,valid_count=count)
        return out

    def _enthalpy(self,x):
        hs,_=self.thermo.species_h_cp(x[:,0]);yn=x[:,1:10]/x[:,1:10].sum(axis=1)[:,None]
        return np.sum(yn*hs,axis=1)

    def predict(self,x):return self.predict_full(x)[:,:10]

    def save(self,path):
        self.predictor.save(path);path=Path(path);info=json.loads(path.read_text())
        info['temperature_mode']='wang' if self.guard else 'wang_noguard';path.write_text(json.dumps(info,indent=2))
