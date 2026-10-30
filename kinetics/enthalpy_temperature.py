"""Temperature-only thermodynamic closure; ANN compositions are left unchanged.

Mixture properties use Cantera's normalized mass-fraction convention. This does
not normalize the ANN's stored/self-fed Y, or impose elemental conservation.
"""
import json
from pathlib import Path
import numpy as np
from .vector_physics import VectorThermo


class EnthalpyTemperature:
    def __init__(self,predictor):
        self.predictor=predictor;self.thermo=VectorThermo()

    @property
    def last_diagnostics(self):return self.predictor.last_diagnostics

    def close(self,proposals,x):
        proposals=np.asarray(proposals,dtype=float);x=np.asarray(x,dtype=float)
        out=proposals.copy()
        valid=(np.isfinite(x).all(axis=1)&np.isfinite(out[:,1:]).all(axis=1)&
            (x[:,1:10]>=0).all(axis=1)&(out[:,1:10]>=0).all(axis=1)&
            (x[:,1:10].sum(axis=1)>0)&(out[:,1:10].sum(axis=1)>0)&(out[:,-1]>0))
        out[:,0]=np.nan;live=np.flatnonzero(valid)
        if len(live):
            hs,_=self.thermo.species_h_cp(x[live,0])
            yn=x[live,1:10]/x[live,1:10].sum(axis=1)[:,None]
            target=np.sum(yn*hs,axis=1)
            out[live,0]=self.thermo.temperatures(proposals[live,1:10],x[live,0],target,proposals[live,-1])
        return out

    def candidates(self,x):
        proposals=self.predictor.candidates(x)
        return np.stack([self.close(p,x) for p in proposals])

    def predict_full(self,x):return self.close(self.predictor.predict_full(x),x)

    def predict(self,x):return self.predict_full(x)[:,:10]

    def save(self,path):
        self.predictor.save(path);path=Path(path);info=json.loads(path.read_text())
        info['temperature_mode']='enthalpy';path.write_text(json.dumps(info,indent=2))


def load_predictor(path):
    from .zhang_model import ZhangCommittee
    info=json.loads(Path(path).read_text());mode=info.pop('temperature_mode','raw')
    p=ZhangCommittee(**info)
    return EnthalpyTemperature(p) if mode=='enthalpy' else p
