"""Wall time of the pure Cantera path on the 36 T>=1000 K cases, 20 ms in 2-us outputs (same settings as the reference)."""
import json,time
import cantera as ct,numpy as np
from .data import mechanism
from .multiscale_large_data import atomic_json
with np.load('results/t1000_reference/reference20ms.npz') as d:ids=d['ids'];P=d['pressures'];S=d['states'][:,0];dt=float(d['dt'])
rows=[]
for i,p,s in zip(ids,P,S):
    t0=time.perf_counter();g=mechanism();g.TPY=s[0],p,s[1:];r=ct.IdealGasConstPressureReactor(g,clone=True);n=ct.ReactorNet([r]);n.rtol=1e-10;n.atol=1e-20
    for j in range(1,10001):n.advance(j*dt)
    rows.append(dict(id=int(i),seconds=time.perf_counter()-t0))
atomic_json('results/t1000_gated_hybrid/ode_cost.json',dict(cases=rows,total_seconds=sum(r['seconds'] for r in rows),note='continuous ReactorNet, 10000 outputs per case'))
print('ODE_COST',sum(r['seconds'] for r in rows))
