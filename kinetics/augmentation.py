"""Training-only off-trajectory perturbations with re-integrated Cantera labels."""
import hashlib
import json
import time
from pathlib import Path
import cantera as ct
import numpy as np
from .data import mechanism
from .physics import element_matrix,project_species


def augment(source,destination,count=8000,seed=710,level='mild'):
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Never overwrite augmentation')
    source_path=Path(source);d=np.load(source_path)
    ids=np.flatnonzero(d['split']=='train');states=d['states'][ids];pressures=d['pressures'][ids]
    dt=float(d['dt']);rng=np.random.default_rng(seed);gas=mechanism();a=element_matrix(gas)
    if level=='mild':sigma,temp_sigma,trace_upper=.03,.003,1e-11
    elif level=='moderate':sigma,temp_sigma,trace_upper=.12,.01,1e-9
    elif level=='broad':sigma,temp_sigma,trace_upper=.3,.02,1e-7
    else:raise ValueError('unknown perturbation level')
    change=np.abs(np.diff(states[:,:,0],axis=1)).reshape(-1)
    weights=.5/len(change)+.5*(change+1e-8)/(change+1e-8).sum()
    picks=rng.choice(len(change),size=count,p=weights)
    x=[];y=[];parents=[];max_elements=0.;start=time.perf_counter()
    for k,pick in enumerate(picks):
        i,j=divmod(int(pick),states.shape[1]-1);state=states[i,j]
        proposal=state[1:]*np.exp(rng.normal(0,sigma,9))
        # Inject trace seeds into reactive species, keep original elemental inventory.
        proposal[:8]+=10**rng.uniform(-22,np.log10(trace_upper),8)
        composition=project_species(proposal,state[1:],a)
        temperature=float(np.clip(state[0]*np.exp(rng.normal(0,temp_sigma)),800,3500))
        gas.TPY=temperature,pressures[i],composition
        reactor=ct.IdealGasConstPressureReactor(gas,clone=True)
        net=ct.ReactorNet([reactor]);net.rtol=1e-10;net.atol=1e-20
        net.advance(dt)
        next_y=reactor.phase.Y
        if next_y.min() < -1e-14:raise RuntimeError('Cantera negative target beyond roundoff')
        x.append(np.r_[temperature,composition,pressures[i]])
        y.append(np.r_[reactor.T,np.maximum(next_y,0)])
        parents.append([int(ids[i]),j])
        max_elements=max(max_elements,float(np.max(np.abs(a@composition-a@state[1:]))))
        if (k+1)%2000==0:print('AUGMENT',level,k+1,'seconds',round(time.perf_counter()-start,2),flush=True)
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,x=x,y=y,parent=parents,dt=dt,seed=seed)
    meta=dict(count=count,seed=seed,level=level,multiplicative_log_sigma=sigma,temperature_log_sigma=temp_sigma,
              trace_floor_log10_range=[-22,float(np.log10(trace_upper))],pressure_perturbed=False,
              parent_split='train',parent_ids=ids.tolist(),dt=dt,labels='Cantera independently integrated from each perturbed input',
              source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
              sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),max_element_inventory_change=max_elements,
              seconds=time.perf_counter()-start,selection_weights='50% uniform, 50% temperature increment magnitude')
    destination.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    return meta
