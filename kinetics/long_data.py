"""Long fixed-dt chemistry references plus training-only equilibrium thinning.

Input times are nonuniform after selection; EVERY supervised target remains
exactly one fixed dt ahead. Slow induction is not labeled equilibrium merely
because its instantaneous reaction/temperature change is small.
"""
import argparse,hashlib,json,time
from pathlib import Path
import cantera as ct
import numpy as np
import pandas as pd
from .data import mechanism


def equilibrium_mask(states,equilibrium,temperature_tol=1.,relative_species_tol=.05,
                     absolute_species_tol=1e-4,species_floor=1e-8,hold_steps=25):
    state=np.asarray(states);eq=np.asarray(equilibrium)
    dy=np.abs(state[:,1:]-eq[1:])
    near=(np.abs(state[:,0]-eq[0])<=temperature_tol)&(dy.sum(axis=1)<=absolute_species_tol)&np.all(dy/np.maximum(eq[1:],species_floor)<=relative_species_tol,axis=1)
    near_pair=near[:-1]&near[1:]
    result=np.zeros(len(near_pair),bool)
    if len(near_pair)>=hold_steps:
        count=np.convolve(near_pair.astype(int),np.ones(hold_steps,int),mode='valid')
        result[hold_steps-1:]=count==hold_steps
    return result


def select_transitions(states,equilibrium,max_active=320,max_equilibrium=8,seed=71):
    eq=equilibrium_mask(states,equilibrium);active=np.flatnonzero(~eq);steady=np.flatnonzero(eq)
    delta=np.abs(np.diff(states,axis=0))
    species_scale=np.maximum(np.maximum(states[:-1,1:],states[1:,1:]),1e-8)
    activity=np.max(delta[:,1:]/species_scale,axis=1)+delta[:,0]/np.maximum(states[:-1,0],300.)
    rng=np.random.default_rng(seed)
    if len(active)<=max_active:chosen_active=active
    else:
        mandatory=active[:min(8,max_active)]
        ntime=max(1,int(max_active*.4))
        time_points=active[np.linspace(0,len(active)-1,ntime).astype(int)]
        weights=activity[active]
        weights=weights+max(float(weights.mean())*.02,1e-30)
        cumulative=np.cumsum(weights)
        targets=np.linspace(0,cumulative[-1],max_active-ntime+1)[1:]
        arc_points=active[np.minimum(np.searchsorted(cumulative,targets),len(active)-1)]
        chosen_active=np.unique(np.r_[mandatory,time_points,arc_points])
        # Reserve mandatory start states, then thin excess evenly if needed.
        if len(chosen_active)>max_active:
            others=np.setdiff1d(chosen_active,mandatory)
            others=others[np.linspace(0,len(others)-1,max_active-len(mandatory)).astype(int)]
            chosen_active=np.unique(np.r_[mandatory,others])
        elif len(chosen_active)<max_active:
            remaining=np.setdiff1d(active,chosen_active)
            w=activity[remaining]+max(float(activity[active].mean())*.02,1e-30)
            extra=rng.choice(remaining,max_active-len(chosen_active),replace=False,p=w/w.sum())
            chosen_active=np.sort(np.r_[chosen_active,extra])
    chosen_eq=steady[np.linspace(0,len(steady)-1,min(max_equilibrium,len(steady))).astype(int)] if len(steady) and max_equilibrium else np.array([],int)
    selected=np.sort(np.r_[chosen_active,chosen_eq]).astype(int)
    return selected,eq,activity


def generate(source,destination,horizon=.01,max_active=320,max_equilibrium=8,seed=730):
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Never overwrite long data')
    destination.mkdir(parents=True)
    source=Path(source);old=np.load(source);source_states=old['states'];pressures=old['pressures'];splits=old['split'];dt=float(old['dt']);steps=round(horizon/dt)
    start=time.perf_counter();all_states=[];equilibria=[];minimum=0.
    for i in range(len(splits)):
        initial=source_states[i,0]
        gas=mechanism();gas.TPY=initial[0],pressures[i],initial[1:]
        reactor=ct.IdealGasConstPressureReactor(gas,clone=True)
        network=ct.ReactorNet([reactor]);network.rtol=1e-10;network.atol=1e-20
        trace=np.empty((steps+1,10));trace[0]=initial
        for j in range(1,steps+1):
            network.advance(j*dt);y=reactor.phase.Y
            minimum=min(minimum,float(y.min()))
            if y.min() < -1e-14:raise RuntimeError('Cantera negative beyond roundoff')
            trace[j]=np.r_[reactor.T,np.maximum(y,0)]
        eq=mechanism();eq.TPY=initial[0],pressures[i],initial[1:];eq.equilibrate('HP',rtol=1e-10)
        all_states.append(trace);equilibria.append(np.r_[eq.T,eq.Y])
        if (i+1)%10==0:print('LONG_REFERENCE',i+1,'/',len(splits),'seconds',round(time.perf_counter()-start,2),flush=True)
    full=np.asarray(all_states);equilibria=np.asarray(equilibria)
    np.savez_compressed(destination/'trajectories.npz',states=full,pressures=pressures,split=splits,species=old['species'],
        element_matrix=old['element_matrix'],elements=old['elements'],initials=old['initials'],equilibrium=equilibria,dt=dt)
    inputs=[];outputs=[];parents=[];sample_rows=[];audit=[]
    for i in np.flatnonzero(splits=='train'):
        selected,mask,activity=select_transitions(full[i],equilibria[i],max_active,max_equilibrium,seed+int(i))
        inputs.append(np.c_[full[i,selected],np.full(len(selected),pressures[i])]);outputs.append(full[i,selected+1])
        parents.extend([[int(i),int(j)] for j in selected])
        sample_rows.append(dict(trajectory=int(i),total_candidates=steps,available_equilibrium=int(mask.sum()),
            available_non_equilibrium=int((~mask).sum()),selected=len(selected),selected_equilibrium=int(mask[selected].sum()),
            selected_non_equilibrium=int((~mask[selected]).sum()),first_input_time=float(selected[0]*dt),last_input_time=float(selected[-1]*dt)))
        audit.extend(dict(trajectory=int(i),step=int(j),input_time=float(j*dt),target_time=float((j+1)*dt),equilibrium=bool(mask[j]),activity=float(activity[j])) for j in selected)
    x=np.concatenate(inputs);y=np.concatenate(outputs)
    np.savez_compressed(destination/'training_pairs.npz',x=x,y=y,parent=np.asarray(parents),dt=dt)
    pd.DataFrame(sample_rows).to_csv(destination/'sampling_summary.csv',index=False)
    pd.DataFrame(audit).to_csv(destination/'selected_inputs.csv',index=False)
    meta=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        dense_sha256=hashlib.sha256((destination/'trajectories.npz').read_bytes()).hexdigest(),
        pair_sha256=hashlib.sha256((destination/'training_pairs.npz').read_bytes()).hexdigest(),
        horizon=horizon,dt=dt,steps=steps,train_pairs=len(x),train_parent_ids=np.flatnonzero(splits=='train').tolist(),
        max_non_equilibrium_per_trajectory=max_active,max_equilibrium_per_trajectory=max_equilibrium,
        selection='first states + time stratification + concentration/temperature arc-length; train only',
        equilibrium_criterion=dict(temperature_K=1.,species_relative=.05,species_L1=1e-4,species_floor=1e-8,persistence_steps=25),
        equilibrium_reference='Cantera HP equilibrium from the SAME initial enthalpy/pressure/element inventory',
        validation_test='full fixed-dt trajectories, never thinned',seed=seed,min_solver_mass_fraction=minimum,
        seconds=time.perf_counter()-start)
    (destination/'manifest.json').write_text(json.dumps(meta,indent=2))
    print('LONG_DATA_COMPLETE',meta,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='results/verified/dataset.npz');parser.add_argument('--output',default='results/long10ms_data');parser.add_argument('--horizon',type=float,default=.01);parser.add_argument('--max-active',type=int,default=320);parser.add_argument('--max-equilibrium',type=int,default=8);args=parser.parse_args()
    generate(args.source,args.output,args.horizon,args.max_active,args.max_equilibrium)
