"""Independent fixed-step, self-fed 0-D ANN vs Cantera ignition/profile study.

No teacher forcing, training, checkpoint selection or transformation tuning.
Raw and clip/sum-normalized rollouts are separate numerical experiments.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/kinetics-cache')
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import cantera as ct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import mechanism
from .study import Predictor, rollouts


def ignition(trace,dt,initial_temperature):
    """400 K rise with interpolated crossing; max dT/dt only for resolved ignition."""
    cross=np.flatnonzero(trace[:,0]>=initial_temperature+400)
    if not len(cross):return dict(tau400=float('nan'),tau_dTdt=float('nan'),cross_index=None)
    j=int(cross[0])
    tau=((j-1)+(initial_temperature+400-trace[j-1,0])/(trace[j,0]-trace[j-1,0]))*dt if j else 0.
    if len(trace)>=3:
        slope=np.gradient(trace[:,0],dt);peak=int(np.argmax(slope))
        tau_peak=peak*dt if 0<peak<len(trace)-1 else float('nan')
    else:tau_peak=float('nan')
    return dict(tau400=float(tau),tau_dTdt=float(tau_peak),cross_index=j)


def reference(source,horizon,destination):
    if destination.exists():raise FileExistsError('Do not overwrite 0-D reference')
    ids=np.flatnonzero(source['split']=='test');dt=float(source['dt']);steps=round(horizon/dt)
    result=np.empty((len(ids),steps+1,10));start=time.perf_counter()
    source_states=source['states'];pressures=source['pressures']
    for k,i in enumerate(ids):
        g=mechanism();initial=source_states[i,0]
        g.TPY=initial[0],pressures[i],initial[1:]
        r=ct.IdealGasConstPressureReactor(g,clone=True);sim=ct.ReactorNet([r]);sim.rtol=1e-10;sim.atol=1e-20
        result[k,0]=initial
        for j in range(1,steps+1):
            sim.advance(j*dt)
            if r.phase.Y.min() < -1e-14:raise ValueError('Negative reference species beyond roundoff')
            result[k,j]=np.r_[r.T,np.maximum(r.phase.Y,0)]
    np.savez_compressed(destination,states=result,ids=ids,dt=dt,pressures=pressures[ids])
    return result,ids,time.perf_counter()-start


def metrics(trace,truth,dt,matrix,horizon_steps):
    n=min(len(trace),horizon_steps+1);a=trace[:n];b=truth[:n]
    delta=a[:,1:]-b[:,1:]
    duration=(n-1)*dt
    integrate=lambda v:np.trapezoid(v,dx=dt,axis=0)/duration if duration else v[0]
    prefix_rmse=np.sqrt(integrate(delta**2))
    reference_l2=integrate(b[:,1:]**2)
    relative_l2=np.sqrt(integrate(delta**2)/np.maximum(reference_l2,1e-16))
    temperature_rmse=float(np.sqrt(integrate((a[:,0]-b[:,0])**2)))
    mass=np.abs(a[:,1:].sum(axis=1)-1)
    element=np.max(np.abs((a[:,1:]-truth[0,1:])@matrix.T),axis=1)
    bad=(mass>.01)|(element>.01)|np.any(a[:,1:]>1.01,axis=1)
    invalid=np.flatnonzero(bad)
    first_invalid=int(invalid[0]) if len(invalid) else None
    complete=n==horizon_steps+1
    ref_ign=ignition(truth[:horizon_steps+1],dt,truth[0,0])
    pred_ign=ignition(a,dt,truth[0,0])
    tau=pred_ign['tau400'];tau_ref=ref_ign['tau400']
    valid_ign=np.isfinite(tau) and (first_invalid is None or pred_ign['cross_index']<first_invalid)
    if not complete and not np.isfinite(tau):status='integration_failed_before_ignition'
    elif np.isfinite(tau_ref) and np.isfinite(tau):status='both_ignite'
    elif np.isfinite(tau_ref):status='missed_ignition'
    elif np.isfinite(tau):status='false_ignition_within_horizon'
    elif complete:status='both_no_ignition_within_horizon'
    else:status='integration_failed'
    main=dict(complete=complete,coverage=(n-1)/horizon_steps,completed_steps=n-1,
              mass_sum_max=float(mass.max()),elemental_max=float(element.max()),
              first_physical_violation_seconds=first_invalid*dt if first_invalid is not None else float('nan'),
              physically_admissible_complete=bool(complete and first_invalid is None),
              prefix_temperature_rmse=temperature_rmse,prefix_species_rmse=float(np.sqrt(np.mean(prefix_rmse**2))),
              full_temperature_rmse=temperature_rmse if complete else float('nan'),
              full_species_rmse=float(np.sqrt(np.mean(prefix_rmse**2))) if complete else float('nan'),
              full_species_relative_l2_mean=float(relative_l2.mean()) if complete else float('nan'),
              reference_tau400=tau_ref,predicted_tau400=tau,reference_tau_dTdt=ref_ign['tau_dTdt'],
              predicted_tau_dTdt=pred_ign['tau_dTdt'],ignition_status=status,ignition_physically_admissible=bool(valid_ign),
              tau400_relative_error=abs(tau-tau_ref)/tau_ref if np.isfinite(tau) and np.isfinite(tau_ref) else float('nan'),
              physical_tau400_relative_error=abs(tau-tau_ref)/tau_ref if valid_ign and np.isfinite(tau_ref) else float('nan'))
    rows=[dict(species=j,prefix_rmse=float(prefix_rmse[j]),full_rmse=float(prefix_rmse[j]) if complete else float('nan'),
               full_relative_l2=float(relative_l2[j]) if complete else float('nan')) for j in range(9)]
    return main,rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',default='results/verified');ap.add_argument('--horizon',type=float,default=.002);args=ap.parse_args()
    study=Path(args.study);out=study/'posteriori'
    if out.exists():raise FileExistsError('Use a new study/output; a posteriori results immutable')
    out.mkdir();start=time.perf_counter()
    source=np.load(study/'dataset.npz');states,ids,reference_seconds=reference(source,args.horizon,out/'cantera_reference.npz')
    dt=float(source['dt']);matrix=source['element_matrix'];species=source['species']
    rollout_data=dict(states=states,pressures=source['pressures'][ids],split=np.full(len(ids),'test'),element_matrix=matrix)
    comparison=pd.read_csv(study/'test_comparison.csv');rows=[];species_rows=[];times=[]
    for _,model in comparison.iterrows():
        name=model['name'];seed=int(model['seed']);tag=f'{name}_seed{seed}'
        p=Predictor.load(study/'models'/tag)
        for repair in (False,True):
            began=time.perf_counter();diagnostics,traces=rollouts(p,rollout_data,'test',repair)
            elapsed=time.perf_counter()-began
            np.savez_compressed(out/f'{tag}_repair{int(repair)}.npz',**{f'trajectory_{int(ids[i])}':trace for i,trace in traces})
            times.append(dict(name=name,seed=seed,repair=repair,seconds=elapsed))
            for i,trace in traces:
                for horizon in (.0004,args.horizon):
                    steps=round(horizon/dt)
                    m,sp=metrics(trace,states[i],dt,matrix,steps)
                    common=dict(name=name,seed=seed,repair=repair,trajectory=int(ids[i]),horizon=horizon)
                    rows.append(dict(**common,**m))
                    species_rows.extend(dict(**common,**{**v,'species':str(species[v['species']])}) for v in sp)
            pd.DataFrame(rows).to_csv(out/'trajectory_metrics.csv',index=False)
            pd.DataFrame(species_rows).to_csv(out/'species_profile_errors.csv',index=False)
            pd.DataFrame(times).to_csv(out/'timing.csv',index=False)
        print('POSTERIORI',tag,flush=True)
    frame=pd.DataFrame(rows)
    summary=frame.groupby(['name','repair','horizon']).agg(
        trajectories=('complete','size'),completion_fraction=('complete','mean'),
        physically_admissible_fraction=('physically_admissible_complete','mean'),
        full_temperature_rmse_completed_only=('full_temperature_rmse','mean'),
        full_species_rmse_completed_only=('full_species_rmse','mean'),
        finite_ignition_pairs=('tau400_relative_error','count'),
        tau400_relative_error_igniting_pairs=('tau400_relative_error','mean'),
        physically_valid_ignition_count=('physical_tau400_relative_error','count'),
        physical_tau400_relative_error=('physical_tau400_relative_error','mean'))
    summary.to_csv(out/'summary.csv')
    reference_ignitions=[dict(trajectory=int(i),**ignition(states[k],dt,states[k,0,0])) for k,i in enumerate(ids)]
    pd.DataFrame(reference_ignitions).to_csv(out/'reference_ignition.csv',index=False)
    # All held-out initial conditions, fixed seed 11, no selection by visual quality.
    names=comparison.name.unique()
    for repair in (False,True):
        figure,axes=plt.subplots(3,6,figsize=(20,10),sharex=True)
        for k,(i,ax) in enumerate(zip(ids,axes.flat)):
            t=np.arange(states.shape[1])*dt*1e3;ax.plot(t,states[k,:,0],'k--',lw=2,label='Cantera')
            for name in names:
                trace=np.load(out/f'{name}_seed11_repair{int(repair)}.npz')[f'trajectory_{int(i)}']
                ax.plot(t[:len(trace)],trace[:,0],lw=.8,label=name)
            ax.set_title(f'IC {i}: T0={states[k,0,0]:.0f} K',fontsize=9)
            ax.set_ylim(700,6200);ax.set_xlabel('ms');ax.set_ylabel('K')
        axes.flat[-1].legend(fontsize=5);figure.suptitle('Self-fed 0-D temperature profiles: '+('clip/normalize diagnostic' if repair else 'raw ANN'))
        figure.tight_layout();figure.savefig(out/f'all_temperature_profiles_repair{int(repair)}.png',dpi=150);plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(12,5))
    for repair,ax in zip((False,True),axes):
        subset=frame[(frame.repair==repair)&(frame.horizon==args.horizon)]
        for name,group in subset.groupby('name'):
            paired=group[np.isfinite(group.reference_tau400)&np.isfinite(group.predicted_tau400)]
            ax.scatter(paired.reference_tau400*1e6,paired.predicted_tau400*1e6,s=12,label=name)
        ax.plot([0,args.horizon*1e6],[0,args.horizon*1e6],'k--')
        ax.set(xlabel='Cantera tau(T0+400 K), microseconds',ylabel='ANN tau, microseconds',title='Repaired (numerical crossings)' if repair else 'Raw (numerical crossings)')
    axes[1].legend(fontsize=6);figure.tight_layout();figure.savefig(out/'ignition_delay.png',dpi=180);plt.close(figure)
    (out/'manifest.json').write_text(json.dumps(dict(dt=dt,horizons=[.0004,args.horizon],test_initial_conditions=ids.tolist(),
        seeds=[11,22,33],teacher_forcing=False,reference_seconds=reference_seconds,wall_seconds=time.perf_counter()-start,
        ignition_definition='primary: first T0+400 K crossing, linear interpolation; secondary: interior maximum sampled dT/dt after 400K rise',
        physical_tolerance='absolute mass-sum and max elemental mass-fraction drift <=0.01 and Y<=1.01',
        incomplete_profile_policy='full error NaN; report coverage and failure separately; prefix errors labeled',
        parameter_selection='none; frozen models from validation-only study'),indent=2))
    print('COMPLETE',time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
