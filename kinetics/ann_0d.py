"""Standalone 0-D constant-pressure ignition: Cantera vs. the trained GELU DNN.

Cantera integrates IdealGasConstPressureReactor (rtol 1e-10, atol 1e-20) and is
sampled every dt. The DNN starts from the same t=0 state and is self-fed for the
same number of steps. Each step the network predicts the next (Y, P); T is
recovered from the current mixture enthalpy and the new composition (the
configured enthalpy mode). Several --model paths are averaged at every step.
There is no projection, clipping or mass renormalization; a run stops when the
DNN state becomes nonfinite or nonphysically huge, and the stop time is saved.

  python -m kinetics.ann_0d                      # 18 test ICs, best seed-11 model
  python -m kinetics.ann_0d --T0 1100 --P 2 --phi 1 --output results/ann_0d_custom
"""
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl')
import cantera as ct
import numpy as np
from .data import mechanism,SPECIES

BEST=Path('results/zhang_5p3m_gelu_1000/g0.1/seed11/best/model')


def cantera_run(T0,P,phi,dt,steps):
    gas=mechanism();gas.TP=T0,P;gas.set_equivalence_ratio(phi,'H2','O2:1,N2:3.76')
    reactor=ct.IdealGasConstPressureReactor(gas,clone=True)
    net=ct.ReactorNet([reactor]);net.rtol=1e-10;net.atol=1e-20
    out=np.empty((steps+1,10))
    for j in range(steps+1):
        if j:net.advance(j*dt)
        out[j]=np.r_[reactor.T,np.maximum(reactor.phase.Y,0)]
    return out


def dnn_run(predictor,initial,pressures,steps):
    """Batched self-fed rollout; returns per-IC [T,Y9] histories and stop reasons."""
    state=np.c_[initial,pressures];history=[[s[:10].copy()] for s in state]
    active=np.ones(len(state),bool);status=['completed']*len(state)
    for step in range(1,steps+1):
        live=np.flatnonzero(active)
        if not len(live):break
        out=predictor.predict_full(state[live])
        for j,k in enumerate(live):
            y=out[j]
            if not np.isfinite(y).all():active[k]=False;status[k]=f'nonfinite at step {step}';continue
            if abs(y[0])>1e6 or np.max(np.abs(y[1:10]))>1e3 or abs(y[-1])>1e15:
                active[k]=False;status[k]=f'magnitude guard at step {step}';continue
            state[k]=y;history[k].append(y[:10].copy())
    return [np.asarray(h) for h in history],status


def ignition_delay(T,dt,rise=400.):
    hit=np.flatnonzero(T>=T[0]+rise)
    return float(hit[0]*dt) if len(hit) else float('nan')


def plot(cases,dt,path):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    cols=min(6,len(cases));rows=-(-len(cases)//cols)
    fig,axes=plt.subplots(rows,cols,figsize=(3.7*cols,3.3*rows),squeeze=False)
    for c,ax in zip(cases,axes.flat):
        t=np.arange(1,len(c['cantera']))*dt*1e3;ax.plot(t,c['cantera'][1:,0],'k--',lw=1.4,label='Cantera')
        a=c['dnn'];ta=np.arange(1,len(a))*dt*1e3
        if len(a)>1:ax.plot(ta,a[1:,0],'tab:red',lw=1,label='DNN');ax.plot(ta[-1],a[-1,0],'x',color='tab:red',ms=4)
        ax.set(xscale='log',xlim=(dt*1e3,t[-1]),ylim=(200,4000),title=c['label'],xlabel='Time (ms)',ylabel='T (K)')
        ax.text(.03,.95,f"DNN reached {c['dnn_end_ms']:.3g} ms",transform=ax.transAxes,va='top',fontsize=7);ax.grid(alpha=.2)
    for ax in axes.flat[len(cases):]:ax.axis('off')
    axes.flat[0].legend(fontsize=7);fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model',type=Path,nargs='+',default=[BEST],help='model path(s) without .keras; several are averaged each step')
    parser.add_argument('--T0',type=float);parser.add_argument('--P',type=float,help='atm');parser.add_argument('--phi',type=float)
    parser.add_argument('--source',type=Path,default=Path('results/long10ms_data/trajectories.npz'),help='ICs when T0/P/phi are not given')
    parser.add_argument('--split',default='test')
    parser.add_argument('--tend',type=float,default=.02,help='s');parser.add_argument('--dt',type=float,default=2e-6)
    parser.add_argument('--output',type=Path,default=Path('results/ann_0d_best_seed11'))
    args=parser.parse_args();steps=round(args.tend/args.dt)
    if args.output.exists():raise FileExistsError('Use a new --output')
    if args.T0 is not None:ics=[(f'T0={args.T0:g} K, P={args.P:g} atm, phi={args.phi:g}',args.T0,args.P*ct.one_atm,args.phi)]
    else:
        with np.load(args.source) as d:
            ids=np.flatnonzero(d['split']==args.split)
            ics=[(f'IC {i}',*d['initials'][i]) for i in ids]
    began=time.perf_counter()
    truth=[cantera_run(T0,P,phi,args.dt,steps) for _,T0,P,phi in ics];t_cantera=time.perf_counter()-began
    from .zhang_model import ZhangCommittee
    from .enthalpy_temperature import EnthalpyTemperature
    metric=dict(temperature_scale=1.,species_scale=[1.]*9)  # unused by the mean rule
    predictor=EnthalpyTemperature(ZhangCommittee(args.model,list(range(len(args.model))),metric,'mean'))
    began=time.perf_counter()
    traces,status=dnn_run(predictor,np.array([c[0] for c in truth]),np.array([ic[2] for ic in ics]),steps)
    t_dnn=time.perf_counter()-began
    args.output.mkdir(parents=True);cases=[]
    for (label,T0,P,phi),ref,a,s in zip(ics,truth,traces,status):
        n=len(a);err=a[:,0]-ref[:n,0]
        cases.append(dict(label=label,T0=float(T0),P_atm=float(P/ct.one_atm),phi=float(phi),status=s,
            dnn_end_ms=(n-1)*args.dt*1e3,completed=n==steps+1,
            T_rmse_reached_K=float(np.sqrt(np.mean(err**2))),
            T_final_cantera=float(ref[-1,0]),T_final_dnn=float(a[-1,0]),
            tau400_cantera_ms=ignition_delay(ref[:,0],args.dt)*1e3,tau400_dnn_ms=ignition_delay(a[:,0],args.dt)*1e3,
            cantera=ref,dnn=a))
    np.savez_compressed(args.output/'trajectories.npz',dt=args.dt,species=SPECIES,labels=[c['label'] for c in cases],
        **{f'cantera_{k}':c['cantera'] for k,c in enumerate(cases)},**{f'dnn_{k}':c['dnn'] for k,c in enumerate(cases)})
    summary=[{k:v for k,v in c.items() if k not in ('cantera','dnn')} for c in cases]
    (args.output/'summary.json').write_text(json.dumps(dict(models=[str(m) for m in args.model],dt=args.dt,tend=args.tend,
        temperature='enthalpy recovery',cantera_seconds=t_cantera,dnn_seconds=t_dnn,cases=summary),indent=2))
    plot(cases,args.dt,args.output/'temperature.png')
    for c in summary:print(c['label'],c['status'],f"end {c['dnn_end_ms']:.3g} ms",f"T RMSE {c['T_rmse_reached_K']:.1f} K",flush=True)
    print('WROTE',args.output,f'(Cantera {t_cantera:.1f} s, DNN {t_dnn:.1f} s)')


if __name__=='__main__':main()
