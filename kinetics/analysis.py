"""Post-selection diagnostics and publication figures; never trains or selects."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/kinetics-cache')
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import samples
from .study import Predictor


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/verified');args=ap.parse_args()
    out=Path(args.output);fig=out/'figures';fig.mkdir(exist_ok=True)
    data=np.load(out/'dataset.npz'); x,y,ids=samples(data,'test')
    comparison=pd.read_csv(out/'test_comparison.csv')
    rows=[];timings=[];eps_rows=[];reload_rows=[];negative_rows=[]
    for _,row in comparison.iterrows():
        tag=f'{row["name"]}_seed{int(row["seed"])}'
        p=Predictor.load(out/'models'/tag)
        clean=p.predict(x)
        eps=np.broadcast_to(p.transform.epsilon,(9,))
        eps_rows.extend(dict(name=row['name'],seed=int(row['seed']),species=str(s),epsilon=float(e)) for s,e in zip(data['species'],eps))
        rng=np.random.default_rng(100)
        for amplitude in (1e-16,1e-12,1e-8):
            noisy=x.copy();noise=rng.normal(size=(len(x),8))*amplitude
            noisy[:,1:9]=np.maximum(noisy[:,1:9]+noise,0)
            # Adjust N2 to keep total mass fixed; this does not preserve elemental totals.
            noisy[:,9]=1-noisy[:,1:9].sum(axis=1)
            changed=p.predict(noisy)
            d=changed[:,1:]-clean[:,1:]
            rows.append(dict(name=row['name'],seed=int(row['seed']),noise_std=amplitude,
                             prediction_change_rmse=float(np.sqrt(np.nanmean(d*d))),
                             perturbed_invalid_fraction=float(np.mean(~np.all(np.isfinite(changed),axis=1))),
                             max_input_change=float(np.max(np.abs(noisy-x)))))
        # Warm-up then median of five timed calls; batch and serial costs differ.
        p.predict(x[:1]);p.predict(x)
        batch=[];single=[]
        for _ in range(5):
            start=time.perf_counter();p.predict(x);batch.append((time.perf_counter()-start)/len(x))
            start=time.perf_counter();p.predict(x[:1]);single.append(time.perf_counter()-start)
        timings.append(dict(name=row['name'],seed=int(row['seed']),batch_microseconds_per_sample=1e6*np.median(batch),single_state_microseconds=1e6*np.median(single)))
        saved=np.load(out/f'predictions_{tag}.npz')['pred']
        np.testing.assert_allclose(clean,saved,rtol=1e-10,atol=1e-12,equal_nan=True)
        reload_rows.append(dict(name=row['name'],seed=int(row['seed']),max_absolute_difference=float(np.nanmax(np.abs(clean-saved)))))
        negative=clean[:,1:][clean[:,1:]<0]
        negative_rows.append(dict(name=row['name'],seed=int(row['seed']),minimum_mass_fraction=float(np.nanmin(clean[:,1:])),negative_count=int(len(negative)),negative_absolute_p95=float(np.quantile(-negative,.95)) if len(negative) else 0))
    pd.DataFrame(rows).to_csv(out/'noise_robustness.csv',index=False)
    pd.DataFrame(timings).to_csv(out/'inference_benchmark.csv',index=False)
    pd.DataFrame(eps_rows).to_csv(out/'epsilon_values.csv',index=False)
    pd.DataFrame(reload_rows).to_csv(out/'reload_validation.csv',index=False)
    pd.DataFrame(negative_rows).to_csv(out/'negative_magnitudes.csv',index=False)
    # Aggregate all outcomes; invalid rows remain a separate column.
    numeric=comparison.select_dtypes('number').columns.drop('seed')
    summary=comparison.groupby('name')[numeric].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(out/'summary.csv')
    species=pd.read_csv(out/'species_errors.csv')
    plt.figure(figsize=(11,5))
    for name,group in species.groupby('name'):
        values=group.groupby('species').rmse.mean().reindex(data['species'])
        plt.plot(data['species'],values,'o-',label=name)
    plt.yscale('log');plt.ylabel('Mass-fraction RMSE (finite predictions)');plt.legend(fontsize=7)
    plt.tight_layout();plt.savefig(fig/'species_errors.png',dpi=180);plt.close()
    bins=pd.read_csv(out/'concentration_errors.csv')
    plt.figure(figsize=(10,5))
    # Count-weighted pooled finite RMSE; preserve full per-species distribution CSV.
    for name,group in bins.groupby('name'):
        v=group.assign(sse=group.rmse**2*group.finite_count).groupby('lower')[['sse','finite_count']].sum()
        plt.plot(np.maximum(v.index,1e-20),np.sqrt(v.sse/v.finite_count),'o-',label=name)
    plt.xscale('log');plt.yscale('log');plt.xlabel('True mass-fraction bin lower bound (zero shown at 1e-20)')
    plt.ylabel('Pooled finite RMSE');plt.legend(fontsize=7);plt.tight_layout();plt.savefig(fig/'concentration_errors.png',dpi=180);plt.close()
    roll=pd.read_csv(out/'rollouts.csv')
    stability=[]
    for _,r in roll.iterrows():
        tag=f'{r["name"]}_seed{int(r["seed"])}'
        i=int(r['trajectory']);repair=bool(r['repair'])
        trace=np.load(out/f'rollout_{tag}_repair{int(repair)}.npz')[f'trajectory_{i}']
        truth=data['states'][i,:len(trace)]
        mass=np.abs(trace[:,1:].sum(axis=1)-1)
        element=np.max(np.abs((trace[:,1:]-truth[0,1:])@data['element_matrix'].T),axis=1)
        admissible=(mass<=.01)&(element<=.01)&np.all((trace[:,1:]>=0)&(trace[:,1:]<=1.01),axis=1)
        bad=np.flatnonzero(~admissible)
        prefix=int(bad[0]-1) if len(bad) else len(trace)-1
        complete=len(trace)==data['states'].shape[1]
        accurate=bool(complete and np.sqrt(np.mean((trace[:,1:]-truth[:,1:])**2))<=.01 and np.sqrt(np.mean((trace[:,0]-truth[:,0])**2))<=100)
        stability.append(dict(name=r['name'],seed=int(r['seed']),trajectory=i,repair=repair,
                              completion=complete,conservative_prefix_steps=prefix,
                              conservative_complete=bool(complete and np.all(admissible)),
                              accurate_complete=accurate,
                              stable_accurate_complete=bool(complete and np.all(admissible) and accurate)))
    stability=pd.DataFrame(stability)
    stability.to_csv(out/'stability.csv',index=False)
    figobj,axes=plt.subplots(1,2,figsize=(12,4))
    for repair,ax in zip((False,True),axes):
        for name,group in roll[roll.repair==repair].groupby('name'):
            survival=[np.mean(group.completed_steps>=s) for s in range(data['states'].shape[1])]
            ax.plot(np.arange(len(survival))*float(data['dt'])*1e6,survival,label=name)
        ax.set(xlabel='Time (microseconds)',ylabel='Fraction without stopping',ylim=(-.02,1.02),title='Clip/renormalize diagnostic' if repair else 'Raw numerical completion (not conservation)')
    axes[1].legend(fontsize=7);figobj.tight_layout();figobj.savefig(fig/'rollout_survival.png',dpi=180);plt.close(figobj)
    figobj,axes=plt.subplots(1,2,figsize=(12,4))
    for repair,ax in zip((False,True),axes):
        for name,group in stability[stability.repair==repair].groupby('name'):
            survival=[np.mean(group.conservative_prefix_steps>=s) for s in range(data['states'].shape[1])]
            ax.plot(np.arange(len(survival))*float(data['dt'])*1e6,survival,label=name)
        ax.set(xlabel='Time (microseconds)',ylabel='Conservative admissible fraction',ylim=(-.02,1.02),title='Repaired: mass + elements within 0.01' if repair else 'Raw: mass + elements within 0.01')
    axes[1].legend(fontsize=7);figobj.tight_layout();figobj.savefig(fig/'conservative_survival.png',dpi=180);plt.close(figobj)
    # Predetermined first held-out trajectory, seed 11: no cherry-picked best case.
    i=int(ids[0]);truth=data['states'][i];times=np.arange(len(truth))*float(data['dt'])*1e6
    figobj,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(times,truth[:,0],'k--',label='Cantera');axes[1].plot(times,truth[:,6],'k--',label='Cantera')
    for name in comparison.name.unique():
        trace=np.load(out/f'rollout_{name}_seed11_repair1.npz')[f'trajectory_{i}']
        axes[0].plot(times[:len(trace)],trace[:,0],label=name)
        axes[1].plot(times[:len(trace)],trace[:,6],label=name)
    axes[0].set(xlabel='Time (microseconds)',ylabel='Temperature (K)',title=f'Repaired rollout, held-out trajectory {i}, seed 11')
    axes[1].set(xlabel='Time (microseconds)',ylabel='H2O mass fraction');axes[1].legend(fontsize=7)
    figobj.tight_layout();figobj.savefig(fig/'rollout_example.png',dpi=180);plt.close(figobj)
    plt.figure(figsize=(8,5))
    for name,group in comparison.groupby('name'):
        plt.scatter(group.transformed_mse,group.rmse,label=name)
    plt.xscale('log');plt.yscale('log');plt.xlabel('Standardized transformed-increment MSE');plt.ylabel('Finite physical mass-fraction RMSE')
    plt.legend(fontsize=7);plt.tight_layout();plt.savefig(fig/'loss_vs_physical.png',dpi=180);plt.close()
    # Local derivatives explain low-Y sensitivity, without claiming model stability.
    from .transforms import Transform
    ys=np.geomspace(1e-30,1,301)
    plt.figure(figsize=(8,5))
    for name in comparison.name.unique():
        p=Predictor.load(out/'models'/f'{name}_seed11');t=p.transform
        e=float(np.broadcast_to(t.epsilon,(9,))[0]);g=t.gamma
        if t.kind=='boxcox':derivative=ys**(g-1)
        elif t.kind=='softlog':derivative=1/(ys+e)
        elif t.kind=='asinh':derivative=1/np.sqrt(ys**2+e**2)
        else:derivative=(1+ys/e)**(g-1)/e
        plt.plot(ys,derivative,label=name)
    plt.xscale('log');plt.yscale('log');plt.xlabel('Mass fraction Y');plt.ylabel('Unstandardized |df/dY| (H2 epsilon)')
    plt.legend(fontsize=7);plt.tight_layout();plt.savefig(fig/'transform_sensitivity.png',dpi=180);plt.close()
    (out/'reload_validation.json').write_text(json.dumps(dict(models_checked=len(comparison),all_saved_predictions_reproduced=True),indent=2))
    source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('kinetics').glob('*.py'))}
    (out/'source_hashes.json').write_text(json.dumps(source,indent=2))
    print(summary.to_string())
    print('Reload, perturbation, timing and figures complete.')


if __name__=='__main__':main()
