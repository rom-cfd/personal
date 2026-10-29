"""Training-only thinning audit and plots; never fits or selects on test data."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/kinetics-cache')
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('results/long10ms_data');fig=root/'figures';fig.mkdir(exist_ok=True)
    dense=np.load(root/'trajectories.npz');states=dense['states'];pressure=dense['pressures'];dt=float(dense['dt'])
    sample=pd.read_csv(root/'sampling_summary.csv');selected=pd.read_csv(root/'selected_inputs.csv');meta=json.loads((root/'manifest.json').read_text())
    before=int(sample.total_candidates.sum());before_eq=int(sample.available_equilibrium.sum());after=len(selected);after_eq=int(selected.equilibrium.sum())
    f,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(['Dense candidates','Selected train'],[before,after]);axes[0].set_ylabel('Training transitions')
    axes[1].bar(['Dense candidates','Selected train'],[100*before_eq/before,100*after_eq/after]);axes[1].set(ylabel='Equilibrium fraction (%)',ylim=(0,100))
    f.tight_layout();f.savefig(fig/'sampling_counts.png',dpi=180);plt.close(f)
    f,ax=plt.subplots(figsize=(10,4))
    bins=np.linspace(0,meta['horizon']*1e3,41)
    ax.hist([selected.loc[~selected.equilibrium,'input_time']*1e3,selected.loc[selected.equilibrium,'input_time']*1e3],bins=bins,stacked=True,label=['Non-equilibrium / induction','Equilibrium anchors'])
    ax.set(xlabel='Selected input time (ms)',ylabel='Training transitions',yscale='log');ax.legend();f.tight_layout();f.savefig(fig/'selected_times.png',dpi=180);plt.close(f)
    ids=np.flatnonzero(dense['split']=='train');ordered=ids[np.argsort(states[ids,0,0])];examples=[int(ordered[0]),int(ordered[len(ordered)//2]),int(ordered[-1])]
    f,axes=plt.subplots(3,1,figsize=(11,9))
    times=np.arange(states.shape[1])*dt*1e3
    for i,ax in zip(examples,axes):
        a=selected[selected.trajectory==i];jj=a.step.to_numpy(int);eq=a.equilibrium.to_numpy(bool)
        ax.plot(times,states[i,:,0],'k-',lw=1,label='Dense Cantera')
        ax.scatter(times[jj[~eq]],states[i,jj[~eq],0],s=8,label='Retained non-equilibrium')
        ax.scatter(times[jj[eq]],states[i,jj[eq],0],s=25,marker='x',label='Retained equilibrium')
        ax.axhline(dense['equilibrium'][i,0],color='gray',ls=':',label='HP equilibrium T')
        ax.set(xlabel='Time (ms)',ylabel='T (K)',title=f'Training IC {i}: T0={states[i,0,0]:.0f} K, P={pressure[i]/101325:.2f} atm')
    axes[0].legend(fontsize=8);f.tight_layout();f.savefig(fig/'sampling_examples.png',dpi=180);plt.close(f)
    text=f'''# 10 ms training-data generation and equilibrium downsampling

Full Cantera trajectories cover **0–10 ms**, with dt=2 microseconds and 5,000
transitions per initial condition. The original 90 initial conditions and
trajectory-disjoint train/validation/test membership are unchanged.

Only training input times are thinned. A selected input at j always has target
j+1, not the next retained sample. The supervised prediction interval is therefore
**2 microseconds for every pair**, even when input sampling times have large gaps.
Validation and test retain the entire 5,001-point profile.

| Quantity | Count |
|---|---:|
| Dense training candidates | {before:,} |
| Dense equilibrium candidates | {before_eq:,} ({100*before_eq/before:.1f}%) |
| Selected training pairs | {after:,} |
| Selected non-equilibrium/induction | {after-after_eq:,} |
| Selected equilibrium anchors | {after_eq:,} ({100*after_eq/after:.1f}%) |

A state is considered near equilibrium only when it agrees with the Cantera HP
equilibrium at the same initial enthalpy, pressure and elemental inventory:
temperature within 1 K, species L1 difference <=1e-4, and each species within 5%
with a 1e-8 floor. Both current/next states must satisfy this, persistently for
25 transitions (50 microseconds). Slow initial induction far from HP equilibrium
is NOT classified as equilibrium simply because dT/dt is small.

Each training trajectory keeps at most 320 non-equilibrium input times, combining
initial states, time stratification and concentration/temperature arc length.
At most eight equilibrium anchors are retained, including late-time points.
Anchors are kept because removing all equilibrium examples could worsen long-term
drift. Selection may change training-distribution weights and must be assessed by
full-profile validation, not assumed to improve the ANN automatically.

No perturbation augmentation is used in this dataset. Its statistics and scalers
must be fit afresh; the old 0.4-ms-trained ten networks are comparison artifacts,
not reused as newly trained members.

Original source SHA256: `{meta['source_sha256']}`.
Dense 10-ms SHA256: `{meta['dense_sha256']}`.
Selected pair SHA256: `{meta['pair_sha256']}`.

- [Sampling summary](results/long10ms_data/sampling_summary.csv)
- [Every selected input and target time](results/long10ms_data/selected_inputs.csv)
- [Counts](results/long10ms_data/figures/sampling_counts.png)
- [Retained time distribution](results/long10ms_data/figures/selected_times.png)
- [Representative training profiles](results/long10ms_data/figures/sampling_examples.png)
- [Generator](kinetics/long_data.py)
'''
    Path('LONG_DATA_REPORT.md').write_text(text)
    print('Sampling report written',before,after,before_eq,after_eq)


if __name__=='__main__':main()
