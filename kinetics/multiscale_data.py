"""Independent multi-scale log sampling based on Zhang 2022, Section 2.5.

No state is obtained by perturbing a trajectory or a previously sampled state.
The normalization choice and deviations from the paper are recorded explicitly.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cantera as ct
import numpy as np
import pandas as pd
from .data import mechanism, SPECIES


def sample_states(count, k, rng, cfg, hot=False):
    low,high=cfg['hot_temperature_K'] if hot else cfg['temperature_K']
    t=rng.uniform(low,high,count)
    p=rng.uniform(*cfg['pressure_atm'],count)*ct.one_atm
    # N2 is independently log sampled, then retained as the simplex anchor.
    n2=10**rng.uniform(*np.log10(cfg['nitrogen_mass_fraction']),count)
    exponents=rng.uniform(-4 if hot else -k,0,(count,9))
    if not hot:
        for species in ('H2','O2'):
            exponents[:,SPECIES.index(species)]=rng.uniform(*cfg['fuel_oxidizer_log10_bounds'],count)
    # Nitrogen has its own distribution, so the dummy column is discarded.
    raw=10**exponents[:,:8]
    factor=(1-n2)/raw.sum(axis=1)
    y=np.c_[raw*factor[:,None],n2]
    return np.c_[t,y,p],factor


def integrate_labels(x,steps,dt,gas):
    gas.TPY=x[0],x[-1],x[1:10]
    reactor=ct.IdealGasConstPressureReactor(gas,clone=True)
    net=ct.ReactorNet([reactor]);net.rtol=1e-10;net.atol=1e-20
    inputs=[];outputs=[];hrr=[]
    for j in range(steps):
        if reactor.phase.Y.min() < -1e-14:raise ValueError('negative_Cantera_input')
        initial=np.r_[reactor.T,np.maximum(reactor.phase.Y,0),x[-1]]
        hrr.append(reactor.phase.heat_release_rate)
        net.advance((j+1)*dt)
        yy=reactor.phase.Y.copy()
        if yy.min() < -1e-14:raise ValueError('negative_Cantera_target')
        # Only solver roundoff is floored, as in the original generator.
        yy=np.maximum(yy,0)
        inputs.append(initial);outputs.append(np.r_[reactor.T,yy])
    return np.asarray(inputs),np.asarray(outputs),np.asarray(hrr)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/zhang_multiscale_data.json'))
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=Path(cfg['output'])
    if out.exists():raise FileExistsError('Preserve data; choose a new output')
    out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    start=time.perf_counter();rng=np.random.default_rng(cfg['seed']);gas=mechanism()
    inputs=[];outputs=[];scales=[];groups=[];parts=[];positions=[];rates=[];norm_factors=[];rejected=[];audit=[]
    total_attempts=0;group=0
    for hot,k,goal in [(False,k,cfg['per_scale']) for k in cfg['scales']]+[(True,4,cfg['hot_initial_states'])]:
        accepted=0;attempts=0
        while accepted<goal:
            if time.perf_counter()-start>cfg['max_seconds']:raise TimeoutError('Data generation budget exceeded')
            if attempts>max(100,20*goal):raise RuntimeError('Too many inadmissible candidates')
            draws,factors=sample_states(min(goal-accepted,128),k,rng,cfg,hot)
            for x,factor in zip(draws,factors):
                attempts+=1;total_attempts+=1
                reason=None
                try:
                    xx,yy,q=integrate_labels(x,2 if hot else 1,cfg['dt'],gas)
                    if not np.isfinite(xx).all() or not np.isfinite(yy).all() or not np.isfinite(q).all():
                        reason='nonfinite_label_or_HRR'
                    elif np.any(yy[:,0]<cfg['temperature_admissibility_K'][0]) or np.any(yy[:,0]>cfg['temperature_admissibility_K'][1]):
                        reason='target_outside_existing_200_6000K_model_domain'
                except (ValueError,ct.CanteraError) as e:
                    reason='Cantera_failure:'+type(e).__name__
                if reason:
                    rejected.append(dict(hot=hot,k=k,reason=reason,T=x[0],P_Pa=x[-1]))
                    continue
                for j in range(len(xx)):
                    inputs.append(xx[j]);outputs.append(yy[j]);rates.append(q[j]);scales.append(k)
                    groups.append(group);parts.append('hot_two_step' if hot else 'multiscale');positions.append(j);norm_factors.append(factor)
                accepted+=1;group+=1
        audit.append(dict(kind='hot_two_step' if hot else 'multiscale',k=k,initial_states=accepted,attempts=attempts,pairs=accepted*(2 if hot else 1)))
        print('MULTISCALE',audit[-1],'seconds',round(time.perf_counter()-start,2),flush=True)
    x=np.asarray(inputs);y=np.asarray(outputs);matrix=np.array([[gas.n_atoms(s,e)*gas.atomic_weight(e)/gas.molecular_weights[j]
        for j,s in enumerate(SPECIES)] for e in ('H','O','N')])
    np.testing.assert_allclose(x[:,1:10].sum(axis=1),1,atol=2e-14)
    np.testing.assert_allclose(y[:,1:].sum(axis=1),1,atol=2e-10)
    error=np.max(np.abs((y[:,1:]-x[:,1:10])@matrix.T))
    assert error<1e-9
    np.savez_compressed(out/'training_pairs.npz',x=x,y=y,dt=cfg['dt'],scale_k=scales,group_id=groups,
        kind=parts,group_step=positions,initial_hrr_W_m3=rates,normalization_factor=norm_factors,
        sampling_method='independent_multiscale_log_sampling',used_manifold_data=False)
    pd.DataFrame(audit).to_csv(out/'sampling_summary.csv',index=False)
    pd.DataFrame(rejected,columns=['hot','k','reason','T','P_Pa']).to_csv(out/'rejections.csv',index=False)
    pd.DataFrame(dict(kind=parts,k=scales,group_id=groups,group_step=positions,T_K=x[:,0],P_atm=x[:,-1]/ct.one_atm,
        N2=x[:,9],initial_hrr_W_m3=rates)).to_csv(out/'samples.csv',index=False)
    meta=dict(status='completed',method='Zhang et al. 2022 Section 2.5, downscaled sampling ablation',
        doi='10.1016/j.combustflame.2022.112319',pairs=len(x),independent_initial_states=group,
        attempts=total_attempts,rejected_initial_states=len(rejected),seconds=time.perf_counter()-start,
        perturbation=False,manifold_data_used=False,validation_or_test_used=False,
        pressure_atm=cfg['pressure_atm'],target_dt=cfg['dt'],species=SPECIES,
        normalization=cfg['normalization'],normalization_is_explicit_implementation_assumption=True,
        hrr_filter_applied=False,hrr_filter_reason='No HRR rejection threshold is specified in Section 2.5',
        negative_initial_hrr_pairs=int((np.asarray(rates)<0).sum()),
        max_element_residual=float(error),
        negative_HRR_policy='retained when the Cantera state/label is admissible; HRR sign alone is not a rejection criterion',
        label_admissibility='finite Cantera results, negative Y only within solver roundoff, target T within existing 200-6000 K model domain',
        fidelity_exceptions=['8834 pairs instead of 5.3 million','9-species/28-reaction existing mechanism instead of 8-species/16-reaction paper mechanism',
            'fixed dt=2us for existing benchmark','explicit N2-preserving simplex normalization; Section 2.5 does not specify normalization details'],
        pair_sha256=hashlib.sha256((out/'training_pairs.npz').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(meta,indent=2));print('COMPLETED',meta,flush=True)


if __name__=='__main__':main()
