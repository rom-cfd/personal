"""Audit saved raw hydrogen predictions; no model inference or retraining.

CSV reports are evidence inventories, not additive trial counts: several reuse
the same baseline. Successful configurations are cross-checked against full
saved trajectories and counted once. Shorter historical horizons are separate.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
REPO = Path('/workspace/personal')
VARIABLES = ['T','H2','H','O','O2','OH','H2O','HO2','H2O2','N2']


def scores(truth, trace):
    if truth.shape != trace.shape or not np.isfinite(trace).all():
        raise ValueError('Complete, aligned, finite traces are required')
    # Independent L2 implementation equals RMSE/RMS for equal sample counts.
    return np.linalg.norm(truth-trace,axis=0)/np.linalg.norm(truth,axis=0)


def case_rows(path, reference, label, horizon_ms):
    rows=[]
    with np.load(path) as archive:
        for key in archive.files:
            ic=int(key.split('_')[-1])
            trace=archive[key]
            truth=reference[ic]
            complete=trace.shape==truth.shape and np.isfinite(trace).all()
            metric=scores(truth,trace) if complete else np.full(10,np.nan)
            rows.append(dict(configuration=label,trajectory=ic,horizon_ms=horizon_ms,
                complete=bool(complete),worst9=float(metric[:9].max()) if complete else np.nan,
                worst10=float(metric.max()) if complete else np.nan,
                pass9=bool(complete and metric[:9].max()<=.1),
                pass10=bool(complete and metric.max()<=.1),source=str(path)))
    return rows


def main():
    sources=ROOT/'sources'
    coverage=[]
    evidence=[]
    metric_checks=0
    # All canonical raw NRMSE reports. Duplicate baseline rows are not summed.
    for path in sorted(sources.rglob('per_IC_NRMSE.csv')):
        frame=pd.read_csv(path)
        assert set(frame['mode'])=={'raw'}
        if 'split' in frame:
            assert set(frame['split'])=={'test'}
        for (method,ic),g in frame.groupby(['method','trajectory']):
            assert set(g.variable)==set(VARIABLES) and len(g)==10
            complete=bool(g.complete.all())
            if complete:
                assert set(g.points)=={5001}
                np.testing.assert_allclose(g.nrmse,g.full_rmse/g.reference_rms,rtol=2e-10,atol=1e-12)
                metric_checks+=10
            elif g.nrmse.notna().any():
                raise ValueError('Full-horizon score assigned to incomplete trace')
            values=g.set_index('variable').loc[VARIABLES,'nrmse'].to_numpy()
            evidence.append(dict(configuration=method,trajectory=int(ic),horizon_ms=10.,complete=complete,
                worst9=float(values[:9].max()) if complete else np.nan,
                worst10=float(values.max()) if complete else np.nan,
                pass9=bool(complete and values[:9].max()<=.1),
                pass10=bool(complete and values.max()<=.1),source=str(path)))
        coverage.append(dict(source=str(path),kind='canonical_raw_report',horizon_ms=10.,
            predictor_groups=int(frame.method.nunique()),note='Baseline reuse: do not sum denominators across reports'))

    with np.load(REPO/'results/long10ms_data/trajectories.npz') as data:
        truth=data['states'];reference={i:x for i,x in enumerate(truth)}
        matrix=data['element_matrix'];initial=data['initials']
    for path in sorted(sources.rglob('traces.npz')):
        if not any('/'+name+'/' in str(path) for name in ['deeponet_long_study','tp_phi_study']):
            continue
        if '_raw_test/' not in str(path):
            continue
        label=path.parents[4].name+'/'+path.parents[1].name
        records=case_rows(path,reference,label,10.)
        evidence.extend(records)
        marker=json.loads((path.parent/'complete.json').read_text())
        assert marker['mode']=='raw' and marker['split']=='test'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==marker['output_sha256']['traces.npz']
        coverage.append(dict(source=str(path),kind='legacy_10ms_raw_trace',horizon_ms=10.,predictor_groups=1,note='Saved evaluation marked raw'))

    for family in ['temporal_pilot','temporal_pilot_long']:
        base=REPO/'results'/family/'run_001'
        frame=pd.read_csv(base/'evaluations/test/rollouts.csv')
        frame=frame[frame['mode']=='raw']
        for predictor in frame.predictor.unique():
            path=base/'evaluations/test'/f'{predictor}_raw'/'traces.npz'
            rows=case_rows(path,reference,family+'/'+predictor,10.)
            for row in rows:
                saved=frame[(frame.predictor==predictor)&(frame.trajectory==row['trajectory'])].iloc[0]
                assert row['complete']==bool(saved.complete)
                assert row['pass9']==bool(saved.joint9_le_0p1)
            evidence.extend(rows)
        coverage.append(dict(source=str(base/'evaluations/test/rollouts.csv'),kind='temporal_raw_seed_and_mean3',
                             horizon_ms=10.,predictor_groups=12,note='Nine seeds and three mean3 predictors'))

    # Earlier short-horizon studies are audited at their actual saved horizon.
    for family in ['verified','final']:
        base=REPO/'results'/family
        with np.load(base/'dataset.npz') as data:
            old_reference={i:x for i,x in enumerate(data['states'])}
        paths=sorted(base.glob('rollout_*repair0.npz'))
        for path in paths:
            evidence.extend(case_rows(path,old_reference,family+'/'+path.stem,.4))
        coverage.append(dict(source=str(base),kind='early_0p4ms_raw',horizon_ms=.4,
                             predictor_groups=len(paths),note='final is superseded; no successes in either snapshot'))
    with np.load(REPO/'results/verified/posteriori/cantera_reference.npz') as data:
        old_reference={int(i):x for i,x in zip(data['ids'],data['states'])}
    for folder,pattern in [('results/verified/posteriori','*_repair0.npz'),('results/refinement_test/traces','*_raw.npz')]:
        base=REPO/folder;paths=sorted(base.glob(pattern))
        for path in paths:
            evidence.extend(case_rows(path,old_reference,folder+'/'+path.stem,2.))
        coverage.append(dict(source=str(base),kind='early_2ms_raw',horizon_ms=2.,predictor_groups=len(paths),note='Same saved Cantera reference used by original evaluation'))

    frame=pd.DataFrame(evidence)
    frame.to_csv(ROOT/'all_saved_raw_evidence.csv',index=False)
    pd.DataFrame(coverage).to_csv(ROOT/'source_coverage.csv',index=False)
    passed=frame[frame.pass9]
    assert set(passed.trajectory)=={39}
    assert len(passed)==3 and passed.pass10.all()
    targets=[
        ('old_TP_long_gamma0.1',next((sources/'results/deeponet_long_study').glob('run_001/evaluations/tp10_raw_test/*/traces.npz'))),
        ('gamma_P_only_long_gamma0.2',next((sources/'results/gamma_long').glob('run_001/gamma_0.2/evaluations/p_only10_raw_test/*/traces.npz'))),
        ('current_single_step_long_gamma0.2',REPO/'results/temporal_pilot_long/run_001/evaluations/test/single_step_mean3_raw/traces.npz')]
    independently_verified=[]
    for label,path in targets:
        with np.load(path) as archive:
            trace=archive['IC_39'] if 'IC_39' in archive.files else archive['trajectory_39']
        metric=scores(reference[39],trace)
        early=scores(reference[39][:51],trace[:51])
        independently_verified.append(dict(configuration=label,IC=39,nrmse=dict(zip(VARIABLES,metric.tolist())),
            max10=float(metric.max()),early_max10=float(early.max()),
            mass_sum_max=float(np.abs(trace[:,1:].sum(axis=1)-1).max()),
            elemental_max=float(np.abs((trace[:,1:]-reference[39][0,1:])@matrix.T).max()),
            trace_path=str(path),initial_T=float(initial[39,0]),initial_P_atm=float(initial[39,1]/101325),initial_phi=float(initial[39,2])))
    (ROOT/'independently_verified_passes.json').write_text(json.dumps(independently_verified,indent=2))
    table=[];variables=[]
    for row in independently_verified:
        table.append({k:v for k,v in row.items() if k!='nrmse'})
        for variable,value in row['nrmse'].items():
            variables.append(dict(configuration=row['configuration'],trajectory=row['IC'],variable=variable,nrmse=value))
    pd.DataFrame(table).to_csv(ROOT/'passing_configurations.csv',index=False)
    pd.DataFrame(variables).to_csv(ROOT/'passing_variable_NRMSE.csv',index=False)
    summary=dict(status='passed',revision='53df7a5a54b1373110269a98a8d6b85e93763aac',
        criterion='Complete saved horizon and T+8 reactive species all NRMSE<=0.1; adding N2 leaves the same successes',
        formula='RMSE(Cantera, NN)/RMS(Cantera); dimensionless, equal samples, no percentage',
        successful_10ms_configuration_IC_pairs=3,unique_successful_10ms_ICs=[39],
        tested_IC_count=18,successful_0p4ms_pairs=0,successful_2ms_pairs=0,
        complete_full_and_early_0p1ms_pairs=sum(r['early_max10']<=.1 for r in table),
        report_ratio_checks=metric_checks,
        primary_scope='All saved raw held-out trajectory evaluations, not every trained checkpoint',
        denominator_warning='Reports repeat baseline evaluations. Source coverage counts must not be summed as independent trials.',
        corrected_results_excluded=['committee_original mean/medoid/single wrappers','committee100_long10ms mean/medoid wrappers','all repaired/physics outputs','one-step predictions','validation-only outputs'],
        physical_caveat='NRMSE success is not a mass/element/enthalpy conservation certificate',
        initial_condition=dict(T_K=float(initial[39,0]),P_atm=float(initial[39,1]/101325),phi=float(initial[39,2])))
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
