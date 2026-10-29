"""Raw physical-increment one-step selection and self-fed 10 ms evaluation.

Negative species and physical constraint violations are observed, never repaired
or used to filter finite committee candidates. Only nonfinite/numerically huge
states terminate a trajectory; temperature outside 200--6000 K is a diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .delta_model import DeltaCommittee
from .metrics import evaluate
from .posteriori import metrics
from .long_committee_study import horizon_summary


def raw_rollouts(predictor, truth, pressures, ids, matrix, dt, callback=None):
    state = truth[:, 0].copy()
    histories = [[s.copy()] for s in state]
    active = np.ones(len(ids), bool); events = []; previous = np.full(len(ids), -1)
    statuses = ['completed']*len(ids)
    for step in range(1, truth.shape[1]):
        live = np.flatnonzero(active)
        if not len(live):
            break
        output = predictor.predict(np.c_[state[live], pressures[live]])
        for j, k in enumerate(live):
            value = output[j]
            finite = bool(np.all(np.isfinite(value)))
            huge = bool(finite and (abs(value[0])>1e6 or np.max(np.abs(value[1:]))>1e3))
            accepted = finite and not huge
            selected = int(predictor.last_diagnostics['selected_seed'][j])
            events.append(dict(trajectory=int(ids[k]), step=step, time=step*dt,
                selected_seed=selected, valid_count=int(predictor.last_diagnostics['valid_count'][j]),
                switched=bool(previous[k]>=0 and selected>=0 and previous[k]!=selected),
                negative_species=bool(np.any(value[1:]<0)),
                mass_sum=float(value[1:].sum()), temperature=float(value[0]), accepted=accepted))
            if not accepted:
                active[k] = False
                statuses[k] = 'numeric_magnitude_guard' if huge else 'nonfinite_prediction'
                continue
            state[k] = value; previous[k] = selected; histories[k].append(value.copy())
        if callback and step % 500 == 0:
            callback(step, int(active.sum()))
    rows, species = [], []
    for k, h in enumerate(histories):
        trace = np.array(h); histories[k] = trace
        row, sp = metrics(trace, truth[k], dt, matrix, truth.shape[1]-1)
        y = trace[:, 1:]
        bad = (np.any(y < -1e-12, axis=1) | np.any(y>1+1e-12, axis=1)
               | (trace[:, 0]<200) | (trace[:, 0]>6000)
               | (np.abs(y.sum(axis=1)-1)>.01)
               | (np.max(np.abs((y-truth[k, 0, 1:])@matrix.T), axis=1)>.01))
        failures = np.flatnonzero(bad)
        row.update(trajectory=int(ids[k]), status=statuses[k],
            physically_admissible_complete=bool(row['complete'] and not bad.any()),
            first_physical_violation_seconds=float(failures[0]*dt) if len(failures) else np.nan,
            negative_state_fraction=float(np.mean(np.any(y<0, axis=1))),
            negative_beyond_tolerance_fraction=float(np.mean(np.any(y < -1e-12, axis=1))),
            minimum_mass_fraction=float(y.min()), minimum_temperature=float(trace[:, 0].min()),
            maximum_temperature=float(trace[:, 0].max()),
            mean_mass_sum=float(np.mean(y.sum(axis=1))))
        actual, predicted = np.isfinite(row['reference_tau400']), np.isfinite(row['predicted_tau400'])
        row.update(reference_ignites=bool(actual), prediction_ignites=bool(predicted),
                   ignition_classification_correct=bool(actual==predicted and (predicted or row['complete'])))
        # Raw detected crossings are kept, but do not call them physically valid
        # if the trajectory had already left the declared admissible domain.
        crossing = np.flatnonzero(trace[:,0] >= truth[k,0,0]+400)
        row['ignition_physically_admissible'] = bool(predicted and len(crossing) and (
            not len(failures) or crossing[0] < failures[0]))
        row['physical_tau400_relative_error'] = row['tau400_relative_error'] if row['ignition_physically_admissible'] else np.nan
        rows.append(row)
        species.extend(dict(trajectory=int(ids[k]), **s) for s in sp)
    return rows, species, events, histories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/delta100_raw'))
    args = parser.parse_args(); root = args.root
    cfg = json.loads((root/'config.json').read_text())
    progress = json.loads((root/'training_progress.json').read_text())
    if progress['status'] != 'completed':
        raise RuntimeError('Complete all 100 training members first')
    if (root/'evaluation').exists():
        raise FileExistsError('Preserve completed or interrupted evaluations; use a new output')
    out = root/'evaluation'; out.mkdir(); (out/'methods').mkdir(); (out/'traces').mkdir()
    started = time.perf_counter()
    remaining = min(cfg['max_evaluation_seconds'], max(0., 1800-progress['execution_seconds']))
    deadline = started+remaining
    def budget():
        if time.perf_counter()>deadline:
            raise TimeoutError('Shared training/evaluation CPU 30-minute budget exhausted')
    records = json.loads((root/'members.json').read_text())
    assert len(records)==100
    for r in records:
        assert hashlib.sha256(Path(r['path']+'.keras').read_bytes()).hexdigest()==r['sha256']
    paths = [r['path'] for r in records]; seeds = [r['seed'] for r in records]
    metric = json.loads((root/'distance_metric.json').read_text())
    with np.load(cfg['source']) as data:
        states, pressures, split, dt = data['states'], data['pressures'], data['split'], float(data['dt'])
        matrix, species_names = data['element_matrix'], data['species']
    assert dt==cfg['dt']
    protocol = dict(physics=False, clipping=False, enthalpy_closure=False,
        input='T,P,Y', target='physical delta T and signed-Box-Cox physical delta Y', gamma=.1,
        single_selection='all 90000 validation one-step pairs; existing physical selection_score, seed tie-break',
        test_for_selection=False, posteriori_for_selection=False,
        mean='mean of finite decoded physical candidates, including negative candidates',
        medoid='actual finite decoded candidate minimizing train-scaled summed Euclidean distance',
        finite_negative_candidates_excluded=False,
        numeric_guard=dict(abs_temperature=1e6, abs_species=1e3),
        admissibility_diagnostic_only=dict(negative_Y_tolerance=1e-12,T_range=[200,6000],mass_element_tolerance=.01),
        shared_budget_seconds=1800, remaining_evaluation_budget_seconds=remaining)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    p = DeltaCommittee(paths, seeds, metric, 'mean')
    ids = np.flatnonzero(split=='val'); n = states.shape[1]-1
    x = np.c_[states[ids, :-1].reshape(-1, 10), np.repeat(pressures[ids], n)]
    y = states[ids, 1:].reshape(-1, 10)
    predictions = np.empty((100, len(x), 10))
    transformed_error = np.zeros(100)
    for begin in range(0, len(x), 2048):
        budget(); end = min(begin+2048, len(x)); xx=x[begin:end]
        z = p.forward(p.base.features(xx).astype('float32')).numpy().astype('float64')
        target = p.base.output_norm.forward(p.base.encode_delta(y[begin:end]-xx[:, :10]))
        transformed_error += np.sum((z-target[None])**2, axis=(1,2))
        predictions[:, begin:end] = xx[None, :, :10]+p.base.decode(z)
        if begin % 20480 == 0:
            print('DELTA_APRIORI', end, len(x), flush=True)
    ranking=[]
    for j, seed in enumerate(seeds):
        budget()
        ranking.append(dict(seed=seed, pairs=len(x), transformed_mse=transformed_error[j]/(len(x)*10),
                            **evaluate(y, predictions[j], x, matrix)))
    rank = pd.DataFrame(ranking).sort_values(['selection_score','seed'])
    rank.to_csv(out/'validation_member_ranking.csv', index=False)
    seed = int(rank.iloc[0].seed); index=seeds.index(seed)
    if not np.isfinite(rank.iloc[0].selection_score):
        raise ValueError('No finite validation score')
    frozen=dict(selected_seed=seed, selected_path=paths[index], definitions=['single_best','mean100','medoid100'],
                criterion=protocol['single_selection'], test_for_selection=False, posteriori_for_selection=False,
                physics=False, source_sha256=hashlib.sha256(Path(cfg['source']).read_bytes()).hexdigest())
    (out/'selection_frozen.json').write_text(json.dumps(frozen,indent=2))
    del predictions
    print('DELTA_FROZEN_SINGLE', seed, flush=True)
    single=DeltaCommittee([paths[index]], [seed], metric, 'single')
    summaries, details, species, timing, probes, checks = [], [], [], [], [], []
    for partition in ('val','test'):
        ids=np.flatnonzero(split==partition)
        fine=pd.read_csv(Path('results/committee100_long10ms')/f'reference_ignition_{partition}.csv').set_index('trajectory')
        for name, predictor, rule in [('single_best',single,'single'),('mean100',p,'mean'),('medoid100',p,'medoid')]:
            budget(); predictor.rule=rule; predictor.save(out/'methods'/f'{name}.json')
            x0=np.c_[states[ids,0],pressures[ids]]; first=predictor.predict(x0)
            begin=time.perf_counter()
            def report_step(step, active):
                budget();print('DELTA_ROLLOUT',name,partition,step,active,flush=True)
            rows,sp,events,traces=raw_rollouts(predictor,states[ids],pressures[ids],ids,matrix,dt,report_step)
            timing.append(dict(method=name,split=partition,seconds=time.perf_counter()-begin))
            np.savez_compressed(out/'traces'/f'{name}_{partition}.npz',**{f'trajectory_{int(i)}':t for i,t in zip(ids,traces)})
            pd.DataFrame(events).to_csv(out/f'selection_{name}_{partition}.csv',index=False)
            for b,trace in enumerate(traces):
                if len(trace)>1:
                    np.testing.assert_allclose(first[b],trace[1],atol=1e-8,rtol=1e-10,equal_nan=True)
                else:
                    assert not np.all(np.isfinite(first[b])) or abs(first[b,0])>1e6 or np.max(np.abs(first[b,1:]))>1e3
            for row in rows:
                tau=float(fine.loc[row['trajectory'],'tau400'])
                row['reference_tau400_refined']=tau
                row['tau400_relative_error_refined']=abs(row['predicted_tau400']-tau)/tau if np.isfinite(tau) and row['prediction_ignites'] else np.nan
            summary=horizon_summary(rows)
            summary['median_ignition_error_refined']=float(pd.DataFrame(rows).tau400_relative_error_refined.median())
            common=dict(method=name,split=partition,horizon=.01)
            summaries.append(dict(**common,**summary)); details.extend(dict(**common,**r) for r in rows)
            species.extend(dict(**common,**{**r,'species':str(species_names[r['species']])}) for r in sp)
            j=np.unique(np.r_[np.arange(8),np.linspace(0,n-1,24).astype(int)])
            px=np.c_[states[ids][:,j].reshape(-1,10),np.repeat(pressures[ids],len(j))]
            py=states[ids][:,j+1].reshape(-1,10)
            predicted=predictor.predict(px)
            probes.append(dict(**common,**evaluate(py,predicted,px,matrix)))
            # The actual selected medoid must be a raw candidate; no correction.
            identity=True
            if rule=='medoid':
                chosen=predictor.last_diagnostics['selected_seed'].copy(); c=predictor.candidates(px[:8])
                for b,s in enumerate(chosen[:8]):
                    if s>=0:np.testing.assert_array_equal(predicted[b],c[seeds.index(int(s)),b])
            checks.append(dict(method=name,split=partition,first_step_match=True,raw_candidate_identity=identity))
            for filename,values in [('summary',summaries),('trajectories',details),('species_profiles',species),
                                    ('timing',timing),('one_step_probes',probes),('runtime_checks',checks)]:
                pd.DataFrame(values).to_csv(out/f'{filename}.csv',index=False)
            print('DELTA_RESULT',name,partition,summary,flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(status='completed',physics=False,selected_seed=seed,
        dt=dt,horizon=.01,training_seconds=progress['execution_seconds'],evaluation_seconds=time.perf_counter()-started,
        source_sha256=frozen['source_sha256'],train_pair_sha256=progress['training_pair_sha256'],
        source_hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(Path('kinetics').glob('*.py'))}),indent=2))
    print('DELTA_EVALUATION_COMPLETE',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
