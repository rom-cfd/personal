"""After the sequential pipeline: audit plateau checks, timing and T/P 10k replay.

This reads saved artifacts only. It performs no training, inference or selection.
Use a new output directory; original reports and checkpoints are never rewritten.
"""
import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import pandas as pd
from kinetics.deeponet_long_training import check_plateau,new_monitor
from kinetics.p_only_training import sha256,write_json


def weight_arrays(path):
    arrays={}
    with zipfile.ZipFile(path) as archive:
        with archive.open('model.weights.h5') as stream,h5py.File(stream) as weights:
            def read(name,obj):
                if isinstance(obj,h5py.Dataset) and not name.startswith('optimizer/'):
                    arrays[name]=obj[()]
            weights.visititems(read)
    return arrays


def audit(root,output):
    if output.exists():raise FileExistsError('Preserve existing audit; choose a new output')
    run=json.loads((root/'run.json').read_text());records=json.loads((root/'final_members.json').read_text())
    events=[json.loads(line) for line in (root/'execution.jsonl').read_text().splitlines()]
    checks,times,replay=[],[],[]
    for record in records:
        if record is None:continue
        checkpoint=Path(record['checkpoint']);marker=json.loads((checkpoint/'complete.json').read_text())
        for name,expected in marker['output_sha256'].items():
            assert Path(name).name==name and sha256(checkpoint/name)==expected
        state=json.loads((checkpoint/'state.json').read_text())
        with np.load(checkpoint/'adam.npz',allow_pickle=False) as adam:
            names=[v['name'] for v in state['optimizer_variables']]
            index=names.index('iteration')
            assert int(adam[f'v{index}'])==record['updates']==state['optimizer_iteration']==state['update']
            assert all(np.isfinite(adam[name]).all() for name in adam.files)
        monitor=new_monitor(run['config']['plateau'])
        for old in record['plateau_checks']:
            check_plateau(state['full_train_history'],old['update'],monitor)
            assert monitor['checks'][-1]==old,'Saved/recomputed plateau check differs'
            row=dict(family=record['family'],seed=record['seed'],update=old['update'],passed=old['passed'],
                old_window_count=old['window_counts'][0],new_window_count=old['window_counts'][1])
            for name,values in [('old_quantiles',old['old_quantiles']),('new_quantiles',old['new_quantiles']),
                                 ('relative_changes',old['relative_changes'])]:
                for i,label in enumerate(('q25','median','q75')):
                    row[f'{name}_{label}']=values[i] if values else None
            checks.append(row)
        if record['reason']=='plateau':assert monitor['plateau']
        active=sum(e['active_seconds'] for e in events if e['phase']=='member_finished'
                   and e['family']==record['family'] and e['seed']==record['seed'])
        new_updates=record['updates']-record['origin_update']
        components=record['timing']
        times.append(dict(family=record['family'],seed=record['seed'],reason=record['reason'],
            final_updates=record['updates'],new_updates=new_updates,active_seconds_all_invocations=active,
            baseline_origin_seconds=record['origin_seconds'],
            optimizer_seconds_per_new_update=components['optimizer_seconds']/new_updates if new_updates else None,
            load_terminal_save_and_other_seconds=active-sum(components.values()),**components))
    for record in run['original_tp']:
        seed=record['seed'];paths=sorted((root/'checkpoints/tp'/f'seed{seed}').glob('update_0010000_*/model.keras'))
        if not paths:continue
        original=Path(record['path']+'.keras');assert sha256(original)==record['sha256']
        new=paths[0];marker=json.loads((new.parent/'complete.json').read_text())
        assert sha256(new)==marker['output_sha256']['model.keras']
        a,b=weight_arrays(original),weight_arrays(new)
        same_layout=set(a)==set(b) and all(a[k].shape==b[k].shape for k in a)
        exact=same_layout and all(np.array_equal(a[k],b[k]) for k in a)
        maximum=max(float(np.max(np.abs(a[k]-b[k]))) for k in a) if same_layout else None
        replay.append(dict(seed=seed,old_path=str(original),new_path=str(new),
            same_layout=same_layout,all_weight_arrays_identical=exact,max_weight_abs_difference=maximum))
    workload=[]
    for parent in (root/'evaluations').glob('*'):
        markers=sorted(parent.glob('*/complete.json'))
        if not markers:continue
        marker=markers[-1];info=json.loads(marker.read_text());directory=marker.parent
        events_frame=pd.read_csv(directory/'events.csv');timing=json.loads((directory/'timing.json').read_text())
        one=json.loads((directory/'one_step.json').read_text())
        trajectories=pd.read_csv(directory/'trajectories.csv')
        endpoint=trajectories[np.isclose(trajectories.horizon,.01)]
        actual,predicted=endpoint.reference_ignites,endpoint.prediction_ignites
        workload.append(dict(method=info['method'],mode=info['mode'],split=info['split'],
            full_horizon_completed=int(endpoint.complete.sum()),
            physically_admissible_full_horizon=int(endpoint.physically_admissible_complete.sum()),
            true_positive=int((actual&predicted).sum()),
            false_negative=int((actual&~predicted&endpoint.complete).sum()),
            false_positive=int((~actual&predicted).sum()),
            true_negative=int((~actual&~predicted&endpoint.complete).sum()),
            unresolved_before_ignition=int(endpoint.unresolved_before_ignition.sum()),
            rollout_predict_calls=int(events_frame.step.max()),rollout_evaluated_IC_rows=len(events_frame),
            rollout_candidate_evaluations=len(events_frame)*info['member_count'],
            dense_one_step_pairs=one['pair_count'],
            **{key:timing[key] for key in ('warmup_seconds','rollout_seconds','diagnostics_seconds','one_step_seconds')},
            **{f'rollout_{k}_seconds':v for k,v in timing['rollout_components'].items()},
            **{f'one_step_{k}_seconds':v for k,v in timing['one_step_components'].items()}))
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in [('plateau_checks',checks),('training_time_all_invocations',times),
                      ('tp_10k_weight_replay',replay),('evaluation_workload_and_time',workload)]:
        pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
    result=dict(verified_endpoints=sum(r is not None for r in records),
        saved_plateau_checks_recomputed=len(checks),
        tp_10k_replay_count=len(replay),tp_10k_exact_weight_replays=sum(r['all_weight_arrays_identical'] for r in replay),
        evaluation_arms=len(workload),script_sha256=sha256(__file__),
        notes=['Weight equality verifies a new T/P replay; it does not recover historical Adam files.',
               'neural_decode is an aggregate; features, neural and decode are its subcomponents. Do not sum both.',
               'Failed rollout arms may evaluate fewer rows/calls. Compare workloads with wall time.',
               'Training active time sums all invocations; checkpoint/initialization overhead is included.',
               'ANN historical grouped training is not a matched speed benchmark.'])
    write_json(output/'verification.json',result)
    write_json(output/'complete.json',dict(output_sha256={p.name:sha256(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/deeponet_long_study/run_001'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.root,args.output)
