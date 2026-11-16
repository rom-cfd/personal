"""A frozen, validation-selected React-DeepONet autoregressive reproduction.

Reuse existing published training checkpoints: no training, state repair, budget
restart, or test-based model selection. The actual solver consumes only its own
predicted macro endpoints. Additional interior queries are curve diagnostics;
their failures are reported independently of the executed solver boundaries.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/react-stepper-mpl')

from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import shutil
import time
import traceback

import cantera as ct
import numpy as np
import torch

from .react_deeponet_experiment import FifthRootScale, ORIGINALS
from .react_deeponet_v2_model import DetailedReactDeepONet
from .react_deeponet_reconstruction import ReconstructionNet

REP = [0, 1, 2, 4, 6]
REST = [3, 5, 7, 8, 9, 10]
ACTIVE = list(range(9))
BASE = Path('results/react_deeponet_rebuild/run_001')
SOURCES = ['kinetics/react_stepper_audit.py', 'kinetics/react_stepper_reproduction.py',
           'kinetics/react_stepper_report.py', 'tests/test_react_stepper_reproduction.py',
           'kinetics/react_deeponet_experiment.py', 'kinetics/react_deeponet_v2_model.py',
           'kinetics/react_deeponet_diagnostics.py', 'kinetics/react_deeponet_reconstruction.py',
           'kinetics/physics.py', 'kinetics/data.py']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, p)

def csv_write(p, rows):
    with Path(p).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def score(truth, prediction):
    rmse = np.sqrt(np.mean((prediction - truth) ** 2, axis=0))
    denominator = np.sqrt(np.mean(truth ** 2, axis=0))
    ratio = np.divide(rmse, denominator, out=np.full_like(rmse, np.nan), where=denominator > 0)
    return rmse, denominator, ratio

def state_valid(x):
    return np.isfinite(x).all(axis=-1) & (x[..., 0] >= 200) & (x[..., 0] <= 6000) & (x[..., 1:].min(axis=-1) >= 0)

def feedback(operator, reconstruction, scale, initial, phi, stride, points, reference_dt=1e-7):
    """Only macro endpoint predictions determine every later branch input.

    The mathematical diagnostic is retained after a raw-boundary failure, while
    valid_prefix records exactly where the strict executable solver must stop.
    It never supplies reference states or silently clips negative predictions.
    """
    rs, ss = scale.subset(REP), scale.subset(REST)
    z = torch.from_numpy(rs.forward(initial[:, REP]))
    parameters = torch.as_tensor(phi, dtype=torch.float32).reshape(-1, 1)
    n = len(initial)
    trace = np.empty((n, points + 1, 11), dtype=float)
    trace[:, 0] = initial
    raw_boundary_valid = np.ones(n, dtype=bool)
    first_invalid_step = np.full(n, -1, dtype=int)
    representative_boundary_valid = np.ones(n, dtype=bool)
    offsets = torch.arange(1, stride + 1, dtype=torch.float32)[:, None] * reference_dt
    with torch.inference_mode():
        for start in range(0, points, stride):
            repeated = z[:, None, :].expand(n, stride, 5).reshape(-1, 5)
            p = parameters[:, None, :].expand(n, stride, 1).reshape(-1, 1)
            tau = offsets[None, :, :].expand(n, stride, 1).reshape(-1, 1)
            out = operator(repeated, tau, p)
            representative = rs.inverse(out.numpy()).reshape(n, stride, 5)
            other = ss.inverse(reconstruction(out).numpy()).reshape(n, stride, 6)
            block = np.empty((n, stride, 11))
            block[:, :, REP] = representative
            block[:, :, REST] = other
            trace[:, start + 1:start + stride + 1] = block
            good = state_valid(block[:, -1])
            newly_invalid = (first_invalid_step < 0) & ~good
            first_invalid_step[newly_invalid] = start // stride + 1
            raw_boundary_valid &= good
            representative_boundary_valid &= state_valid(representative[:, -1])
            z = out.reshape(n, stride, 5)[:, -1].contiguous()
    return trace, dict(raw_boundary_valid=raw_boundary_valid,
        representative_boundary_valid=representative_boundary_valid,
        first_invalid_step=first_invalid_step)

def case_definitions(fixture=False):
    rng = np.random.default_rng(20261115 if fixture else 20261116)
    count = 2 if fixture else 16
    cases = []
    # Existing ICs are rejected before any test reference or model prediction.
    known = json.loads((BASE / 'cases.json').read_text())
    prior = Path('results/react_refinement/run_001/cases.json')
    if prior.exists():
        other = json.loads(prior.read_text())
        if isinstance(other, list): known.extend(other)
    fixture_cases = Path('/workspace/react_stepper_followup/run_001/software_fixture_001/cases_frozen.json')
    if not fixture and fixture_cases.exists():
        known.extend(json.loads(fixture_cases.read_text()))
    used = {(round(c['T0'], 8), round(c['phi'], 8)) for c in known if 'T0' in c and 'phi' in c}
    for split in ['interpolation', 'extrapolation']:
        for i in range(count):
            while True:
                if split == 'interpolation':
                    temp = float(rng.uniform(1005, 1195)); phi = float(rng.uniform(.76, 1.14))
                else:
                    temp = float(rng.uniform(980, 1240)); phi = float(rng.uniform(.6, 1.3))
                    if 1000 <= temp <= 1200 and .75 <= phi <= 1.15: continue
                key = (round(temp, 8), round(phi, 8))
                if key not in used: break
            used.add(key)
            cases.append(dict(IC=len(cases), split=split, T0=temp, phi=phi))
    return cases

def prepare(root, fixture=False):
    if root.exists(): raise FileExistsError(root)
    started = time.perf_counter(); root.mkdir(parents=True)
    for directory in ['models', 'source_snapshot/kinetics', 'source_snapshot/tests', 'data', 'evaluations', 'report']:
        (root / directory).mkdir(parents=True, exist_ok=True)
    assert all(sha(p) == s for p,s in ORIGINALS.items())
    audit = Path('results/react_stepper_audit/run_001')
    with (audit / 'boundary_diagnosis.csv').open() as f:
        validation = list(csv.DictReader(f))
    # all_steps is a different objective, so it cannot win paper-model selection.
    eligible = ['pretrain_72500', 'pretrain_100000', 'endpoint_selected']
    ranks = []
    for name in eligible:
        rows = [r for r in validation if r['model'] == name and abs(float(r['dt_s']) - 20e-6) < 1e-12]
        ranks.append(dict(model=name, full_raw_joint9=sum(r['full_joint9'] == 'True' and r['full_boundary_valid'] == 'True' for r in rows),
            full_early_raw_joint9=sum(r['full_and_early_joint9'] == 'True' and r['full_boundary_valid'] == 'True' for r in rows),
            median_worst9=float(np.median([float(r['worst9']) for r in rows])), conditions=len(rows)))
    winner = min(ranks, key=lambda r:(-r['full_raw_joint9'], -r['full_early_raw_joint9'], r['median_worst9']))
    paths = {
        'pretrain_72500': BASE / 'models/pretrain_072500.pt',
        'pretrain_100000': BASE / 'models/pretrain_100000.pt',
        'endpoint_selected': Path('results/react_refinement/run_001/models/endpoint_lr_div100/selected.pt')}
    models = dict(selected=paths[winner['model']], protocol_final=BASE/'models/operator_protocol_final.pt',
        stabilized=paths['endpoint_selected'])
    for name,p in models.items(): shutil.copyfile(p, root/'models'/f'{name}.pt')
    shutil.copyfile(BASE/'models/reconstruction_selected.pt', root/'models/reconstruction.pt')
    for source, dest in [('preprocessing.json', 'preprocessing.json'), ('mechanism.yaml', 'mechanism.yaml')]:
        shutil.copyfile(BASE/source, root/dest)
    shutil.copyfile(audit/'analysis.json', root/'prior_validation_audit.json')
    shutil.copyfile(audit/'boundary_diagnosis.csv', root/'prior_validation.csv')
    manifest = {}
    for name in SOURCES:
        dest = root/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, dest); manifest[name] = sha(dest)
    (root/'source_snapshot/kinetics/__init__.py').write_text('"""Frozen feedback solver verification."""\n')
    cases = case_definitions(fixture)
    write(root/'cases_frozen.json', cases)
    write(root/'selection_frozen.json', dict(status='frozen_before_test_reference_or_predictions',
        at=datetime.now(timezone.utc).isoformat(), selected_source=winner['model'], ranks=ranks,
        criterion='20us executed boundaries: valid full9 passes, then early+full9, then median worst9',
        validation_only=True, historical_test_not_used=True,
        models={name:dict(source=str(p), sha256=sha(p)) for name,p in models.items()},
        note='Selected checkpoint is pretraining or stabilized endpoint training, not necessarily completed K5 final weights. Literal original protocol-final weights are a separate mandatory comparator.'))
    write(root/'config.json', dict(seed=11, reference_dt_s=1e-7, horizon_s=.001,
        evaluated_steps_s=[20e-6, 2e-6], primary_step_s=20e-6, fixture=fixture, threads=2,
        budget_seconds=900, training_updates=0, no_clipping=True, no_physics_repair=True,
        strict_solver='Stop at first inadmissible actual macro endpoint. Interior dense queries do not feed back.',
        dense_diagnostics='Unconstrained dense curves and their physical validity reported separately; never count invalid raw prefixes as full-horizon successes.',
        reproduction_source='Kumar & Echekki, Fuel365(2024)131212; Kumar thesis chapter3',
        fidelity_limits=['Public Burke reduction has25 Cantera entries, not the exact private20 reaction file.',
            'Three-way ParamNet fusion is an unverified reconstruction choice.',
            'This reuses published independent PyTorch training weights; not author weights.',
            'No claim of exact published MAE from our physical-unit NRMSE.']))
    write(root/'source_manifest.json', manifest)
    write(root/'prepared.json', dict(status='passed', at=datetime.now(timezone.utc).isoformat(),
        cases_sha256=sha(root/'cases_frozen.json'), selection_sha256=sha(root/'selection_frozen.json'),
        config_sha256=sha(root/'config.json'), source_manifest_sha256=sha(root/'source_manifest.json'),
        prep_seconds=time.perf_counter()-started, test_reference_not_generated=True,
        static_input_sha256={str(p.relative_to(root)):sha(p) for p in [root/'preprocessing.json',root/'mechanism.yaml',*sorted((root/'models').glob('*.pt'))]},
        original_data_sha256=ORIGINALS))
    write(root/'status.json',dict(phase='prepared', training=False))
    print(json.dumps(dict(phase='prepared', root=str(root), selected=winner['model'], cases=len(cases), new_training=False)))

def summarize(root, cases, truth, predictions, variables):
    rows, metrics = [], []
    points = truth.shape[1] - 1
    scaler=json.loads((root/'preprocessing.json').read_text())
    minimum=np.asarray(scaler['minimum'])[REP]
    span=np.asarray(scaler['maximum'])[REP]-minimum
    for (model, stride), (pred, info) in predictions.items():
        for i,case in enumerate(cases):
            raw_complete = bool(info['raw_boundary_valid'][i])
            dense_physical = bool(state_valid(pred[i]).all())
            case_results = {}
            for grid in ['executed_boundaries','dense_queries']:
                indices = np.arange(0, points + 1, stride if grid == 'executed_boundaries' else 1)
                for window in ['full_1ms','early_0p1ms']:
                    selected = indices if window == 'full_1ms' else indices[indices <= 1000]
                    rmse,den,nrmse = score(truth[i, selected], pred[i, selected])
                    curve5 = bool(np.all(nrmse[REP] <= .1)); curve9 = bool(np.all(nrmse[ACTIVE] <= .1))
                    case_results[(grid,window)] = (curve5, curve9)
                    for j,var in enumerate(variables):
                        metrics.append(dict(model=model, step_us=stride/10, IC=case['IC'], split=case['split'],
                            grid=grid, window=window, variable=var, diagnostic_RMSE=float(rmse[j]),
                            RMS_Cantera=float(den[j]), diagnostic_NRMSE=float(nrmse[j]) if np.isfinite(nrmse[j]) else '',
                            raw_complete=raw_complete, raw_NRMSE=float(nrmse[j]) if raw_complete and np.isfinite(nrmse[j]) else '',
                            dense_physical_valid=dense_physical,
                            note='Invalid/incomplete solver has no raw full-horizon score. Diagnostic curve is explicitly separate.'))
            full = case_results[('executed_boundaries','full_1ms')]
            early = case_results[('executed_boundaries','early_0p1ms')]
            dense_full = case_results[('dense_queries','full_1ms')]
            dense_early = case_results[('dense_queries','early_0p1ms')]
            norm_truth=2*(truth[i][:,REP]**.2-minimum)/span-1
            physical_rep=pred[i][:,REP]
            norm_pred=2*(np.sign(physical_rep)*np.abs(physical_rep)**.2-minimum)/span-1
            normalized_mae=float(np.mean(np.abs(norm_pred[::stride]-norm_truth[::stride])))
            rows.append(dict(model=model,step_us=stride/10,**case, finite=bool(np.isfinite(pred[i]).all()),
                raw_complete=raw_complete, first_invalid_step=int(info['first_invalid_step'][i]),
                raw_joint5=bool(raw_complete and full[0]), raw_joint9=bool(raw_complete and full[1]),
                raw_full_early_joint9=bool(raw_complete and full[1] and early[1]),
                raw_dense_curve_full_early_joint9=bool(raw_complete and dense_full[1] and dense_early[1]),
                dense_curve_joint9=bool(dense_full[1]), dense_physical_valid=dense_physical,
                dense_curve_and_physical_joint9=bool(dense_full[1] and dense_physical),
                raw_mean_abs_normalized5=normalized_mae if raw_complete else '',
                diagnostic_mean_abs_normalized5=normalized_mae,
                max_boundary_sumY_error=float(np.max(np.abs(pred[i,::stride,1:].sum(axis=1)-1))),
                max_dense_sumY_error=float(np.max(np.abs(pred[i,:,1:].sum(axis=1)-1))),
                minimum_boundary_mass_fraction=float(pred[i, ::stride, 1:].min()),
                minimum_dense_mass_fraction=float(pred[i, :, 1:].min())))
    csv_write(root/'evaluations/cases.csv', rows); csv_write(root/'evaluations/NRMSE.csv', metrics)
    summary=[]
    for model in ['selected','protocol_final','stabilized']:
        for step in [20,2]:
            for split in ['interpolation','extrapolation']:
                subset=[r for r in rows if r['model']==model and r['step_us']==step and r['split']==split]
                summary.append(dict(model=model,step_us=step,split=split,total=len(subset),
                    **{k:sum(r[k] for r in subset) for k in ['raw_complete','raw_joint5','raw_joint9','raw_full_early_joint9',
                        'raw_dense_curve_full_early_joint9','dense_curve_joint9','dense_physical_valid','dense_curve_and_physical_joint9']}))
    csv_write(root/'report/summary.csv',summary)
    return rows,metrics,summary

def run(root):
    started=time.perf_counter();cfg=json.loads((root/'config.json').read_text())
    prepared=json.loads((root/'prepared.json').read_text())
    if (root/'worker_started.json').exists(): raise RuntimeError('Already executed; do not restart')
    for name,key in [('cases_frozen.json','cases_sha256'),('selection_frozen.json','selection_sha256'),
                     ('config.json','config_sha256'),('source_manifest.json','source_manifest_sha256')]:
        assert sha(root/name)==prepared[key],name
    for name,expected in json.loads((root/'source_manifest.json').read_text()).items():
        assert sha(root/'source_snapshot'/name)==expected,name
    for name,expected in prepared['static_input_sha256'].items():
        assert sha(root/name)==expected,name
    assert all(sha(p)==s for p,s in ORIGINALS.items())
    torch.set_num_threads(2);deadline=started+cfg['budget_seconds']-prepared['prep_seconds']
    write(root/'worker_started.json',dict(pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),training=False))
    cases=json.loads((root/'cases_frozen.json').read_text());n=10000
    reference=[];variables=None
    for case in cases:
        if time.perf_counter()>=deadline: raise TimeoutError('Reference budget exhausted')
        gas=ct.Solution(str(root/'mechanism.yaml'));gas.TP=case['T0'],ct.one_atm
        gas.set_equivalence_ratio(case['phi'],'H2','O2:1,N2:3.76')
        variables=['T']+gas.species_names
        reactor=ct.IdealGasConstPressureReactor(gas,clone=False);network=ct.ReactorNet([reactor])
        network.rtol=1e-10;network.atol=1e-20;network.max_time_step=1e-7
        trace=np.empty((n+1,11));trace[0]=np.r_[gas.T,gas.Y]
        for k in range(1,n+1):
            network.advance(k*1e-7);trace[k]=np.r_[gas.T,gas.Y]
        assert np.min(trace[:,1:])>=-1e-14
        trace[:,1:]=np.maximum(trace[:,1:],0) # Cantera reference roundoff only; never NN output.
        reference.append(trace)
        write(root/'status.json',dict(phase='reference',IC=case['IC'],total=len(cases),training=False))
    truth=np.asarray(reference);np.savez_compressed(root/'data/reference.npz',states=truth)
    prep=json.loads((root/'preprocessing.json').read_text());scale=FifthRootScale(prep['minimum'],prep['maximum'])
    reconstruction=ReconstructionNet()
    reconstruction.load_state_dict(torch.load(root/'models/reconstruction.pt',weights_only=False)['model']);reconstruction.eval()
    predictions={}
    selected=json.loads((root/'selection_frozen.json').read_text())
    for name in ['selected','protocol_final','stabilized']:
        assert sha(root/'models'/f'{name}.pt')==selected['models'][name]['sha256']
        model=DetailedReactDeepONet();model.load_state_dict(torch.load(root/'models'/f'{name}.pt',weights_only=False)['model']);model.eval()
        for stride in [200,20]:
            if time.perf_counter()>=deadline: raise TimeoutError('Evaluation budget exhausted')
            write(root/'status.json',dict(phase='evaluation',model=name,step_us=stride/10,training=False))
            pred,info=feedback(model,reconstruction,scale,truth[:,0],[c['phi'] for c in cases],stride,n)
            predictions[(name,stride)]=(pred,info)
            np.savez_compressed(root/'evaluations'/f'{name}_{stride}.npz',prediction=pred,**info)
    rows,metrics,summary=summarize(root,cases,truth,predictions,variables)
    write(root/'status.json',dict(phase='reporting',training=False))
    from .react_stepper_report import report,verify
    report(root,cases,truth,predictions,variables,summary,deadline)
    verification=verify(root,truth,predictions,variables,metrics)
    write(root/'verification.json',verification)
    assert verification['status']=='passed'
    assert all(sha(p)==s for p,s in ORIGINALS.items())
    outputs={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and p.name not in ['status.json','worker_started.json','complete.json','exit.json']}
    write(root/'complete.json',dict(status='passed',at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
        training_updates=0,selection_sha256=sha(root/'selection_frozen.json'),verification_sha256=sha(root/'verification.json'),
        report_complete_sha256=sha(root/'report/complete.json'),output_sha256=outputs))
    write(root/'status.json',dict(phase='complete',status='passed',training=False))
    write(root/'exit.json',dict(returncode=0,complete_sha256=sha(root/'complete.json')))
    print(json.dumps(dict(verified_complete=True,elapsed_seconds=time.perf_counter()-started,summary=summary)))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--fixture',action='store_true');a=p.parse_args()
    try:
        prepare(a.root,a.fixture) if a.action=='prepare' else run(a.root)
    except Exception:
        if a.root.exists():
            write(a.root/'failure.json',dict(traceback=traceback.format_exc()))
            write(a.root/'exit.json',dict(returncode=1))
        raise

if __name__=='__main__': main()
