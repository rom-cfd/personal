"""Immutable preparation and sequential training for the React-DeepONet rebuild."""

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/react-deeponet-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/react-deeponet-cache')

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import traceback

import cantera as ct
import numpy as np
import torch

from .react_deeponet_reconstruction import ReactDeepONet, ReconstructionNet, REPRESENTATIVE
from .physics import project_species


ORIGINALS = {
    'results/long10ms_data/trajectories.npz': '8d41f19f2145e12f9e89ee3eaa88fb0b1b93a84dcaaee65dcaa2e8b264077563',
    'results/long10ms_data/training_pairs.npz': '44db17b16e167ea389631dacb8015528f76cbd5c2072e9fb919701e54d50fa3e',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def scores(reference, prediction):
    den = np.linalg.norm(reference, axis=0)
    return np.divide(np.linalg.norm(prediction-reference, axis=0), den,
                     out=np.full_like(den, np.nan, dtype=float), where=den > 0)


class FifthRootScale:
    def __init__(self, minimum, maximum):
        self.minimum = np.asarray(minimum)
        self.maximum = np.asarray(maximum)
        self.constant = self.maximum == self.minimum
        self.span = np.where(self.constant, 1., self.maximum-self.minimum)

    @classmethod
    def fit(cls, training):
        if np.any(training < 0):
            raise ValueError('Fifth-root fitting expects nonnegative training states')
        transformed = training**.2
        return cls(np.min(transformed, axis=(0, 1)), np.max(transformed, axis=(0, 1)))

    def forward(self, values):
        # Physical input should be nonnegative. Do not hide an invalid raw state.
        if np.any(values < 0):
            raise ValueError('Negative physical state at the fifth-root encoder')
        return (2*(np.asarray(values)**.2-self.minimum)/self.span-1).astype(np.float32)

    def inverse(self, values):
        root = (np.asarray(values, dtype=float)+1)*.5*self.span+self.minimum
        root[..., self.constant] = self.minimum[self.constant]
        # Fifth power is signed: raw negative species are not silently clipped.
        return root**5

    def subset(self, indices):
        return FifthRootScale(self.minimum[indices], self.maximum[indices])

    def json(self):
        return dict(minimum=self.minimum.tolist(), maximum=self.maximum.tolist(),
                    constant=self.constant.tolist(), fitted_on='training trajectories only')


def case_definitions(cfg):
    reported = cfg['reported_settings']
    choice = cfg['declared_reconstruction_choices']
    cases = []
    grids = [
        ('train', reported['generated_T0_K'], choice['train_phi']),
        ('val', choice['validation_T0_K'], choice['validation_phi']),
        ('test_interpolation', choice['test_interpolation_T0_K'], choice['test_interpolation_phi']),
        ('test_extrapolation', reported['generated_T0_K'], choice['test_extrapolation_phi']),
    ]
    used = set()
    for split, temperatures, phis in grids:
        for temperature in temperatures:
            for phi in phis:
                key = (float(temperature), float(phi))
                if key in used:
                    raise ValueError('Trajectory split overlap')
                used.add(key)
                cases.append(dict(case_id=len(cases), split=split, T0=float(temperature), phi=float(phi),
                                  P_atm=1., seen_training_curve=split == 'train'))
    return cases


def publication_gate(cfg):
    p = Path(cfg['execution']['publication_record'])
    record = json.loads(p.read_text())
    assert record['status'] == 'passed' and record['remote_verified'] is True
    assert record['branch'] == 'bw'
    assert record['commit_sha'] == cfg['execution']['required_publication_sha']
    assert record['remote_sha'] == record['commit_sha']
    return dict(record=record, record_sha256=sha(p))


def prepare(root, cfg_path):
    if root.exists():
        raise FileExistsError('Use a new root; preparation never overwrites old results')
    cfg = json.loads(cfg_path.read_text())
    gate = publication_gate(cfg)
    assert {p: sha(p) for p in ORIGINALS} == ORIGINALS
    root.mkdir(parents=True)
    write(root/'config.json', cfg)
    write(root/'publication_gate.json', gate)
    for name in ['models', 'evaluations', 'report', 'source_snapshot/kinetics']:
        (root/name).mkdir(parents=True, exist_ok=True)
    source_files = ['kinetics/react_deeponet_reconstruction.py', 'kinetics/react_deeponet_experiment.py',
                    'kinetics/react_deeponet_report.py', 'kinetics/physics.py', 'kinetics/data.py']
    fingerprints = {}
    for name in source_files:
        source = Path(name)
        target = root/'source_snapshot'/name
        shutil.copyfile(source, target)
        fingerprints[name] = sha(source)
    (root/'source_snapshot/kinetics/__init__.py').write_text('"""Frozen experiment source."""\n')
    (root/'source_snapshot/run.py').write_text('from kinetics.react_deeponet_experiment import main\nmain()\n')
    write(root/'source_manifest.json', fingerprints)
    start = time.perf_counter()
    cases = case_definitions(cfg)
    dt = cfg['declared_reconstruction_choices']['sample_dt_s']
    steps = int(round(cfg['reported_settings']['evaluation_horizon_s']/dt))
    gas = ct.Solution('h2o2.yaml')
    mechanism_path = next(Path(d)/'h2o2.yaml' for d in ct.get_data_directories() if (Path(d)/'h2o2.yaml').exists())
    shutil.copyfile(mechanism_path, root/'mechanism.yaml')
    variables = ['T']+gas.species_names
    traces = []
    clipped_roundoff = 0
    for case in cases:
        if time.perf_counter()-start > 180:
            raise TimeoutError('Data preparation exceeded its 180-second guard')
        gas = ct.Solution(str(root/'mechanism.yaml'))
        gas.TP = case['T0'], ct.one_atm
        gas.set_equivalence_ratio(case['phi'], 'H2', 'O2:1,N2:3.76')
        reactor = ct.IdealGasConstPressureReactor(gas, clone=False)
        net = ct.ReactorNet([reactor])
        net.rtol = 1e-10
        net.atol = 1e-20
        net.max_time_step = dt
        states = [np.r_[gas.T, gas.Y]]
        for j in range(steps):
            net.advance((j+1)*dt)
            states.append(np.r_[gas.T, gas.Y])
        states = np.asarray(states)
        if np.min(states[:, 1:]) < -1e-14:
            raise ValueError('Reference has a negative species beyond roundoff')
        clipped_roundoff += int(np.sum(states[:, 1:] < 0))
        states[:, 1:] = np.maximum(states[:, 1:], 0)
        traces.append(states)
        write(root/'status.json', dict(phase='preparing_data', completed_cases=len(traces), total_cases=len(cases)))
    traces = np.asarray(traces)
    ids = np.array([c['case_id'] for c in cases if c['split'] == 'train'])
    scaler = FifthRootScale.fit(traces[ids])
    np.savez_compressed(root/'trajectories.npz', states=traces, times=np.arange(steps+1)*dt)
    write(root/'preprocessing.json', scaler.json())
    write(root/'cases.json', cases)
    write(root/'dataset.json', dict(variables=variables, species=gas.species_names,
        n_reactions=gas.n_reactions, dt=dt, horizon_s=steps*dt,
        shape=traces.shape, split_counts={s: sum(c['split'] == s for c in cases) for s in {c['split'] for c in cases}},
        selected_starts_per_training_trajectory=steps+1-int(round(cfg['reported_settings']['window_s']/dt)),
        clipped_negative_reference_roundoff=clipped_roundoff, SHA256=sha(root/'trajectories.npz'),
        mechanism_sha256=sha(root/'mechanism.yaml'), cantera_version=ct.__version__, prepare_seconds=time.perf_counter()-start))
    originals = {p: sha(p) for p in ORIGINALS}
    assert originals == ORIGINALS
    write(root/'prepared.json', dict(status='passed', original_data_sha256=originals,
        source_sha256=fingerprints, config_sha256=sha(root/'config.json'),
        dataset_sha256=sha(root/'trajectories.npz'), cases_sha256=sha(root/'cases.json'),
        preprocessing_sha256=sha(root/'preprocessing.json'), prepared_at=datetime.now(timezone.utc).isoformat()))
    write(root/'status.json', dict(phase='prepared', training_started=False))
    print('PREPARED', json.dumps(json.loads((root/'dataset.json').read_text())), flush=True)


def sample_windows(states, batch, maximum_offset, dt, generator):
    if states.ndim != 3 or states.shape[1] <= maximum_offset:
        raise ValueError('Insufficient within-trajectory window data')
    ic = torch.randint(states.shape[0], (batch,), generator=generator)
    start = torch.randint(states.shape[1]-maximum_offset, (batch,), generator=generator)
    length = torch.randint(1, maximum_offset+1, (batch,), generator=generator)
    return states[ic, start], length[:, None].float()*dt, states[ic, start+length]


def valid_state(state):
    return bool(np.isfinite(state).all() and 200 <= state[0] <= 6000 and np.min(state[1:]) >= 0 and np.sum(state[1:]) > 0)


def representative_rollout(model, scale, initial, steps, tau):
    """Macro-boundary validation: feed predicted endpoints, never reference states."""
    trace = [initial.copy()]
    with torch.inference_mode():
        for _ in range(steps):
            encoded = torch.from_numpy(scale.forward(trace[-1][None, :]))
            z = model(encoded, torch.tensor([[tau]], dtype=torch.float32)).numpy()
            state = scale.inverse(z)[0]
            if not valid_state(state):
                break
            trace.append(state)
    return np.asarray(trace)


def validation(model, physical, scale, maximum_offset, dt, fixed, indices):
    model.eval()
    with torch.inference_mode():
        vx, vt, vy = fixed
        mse = float(torch.mean((model(vx, vt)-vy)**2))
    if not np.isfinite(mse):
        raise FloatingPointError('Nonfinite fixed-window validation loss')
    complete, passed, values = 0, 0, []
    for truth in physical:
        ref = truth[::maximum_offset][:, indices]
        pred = representative_rollout(model, scale, ref[0], len(ref)-1, maximum_offset*dt)
        if len(pred) != len(ref):
            continue
        complete += 1
        error = float(np.max(scores(ref, pred)))
        values.append(error)
        passed += int(error <= .1)
    median = float(np.median(values)) if values else float('inf')
    rank = (-complete, -passed, median, mse)
    model.train()
    return rank, dict(window_MSE=mse, raw_rep_complete=complete, raw_rep_joint5=passed,
                      median_worst5=median if values else None, total=len(physical))


def checkpoint(path, model, optimizer, generator, update, active_seconds, **extra):
    if path.exists():
        raise FileExistsError('Refuse to overwrite a registered checkpoint')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    generator_state=generator.get_state(), torch_rng=torch.get_rng_state(),
                    update=update, active_seconds=active_seconds, **extra), path)


def train_reconstruction(root, cfg, states, val, all_scale, indices, remainder, deadline):
    torch.manual_seed(cfg['execution']['seed']+1000)
    model = ReconstructionNet(len(remainder))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(cfg['execution']['seed']+1001)
    data = torch.from_numpy(all_scale.forward(states).reshape(-1, states.shape[-1]))
    validation_states = torch.from_numpy(all_scale.forward(val).reshape(-1, states.shape[-1]))
    history, best, best_path, active = [], float('inf'), None, 0.
    start = time.perf_counter()
    for update in range(1, 20001):
        if time.perf_counter() >= deadline:
            raise TimeoutError('Overall budget exhausted before reconstruction training completed')
        step_start = time.perf_counter()
        selected = torch.randint(len(data), (4096,), generator=generator)
        batch = data[selected]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(batch[:, indices])-batch[:, remainder])**2)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite reconstruction loss')
        loss.backward()
        optimizer.step()
        active += time.perf_counter()-step_start
        if update == 1 or update % 100 == 0:
            history.append(dict(stage='reconstruction', update=update, loss=float(loss.detach()),
                                lr=.001, active_seconds=active, wall_seconds=time.perf_counter()-start))
        if update == 1 or update % 1000 == 0:
            model.eval()
            with torch.inference_mode():
                metric = float(torch.mean((model(validation_states[:, indices])-validation_states[:, remainder])**2))
            if not np.isfinite(metric):
                raise FloatingPointError('Nonfinite reconstruction validation loss')
            path = root/'models'/f'reconstruction_{update:06d}.pt'
            checkpoint(path, model, optimizer, generator, update, active, validation_MSE=metric)
            if metric < best:
                best, best_path = metric, path
            model.train()
            csv_write(root/'reconstruction_training.csv', history)
            write(root/'status.json', dict(phase='training_reconstruction', update=update, target_updates=20000,
                active_seconds=active, selected_validation_MSE=best, elapsed_seconds=time.perf_counter()-start))
    model.load_state_dict(torch.load(best_path, weights_only=False)['model'])
    model.eval()
    shutil.copyfile(best_path, root/'models/reconstruction_selected.pt')
    write(root/'reconstruction_selection.json', dict(selected_checkpoint=str(best_path.relative_to(root)),
        validation_MSE=best, updates=20000, active_seconds=active, wall_seconds=time.perf_counter()-start,
        stop_reason='requested_updates_completed', learning_rate=.001, architecture=[5,64,64,len(remainder)]))
    return model


def train_operator(root, cfg, states, val, scale, indices, deadline):
    execution, reported = cfg['execution'], cfg['reported_settings']
    dt = cfg['declared_reconstruction_choices']['sample_dt_s']
    maximum_offset = int(round(reported['window_s']/dt))
    torch.manual_seed(execution['seed'])
    model = ReactDeepONet(dt, reported['window_s'])
    optimizer = torch.optim.Adam(model.parameters(), lr=reported['learning_rate'])
    generator = torch.Generator().manual_seed(execution['seed']+7)
    val_generator = torch.Generator().manual_seed(execution['seed']+8)
    training = torch.from_numpy(scale.forward(states[:, :, indices]))
    validation_data = torch.from_numpy(scale.forward(val[:, :, indices]))
    fixed = sample_windows(validation_data, 8192, maximum_offset, dt, val_generator)
    best_rank, selected = None, None
    history, selections = [], []
    active = 0.
    start = time.perf_counter()
    reason = 'requested_updates_completed'
    update = 0
    last_saved_update = None
    for update in range(1, reported['training_updates']+1):
        if time.perf_counter() >= deadline:
            update -= 1
            reason = 'overall_time_cap'
            break
        step_start = time.perf_counter()
        x, tau, y = sample_windows(training, reported['batch_size'], maximum_offset, dt, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x, tau)-y)**2)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite operator loss at update {update}')
        loss.backward()
        optimizer.step()
        active += time.perf_counter()-step_start
        if update == 1 or update % execution['history_every'] == 0:
            history.append(dict(stage='operator', update=update, loss=float(loss.detach()),
                lr=reported['learning_rate'], active_seconds=active, wall_seconds=time.perf_counter()-start))
        if update == 1 or update % execution['checkpoint_every'] == 0:
            rank, metric = validation(model, val, scale, maximum_offset, dt, fixed, indices)
            path = root/'models'/f'operator_{update:06d}.pt'
            checkpoint(path, model, optimizer, generator, update, active, validation=metric)
            last_saved_update = update
            selections.append(dict(update=update, **metric))
            if best_rank is None or rank < best_rank:
                best_rank, selected = rank, path
            write(root/'selection_progress.json', dict(selected_checkpoint=str(selected.relative_to(root)), validations=selections))
        if update == 1 or update % 100 == 0:
            csv_write(root/'training.csv', history)
            write(root/'status.json', dict(phase='training_operator', update=update, target_updates=reported['training_updates'],
                active_seconds=active, elapsed_seconds=time.perf_counter()-start,
                selected_checkpoint=str(selected.relative_to(root)) if selected else None,
                latest_validation=selections[-1] if selections else None,
                learning_rate=reported['learning_rate'], stop_reason=None))
    if not update:
        raise TimeoutError('No operator update fit the overall budget')
    if update != last_saved_update:
        rank, metric = validation(model, val, scale, maximum_offset, dt, fixed, indices)
        path = root/'models'/f'operator_{update:06d}.pt'
        checkpoint(path, model, optimizer, generator, update, active, validation=metric)
        selections.append(dict(update=update, **metric))
        if best_rank is None or rank < best_rank:
            best_rank, selected = rank, path
    selected_state = torch.load(selected, weights_only=False)
    model.load_state_dict(selected_state['model'])
    model.eval()
    shutil.copyfile(selected, root/'models/operator_selected.pt')
    csv_write(root/'training.csv', history)
    write(root/'selection_frozen.json', dict(status='passed', selected_checkpoint=str(selected.relative_to(root)),
        selected_update=selected_state['update'], total_training_updates=update, selected_validation=selected_state['validation'],
        validations=selections, stop_reason=reason, requested_training_updates=reported['training_updates'],
        paper_update_target_completed=update == reported['training_updates'], active_seconds=active,
        wall_seconds=time.perf_counter()-start, parameters=sum(p.numel() for p in model.parameters()),
        learning_rate=reported['learning_rate'], training_physics=False, test_accessed_before_selection=False,
        selected_sha256=sha(root/'models/operator_selected.pt'), reconstruction_sha256=sha(root/'models/reconstruction_selected.pt')))
    return model


def dense_rollout(model, reconstruction, initial, full_scale, indices, remainder, maximum_offset, dt, mode, mechanism):
    representative = mode == 'representative_raw'
    scale = full_scale.subset(indices)
    remaining_scale = full_scale.subset(remainder)
    current = initial.copy()
    trace = [current[indices].copy() if representative else current.copy()]
    gas = ct.Solution(str(mechanism))
    matrix = np.array([[gas.n_atoms(k, e)*gas.atomic_weight(e)/gas.molecular_weights[k]
                        for k in range(gas.n_species)] for e in gas.element_names])
    tau = torch.arange(1, maximum_offset+1, dtype=torch.float32)[:, None]*dt
    reason = 'complete'
    completed_windows = 0
    with torch.inference_mode():
        for window in range(25):
            feature = scale.forward(current[indices][None, :])
            z = model(torch.from_numpy(np.repeat(feature, maximum_offset, axis=0)), tau).numpy()
            reps = scale.inverse(z)
            if representative:
                candidate = reps
            else:
                others = remaining_scale.inverse(reconstruction(torch.from_numpy(z)).numpy())
                candidate = np.empty((maximum_offset, len(current)))
                candidate[:, indices], candidate[:, remainder] = reps, others
                if mode == 'repaired_full':
                    gas.TPY = current[0], ct.one_atm, current[1:]
                    h = gas.enthalpy_mass
                    for j in range(len(candidate)):
                        try:
                            proposal = np.clip(candidate[j, 1:], 0, 1)
                            y = project_species(proposal, current[1:], matrix)
                            gas.HPY = h, ct.one_atm, y
                            candidate[j] = np.r_[gas.T, y]
                        except (ValueError, RuntimeError):
                            candidate[j:] = np.nan
                            break
            for row in candidate:
                if not valid_state(row):
                    reason = 'invalid_state_or_repair_failure'
                    break
                trace.append(row.copy())
            if reason != 'complete':
                break
            completed_windows += 1
            if representative:
                current[indices] = candidate[-1]
            else:
                current = candidate[-1].copy()
    return np.asarray(trace), dict(reason=reason, completed_windows=completed_windows,
                                  complete=completed_windows == 25, points=len(trace))


def evaluate(root, cfg, model, reconstruction, physical, cases, full_scale, indices, remainder, deadline):
    dt = cfg['declared_reconstruction_choices']['sample_dt_s']
    maximum_offset = int(round(cfg['reported_settings']['window_s']/dt))
    variables = json.loads((root/'dataset.json').read_text())['variables']
    active = [j for j, v in enumerate(variables) if v not in ('N2', 'AR')]
    chemistry = ct.Solution(str(root/'mechanism.yaml'))
    elements = np.array([[chemistry.n_atoms(k, e)*chemistry.atomic_weight(e)/chemistry.molecular_weights[k]
                          for k in range(chemistry.n_species)] for e in chemistry.element_names])
    traces, rows, errors, reconstruction_only = {}, [], [], []
    chosen = [c for c in cases if c['split'] != 'train' or (c['T0'] == 1065 and c['phi'] == 1)]
    for i, case in enumerate(chosen):
        if time.perf_counter() > deadline:
            raise TimeoutError('Overall budget exhausted during evaluation')
        truth = physical[case['case_id']]
        traces[f'IC{case["case_id"]}_reference'] = truth
        # Diagnostic only: hold the representative input at the Cantera truth to
        # separate reconstruction error from propagated operator error. This is
        # not used by any rollout, training, or checkpoint selection.
        with torch.inference_mode():
            given = full_scale.subset(indices).forward(truth[:, indices])
            remaining = full_scale.subset(remainder).inverse(reconstruction(torch.from_numpy(given)).numpy())
        rec_error = scores(truth[:, remainder], remaining)
        for j, column in enumerate(remainder):
            reconstruction_only.append(dict(case_id=case['case_id'], split=case['split'], variable=variables[column],
                diagnostic='reconstruction_given_Cantera_representatives_not_a_rollout',
                NRMSE=float(rec_error[j]) if np.isfinite(rec_error[j]) else None,
                negative_prediction_fraction=float(np.mean(remaining[:, j] < 0))))
        for mode in ['representative_raw', 'raw_full', 'repaired_full']:
            pred, info = dense_rollout(model, reconstruction, truth[0], full_scale, indices, remainder,
                                        maximum_offset, dt, mode, root/'mechanism.yaml')
            traces[f'IC{case["case_id"]}_{mode}'] = pred
            columns = indices if mode == 'representative_raw' else np.arange(len(variables))
            target = truth[:, columns]
            values = scores(target, pred) if info['complete'] else None
            scored_columns = np.arange(5) if mode == 'representative_raw' else active
            worst = float(np.max(values[scored_columns])) if values is not None else None
            early_n = int(round(.0001/dt))+1
            early = scores(target[:early_n], pred[:early_n]) if len(pred) >= early_n else None
            mass_error = float(np.max(np.abs(pred[:, 1:].sum(1)-1))) if mode != 'representative_raw' else None
            element_error = float(np.max(np.abs((pred[:, 1:]-truth[0, 1:])@elements.T))) if mode != 'representative_raw' else None
            rows.append(dict(**case, mode=mode, **info, worst_NRMSE=worst,
                joint_pass=bool(worst is not None and worst <= .1),
                criterion='joint5' if mode == 'representative_raw' else 'joint9',
                mass_sum_max_error=mass_error, element_max_error=element_error,
                early_0p1ms_worst_NRMSE=float(np.max(early[scored_columns])) if early is not None else None,
                normalized_profile_MAE=float(np.mean(np.abs(full_scale.subset(columns).forward(target)-full_scale.subset(columns).forward(pred)))) if info['complete'] else None))
            for j, column in enumerate(columns):
                errors.append(dict(case_id=case['case_id'], split=case['split'], T0=case['T0'], phi=case['phi'], mode=mode,
                    variable=variables[column], complete=info['complete'], NRMSE=float(values[j]) if values is not None and np.isfinite(values[j]) else None,
                    early_0p1ms_NRMSE=float(early[j]) if early is not None and np.isfinite(early[j]) else None,
                    reference_RMS=float(np.sqrt(np.mean(target[:, j]**2))),
                    full_horizon_max_abs_error=float(np.max(np.abs(target[:, j]-pred[:, j]))) if info['complete'] else None))
        write(root/'status.json', dict(phase='evaluating', completed_cases=i+1, total_cases=len(chosen)))
    np.savez_compressed(root/'evaluations/traces.npz', **traces)
    csv_write(root/'evaluations/rollouts.csv', rows)
    csv_write(root/'evaluations/NRMSE.csv', errors)
    csv_write(root/'evaluations/reconstruction_given_truth.csv', reconstruction_only)
    return rows


def verify(root, cfg, cases, indices):
    dataset = json.loads((root/'dataset.json').read_text())
    assert dataset['SHA256'] == sha(root/'trajectories.npz')
    assert dataset['mechanism_sha256'] == sha(root/'mechanism.yaml')
    assert len({(c['T0'], c['phi']) for c in cases}) == len(cases)
    source = json.loads((root/'source_manifest.json').read_text())
    assert all(sha(root/'source_snapshot'/name) == expected for name, expected in source.items())
    assert {p: sha(p) for p in ORIGINALS} == ORIGINALS
    variables = dataset['variables']
    checks = 0
    with np.load(root/'evaluations/traces.npz') as traces:
        for row in csv.DictReader((root/'evaluations/rollouts.csv').open()):
            pred = traces[f'IC{row["case_id"]}_{row["mode"]}']
            assert (len(pred) == 5001) == (row['complete'] == 'True')
            assert np.isfinite(pred).all() and np.min(pred[:, 1:]) >= 0
            if row['mode'] == 'repaired_full':
                assert float(row['mass_sum_max_error']) < 1e-8
                assert float(row['element_max_error']) < 1e-8
        for row in csv.DictReader((root/'evaluations/NRMSE.csv').open()):
            if row['complete'] != 'True':
                assert not row['NRMSE']
                continue
            cols = indices if row['mode'] == 'representative_raw' else np.arange(len(variables))
            j = [variables[c] for c in cols].index(row['variable'])
            ref = traces[f'IC{row["case_id"]}_reference'][:, cols]
            pred = traces[f'IC{row["case_id"]}_{row["mode"]}']
            value = scores(ref, pred)[j]
            if np.isfinite(value):
                assert np.isclose(float(row['NRMSE']), value, rtol=1e-11, atol=1e-13)
                checks += 1
            else:
                assert not row['NRMSE'] and row['reference_RMS'] == '0.0'
    write(root/'verification.json', dict(status='passed', original_data_unchanged=True,
        disjoint_IC_splits=True, frozen_sources_unchanged=True, NRMSE_recomputations=checks,
        NRMSE_definition='RMSE(Cantera,NN)/RMS(Cantera), dimensionless ratio; no score for incomplete full horizon',
        macro_feedback_steps=25, dense_queries_per_window=200, publication_before_training=True,
        no_reference_injection=True, no_training_physics=True, test_after_selection=True,
        scope='Protocol reconstruction with declared assumptions, not exact author code/training reproduction'))


def run(root):
    lock = (root/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root/'worker_started.json').exists():
        raise RuntimeError('This root already started; no duplicate training or time-budget reset')
    cfg = json.loads((root/'config.json').read_text())
    prepared = json.loads((root/'prepared.json').read_text())
    assert prepared['status'] == 'passed'
    source = json.loads((root/'source_manifest.json').read_text())
    assert all(sha(root/'source_snapshot'/name) == expected for name, expected in source.items())
    assert sha(root/'config.json') == prepared['config_sha256']
    assert sha(root/'trajectories.npz') == prepared['dataset_sha256']
    assert sha(root/'cases.json') == prepared['cases_sha256']
    assert sha(root/'preprocessing.json') == prepared['preprocessing_sha256']
    publication_gate(cfg)
    write(root/'worker_started.json', dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
        publication_sha=cfg['execution']['required_publication_sha'], budget_seconds=cfg['execution']['overall_wall_seconds']))
    start = time.perf_counter()
    prepare_seconds = json.loads((root/'dataset.json').read_text())['prepare_seconds']
    deadline = start+cfg['execution']['overall_wall_seconds']-prepare_seconds
    training_deadline = deadline-cfg['execution']['evaluation_reserve_seconds']
    torch.set_num_threads(cfg['execution']['threads'])
    torch.set_num_interop_threads(1)
    try:
        with np.load(root/'trajectories.npz') as f:
            physical = f['states']
        cases = json.loads((root/'cases.json').read_text())
        metadata = json.loads((root/'dataset.json').read_text())
        variables = metadata['variables']
        indices = np.array([variables.index(name) for name in REPRESENTATIVE])
        remainder = np.array([j for j in range(len(variables)) if j not in indices])
        prep = json.loads((root/'preprocessing.json').read_text())
        full_scale = FifthRootScale(prep['minimum'], prep['maximum'])
        training = physical[[c['case_id'] for c in cases if c['split'] == 'train']]
        val = physical[[c['case_id'] for c in cases if c['split'] == 'val']]
        reconstruction = train_reconstruction(root, cfg, training, val, full_scale, indices, remainder, training_deadline)
        model = train_operator(root, cfg, training, val, full_scale.subset(indices), indices, training_deadline)
        evaluate(root, cfg, model, reconstruction, physical, cases, full_scale, indices, remainder, deadline)
        verify(root, cfg, cases, indices)
        from .react_deeponet_report import report
        report(root)
        report_complete = json.loads((root/'report/complete.json').read_text())
        assert report_complete['status'] == 'passed'
        for name, expected in report_complete['output_sha256'].items():
            assert sha(root/'report'/name) == expected
        for name, pages in report_complete['figures_per_pdf'].items():
            actual = len(re.findall(rb'/Type\s*/Page\b', (root/'report'/name).read_bytes()))
            assert actual == pages, (name, actual, pages)
        selection = json.loads((root/'selection_frozen.json').read_text())
        write(root/'complete.json', dict(status='passed', wall_seconds=time.perf_counter()-start,
            training_update_target_completed=selection['paper_update_target_completed'],
            stop_reason=selection['stop_reason'], verification_sha256=sha(root/'verification.json'),
            report_complete_sha256=sha(root/'report/complete.json'),
            inference_scores='raw representative joint5 and full-state joint9 are separate',
            model_sha256={'operator': sha(root/'models/operator_selected.pt'), 'reconstruction': sha(root/'models/reconstruction_selected.pt')}))
        write(root/'status.json', dict(phase='complete', status='passed',
            paper_update_target_completed=selection['paper_update_target_completed'], stop_reason=selection['stop_reason']))
        write(root/'exit.json', dict(returncode=0, finished_at=datetime.now(timezone.utc).isoformat()))
    except BaseException as exc:
        write(root/'failure.json', dict(status='failed', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        write(root/'status.json', dict(phase='failed', error=type(exc).__name__, message=str(exc)))
        write(root/'exit.json', dict(returncode=1, finished_at=datetime.now(timezone.utc).isoformat()))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'run'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/react_deeponet_reconstruction.json'))
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.root.resolve(), args.config)
    else:
        run(args.root.resolve())


if __name__ == '__main__':
    main()
