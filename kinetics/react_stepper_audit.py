"""Validation-only diagnosis of existing React-DeepONet feedback checkpoints.

Never train, clip outputs, open historical test arrays, or modify a source run.
Boundary admissibility and interior-query diagnostics are explicitly separate.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time

import numpy as np
import torch

from .react_deeponet_experiment import FifthRootScale
from .react_deeponet_v2_model import DetailedReactDeepONet
from .react_deeponet_reconstruction import ReconstructionNet

REP = [0, 1, 2, 4, 6]
REST = [3, 5, 7, 8, 9, 10]
ACTIVE = list(range(9))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def errors(truth, prediction):
    denominator = np.sqrt(np.mean(truth ** 2, axis=0))
    return np.divide(np.sqrt(np.mean((prediction - truth) ** 2, axis=0)), denominator,
                     out=np.full_like(denominator, np.nan), where=denominator > 0)

def admissible(x):
    return np.isfinite(x).all(axis=-1) & (x[..., 0] >= 200) & (x[..., 0] <= 6000) & (x[..., 1:].min(axis=-1) >= 0)

def audit(root):
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(2)
    source = Path('results/react_deeponet_rebuild/run_001')
    with np.load(source / 'data/val.npz') as f:
        truth = f['states']
    cases = [c for c in json.loads((source / 'cases.json').read_text()) if c['split'] == 'val']
    prep = json.loads((source / 'preprocessing.json').read_text())
    scale = FifthRootScale(prep['minimum'], prep['maximum'])
    rs, ss = scale.subset(REP), scale.subset(REST)
    phi = torch.tensor([c['phi'] for c in cases], dtype=torch.float32)[:, None]
    cnet = ReconstructionNet()
    cnet.load_state_dict(torch.load(source / 'models/reconstruction_selected.pt', weights_only=False)['model'])
    cnet.eval()
    checkpoints = {
        'pretrain_72500': source / 'models/pretrain_072500.pt',
        'pretrain_100000': source / 'models/pretrain_100000.pt',
        'endpoint_selected': Path('results/react_refinement/run_001/models/endpoint_lr_div100/selected.pt'),
        'all_steps_selected': Path('results/react_refinement/run_001/models/all_steps_lr_div100/selected.pt'),
    }
    rows = []
    with torch.inference_mode():
        for name, path in checkpoints.items():
            model = DetailedReactDeepONet()
            model.load_state_dict(torch.load(path, weights_only=False)['model'])
            model.eval()
            for stride in [20, 200]:
                dt = stride * 1e-7
                n = (truth.shape[1] - 1) // stride
                z = torch.from_numpy(rs.forward(truth[:, 0, REP]))
                trace = [truth[:, 0].copy()]
                for step in range(n):
                    z = model(z, torch.full((len(cases), 1), dt), phi)
                    representative = rs.inverse(z.numpy())
                    rest = ss.inverse(cnet(z).numpy())
                    state = np.empty((len(cases), 11))
                    state[:, REP] = representative
                    state[:, REST] = rest
                    trace.append(state)
                pred = np.stack(trace, axis=1)
                ref = truth[:, ::stride]
                for i, case in enumerate(cases):
                    e = errors(ref[i], pred[i])
                    er = errors(ref[i, :, REP].T, pred[i, :, REP].T)
                    early = errors(ref[i, :int(round(.0001 / dt)) + 1], pred[i, :int(round(.0001 / dt)) + 1])
                    rows.append(dict(model=name, dt_s=dt, IC=case['case_id'], T0=case['T0'], phi=case['phi'],
                        representative_finite=bool(np.isfinite(pred[i, :, REP]).all()),
                        representative_boundary_valid=bool(admissible(pred[i][:, REP]).all()),
                        full_boundary_valid=bool(admissible(pred[i]).all()),
                        representative_joint5=bool(np.all(er <= .1)),
                        full_joint9=bool(np.all(e[ACTIVE] <= .1)),
                        full_and_early_joint9=bool(np.all(e[ACTIVE] <= .1) and np.all(early[ACTIVE] <= .1)),
                        worst5=float(np.max(er)), worst9=float(np.max(e[ACTIVE])),
                        min_representative=float(pred[i][:, REP[1:]].min()), minimum_mass_fraction=float(pred[i, :, 1:].min())))
                np.savez_compressed(root / f'{name}_dt{stride}.npz', prediction=pred, reference=ref)
    with (root / 'boundary_diagnosis.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = []
    for name in checkpoints:
        for dt in [2e-6, 20e-6]:
            selected = [r for r in rows if r['model'] == name and np.isclose(r['dt_s'], dt, atol=1e-12, rtol=0)]
            summary.append(dict(model=name, dt_s=dt, total=len(selected),
                representative_boundary_valid=sum(r['representative_boundary_valid'] for r in selected),
                representative_curve=sum(r['representative_joint5'] for r in selected),
                full_boundary_valid=sum(r['full_boundary_valid'] for r in selected),
                joint9=sum(r['full_joint9'] for r in selected),
                joint9_with_full_physical=sum(r['full_joint9'] and r['full_boundary_valid'] for r in selected),
                full_early_with_physical=sum(r['full_and_early_joint9'] and r['full_boundary_valid'] for r in selected)))
    record = dict(status='passed', elapsed_seconds=time.perf_counter() - started, test_access=False,
        validation_conditions=cases, summary=summary, no_clipping=True, no_training=True,
        complete_sha256=sha(source / 'complete.json'), checkpoint_sha256={k:sha(p) for k,p in checkpoints.items()},
        note='Finite but invalid trajectories remain diagnostic, never valid raw solver successes. Only actual feedback boundaries are checked here; dense interior queries were checked separately in the prior runs.')
    (root / 'analysis.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    audit(p.parse_args().root)
