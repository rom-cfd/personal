"""Controlled optimizer/architecture diagnostics; old results remain read-only.

All candidates use identical train/validation trajectories and mini-batch draws.
The gated model is an independently written port of the mathematical gating
rule in the publicly acknowledged PI-DeepONet base, not a claim that the private
React-DeepONet source used that exact implementation.
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from .react_deeponet_reconstruction import ReactDeepONet, dense
from .react_deeponet_experiment import FifthRootScale, csv_write, write, sha, sample_windows, validation, representative_rollout, scores


class GatedMLP(nn.Module):
    """Two input encoders U,V and tanh gates; same rule as modified_MLP."""
    def __init__(self, sizes):
        super().__init__()
        self.u = nn.Linear(sizes[0], sizes[1])
        self.v = nn.Linear(sizes[0], sizes[1])
        self.gates = nn.ModuleList([nn.Linear(a, b, bias=False) for a, b in zip(sizes[:-2], sizes[1:-1])])
        self.output = nn.Linear(sizes[-2], sizes[-1])
        for layer in [self.u, self.v, *self.gates, self.output]:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        u, v = torch.tanh(self.u(x)), torch.tanh(self.v(x))
        h = x
        for layer in self.gates:
            gate = torch.tanh(layer(h))
            h = gate*u+(1-gate)*v
        return self.output(h)


class GatedReactDeepONet(ReactDeepONet):
    def __init__(self):
        super().__init__()
        self.branch = GatedMLP([5,80,80,200])
        self.trunk = GatedMLP([1,80,80,200])


CANDIDATES = [
    dict(name='dense_adam_1e2_fixed', family='dense', optimizer='Adam', lr=.01, decay=False),
    dict(name='dense_adam_1e3_fixed', family='dense', optimizer='Adam', lr=.001, decay=False),
    dict(name='dense_adamax_1e2_decay', family='dense', optimizer='Adamax', lr=.01, decay=True),
    dict(name='dense_adamax_1e3_decay', family='dense', optimizer='Adamax', lr=.001, decay=True),
    dict(name='gated_adamax_1e2_decay', family='gated', optimizer='Adamax', lr=.01, decay=True),
    dict(name='gated_adamax_1e3_decay', family='gated', optimizer='Adamax', lr=.001, decay=True),
]


def create(spec):
    torch.manual_seed(11)
    return ReactDeepONet() if spec['family'] == 'dense' else GatedReactDeepONet()


def run(source, root, updates, batch, budget):
    if root.exists():
        raise FileExistsError('Use a new diagnostics root; never overwrite an experiment')
    root.mkdir(parents=True)
    shutil.copyfile(__file__, root/'frozen_source.py')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    deadline = started+budget
    with np.load(source/'trajectories.npz') as f:
        states = f['states']
    cases = json.loads((source/'cases.json').read_text())
    train = states[[c['case_id'] for c in cases if c['split']=='train']]
    val = states[[c['case_id'] for c in cases if c['split']=='val']]
    prep = json.loads((source/'preprocessing.json').read_text())
    indices = np.array([0,1,2,4,6])
    scale = FifthRootScale(prep['minimum'], prep['maximum']).subset(indices)
    training, valid = torch.from_numpy(scale.forward(train[:,:,indices])), torch.from_numpy(scale.forward(val[:,:,indices]))
    fixed = sample_windows(valid, 8192, 200, 1e-7, torch.Generator().manual_seed(19))
    write(root/'config.json',dict(source=str(source), dataset_sha256=sha(source/'trajectories.npz'), candidates=CANDIDATES,
        updates_per_candidate=updates, batch_size=batch, budget_seconds=budget, seed=11, threads=2,
        selection_uses='validation only; historical test never read by diagnostic runner',
        gated_source='https://github.com/PredictiveIntelligenceLab/Long-time-Integration-PI-DeepONets/blob/b6ec2cef29cd62da5e7aef34f43ebd0a2ddcc4ba/stiff/PI_DeepONet_stiff.ipynb'))
    all_history, summaries = [], []
    for number, spec in enumerate(CANDIDATES):
        directory=root/spec['name'];directory.mkdir()
        model=create(spec)
        initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        optimizer=getattr(torch.optim,spec['optimizer'])(model.parameters(),lr=spec['lr'])
        generator=torch.Generator().manual_seed(18)
        best=float('inf');history=[];best_update=None
        model_start=time.perf_counter()
        for update in range(1,updates+1):
            if time.perf_counter()>=deadline:
                raise TimeoutError('Diagnostic total budget exhausted; no automatic restart')
            lr=spec['lr']*(.9**((update-1)/5000) if spec['decay'] else 1)
            optimizer.param_groups[0]['lr']=lr
            x,t,y=sample_windows(training,batch,200,1e-7,generator)
            optimizer.zero_grad(set_to_none=True)
            loss=torch.mean((model(x,t)-y)**2)
            if not torch.isfinite(loss):
                raise FloatingPointError(spec['name']+' nonfinite loss')
            loss.backward();optimizer.step()
            if update==1 or update%100==0:
                history.append(dict(candidate=spec['name'],update=update,loss=float(loss.detach()),lr=lr,
                    weight_norm=float(torch.sqrt(sum(p.detach().square().sum() for p in model.parameters()))),
                    wall_seconds=time.perf_counter()-model_start))
            if update==1 or update%500==0:
                rank,metric=validation(model,val,scale,200,1e-7,fixed,indices)
                metric.update(update=update)
                if metric['window_MSE']<best:
                    best=metric['window_MSE'];best_update=update
                    torch.save(dict(model=model.state_dict(),spec=spec,update=update,validation=metric),directory/'best.pt')
                write(root/'status.json',dict(phase='diagnostic_training',candidate=spec['name'],candidate_index=number+1,
                    total_candidates=len(CANDIDATES),update=update,target_updates=updates,latest_validation=metric,
                    elapsed_seconds=time.perf_counter()-started))
                csv_write(directory/'training.csv',history)
        chosen=torch.load(directory/'best.pt',weights_only=False)
        model.load_state_dict(chosen['model']);model.eval()
        dense_complete,dense_pass,worst=[],0,[]
        from .react_deeponet_experiment import dense_rollout
        for truth in val:
            trace,info=dense_rollout(model,None,truth[0],FifthRootScale(prep['minimum'],prep['maximum']),indices,
                np.array([3,5,7,8,9,10]),200,1e-7,'representative_raw',source/'mechanism.yaml')
            dense_complete.append(info['complete'])
            if info['complete']:
                error=float(np.max(scores(truth[:,indices],trace)));worst.append(error);dense_pass+=int(error<=.1)
        torch.save(dict(model=model.state_dict(),spec=spec,update=best_update),directory/'selected.pt')
        summary=dict(**spec,parameters=sum(p.numel() for p in model.parameters()),updates=updates,best_update=best_update,
            best_window_MSE=best,complete=sum(dense_complete),total=len(val),joint5=dense_pass,
            median_worst5=float(np.median(worst)) if worst else None,wall_seconds=time.perf_counter()-model_start)
        summaries.append(summary);all_history.extend(history)
        csv_write(root/'summary.csv',summaries);csv_write(root/'training.csv',all_history)
        print('DIAGNOSTIC',json.dumps(summary),flush=True)
    write(root/'complete.json',dict(status='passed',elapsed_seconds=time.perf_counter()-started,
        dataset_unchanged=sha(source/'trajectories.npz')==json.loads((root/'config.json').read_text())['dataset_sha256'],
        test_used=False,output_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='status.json'}))
    write(root/'status.json',dict(phase='complete',status='passed',finished_candidates=len(summaries)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('results/react_deeponet_reconstruction/run_001'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=6000)
    parser.add_argument('--batch',type=int,default=2048)
    parser.add_argument('--budget',type=float,default=1200)
    args=parser.parse_args()
    run(args.source,args.root,args.updates,args.batch,args.budget)
