"""React-DeepONet protocol reconstruction; reported settings and assumptions split.

Source: Kumar & Echekki, arXiv:2304.12188v1, Table 1, H2 column.
This is new PyTorch code, not released author code or weights. A log-time shift
is supported by the author's thesis, chapter 3; details missing from the 2023
preprint are recorded in the configuration instead of silently guessed.
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/react-deeponet-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/react-deeponet-cache')

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPRESENTATIVE = ['T', 'H2', 'H', 'O2', 'H2O']


def dense(layers):
    modules = []
    for j, (a, b) in enumerate(zip(layers[:-1], layers[1:])):
        layer = nn.Linear(a, b)
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)
        modules.append(layer)
        if j < len(layers)-2:
            modules.append(nn.Tanh())
    return nn.Sequential(*modules)


class ReactDeepONet(nn.Module):
    """5-80-80-200 branch, 1-80-80-200 trunk, 5-10-10-1 ShiftNet."""

    def __init__(self, tau_min_s=1e-7, tau_max_s=20e-6):
        super().__init__()
        self.branch = dense([5, 80, 80, 200])
        self.trunk = dense([1, 80, 80, 200])
        self.shift = dense([5, 10, 10, 1])
        self.bias = nn.Parameter(torch.zeros(5))
        self.register_buffer('log_min', torch.tensor(np.log(tau_min_s), dtype=torch.float32))
        self.register_buffer('log_span', torch.tensor(np.log(tau_max_s/tau_min_s), dtype=torch.float32))

    def forward(self, state, tau_s):
        # Scaling of log(time) is an affine input reparameterization. ShiftNet
        # learns an additive log-time shift, equivalent to a linear-time stretch.
        log_time = 2*(torch.log(tau_s)-self.log_min)/self.log_span-1
        location = log_time+self.shift(state)
        b = self.branch(state).reshape(-1, 5, 40)
        t = self.trunk(location).reshape(-1, 5, 40)
        return (b*t).sum(dim=-1)+self.bias


class ReconstructionNet(nn.Module):
    """Separate remaining-species map; its unreported architecture is an assumption."""

    def __init__(self, outputs=6):
        super().__init__()
        self.net = dense([5, 64, 64, outputs])

    def forward(self, x):
        return self.net(x)


def benchmark():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(11)
    model = ReactDeepONet()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    rng = torch.Generator().manual_seed(113)
    x = torch.rand((20000, 5), generator=rng)*2-1
    tau = torch.exp(torch.rand((20000, 1), generator=rng)*np.log(200))*1e-7
    y = torch.rand((20000, 5), generator=rng)*2-1
    times = []
    for j in range(13):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x, tau)-y)**2)
        loss.backward()
        optimizer.step()
        if j >= 3:
            times.append(time.perf_counter()-start)
    result = dict(kind='synthetic_speed_probe_not_research_training', parameters=sum(p.numel() for p in model.parameters()),
                  batch_size=20000, threads=2, measured_updates=10,
                  median_seconds_per_update=float(np.median(times)),
                  estimated_200000_updates_hours=float(np.median(times)*200000/3600),
                  torch=torch.__version__, device='cpu')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    if args.benchmark:
        benchmark()
