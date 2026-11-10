"""Independent implementation of Venturi--Casey Case 3 (not React-DeepONet).

Protocol reference: arXiv:2204.12670v3, Fig.15 and Table S3, and the author's
ROMNet_ revision 07278fc3ffe9eda7be7eed2cf96068f4be3efa51 / H2_TestCase3.
There is no feedback stepping, correlation net, Box--Cox, or physics loss.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ParallelMLP(nn.Module):
    """Separate dense networks evaluated together; weights are never shared."""
    def __init__(self, count, sizes):
        super().__init__()
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for left, right in zip(sizes[:-1], sizes[1:]):
            w = nn.Parameter(torch.empty(count, left, right, dtype=torch.float64))
            std = math.sqrt(2.0 / (left + right)) / 0.87962566103423978
            nn.init.trunc_normal_(w, std=std, a=-2*std, b=2*std)
            self.weights.append(w)
            self.biases.append(nn.Parameter(torch.zeros(count, right, dtype=torch.float64)))

    def forward(self, x):
        # x: [batch, independent network, feature].
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = torch.einsum('bgi,gio->bgo', x, weight) + bias
            if i < len(self.weights)-1:
                x = torch.tanh(x)
        return x


class FlexDeepONet(nn.Module):
    def __init__(self, input_range=None, output_range=None):
        super().__init__()
        self.branch = ParallelMLP(19, [19, 16, 16, 16, 9])
        self.trunk = ParallelMLP(19, [1, 16, 16, 16, 8])
        self.stretch = ParallelMLP(1, [19, 16, 16, 16, 19])
        self.register_buffer('input_range', torch.as_tensor(
            [1.0]*19 if input_range is None else input_range, dtype=torch.float64))
        self.register_buffer('output_range', torch.as_tensor(
            [1.0]*19 if output_range is None else output_range, dtype=torch.float64))

    def forward(self, initial_state, query_time):
        x = initial_state / self.input_range
        coefficients = self.branch(x[:, None, :].expand(-1, 19, -1))
        stretch = F.softplus(self.stretch(x[:, None, :])[:, 0, :])
        # Upstream PreNet multiplies physical time FIRST; TransFun then logs it.
        coordinates = torch.log(query_time.reshape(-1, 1) * stretch)
        basis = self.trunk(coordinates[:, :, None])
        return (coefficients[:, :, :8] * basis).sum(-1) + coefficients[:, :, 8]

    def physical(self, initial_state, query_time):
        return self(initial_state, query_time) * self.output_range

    def regularization(self):
        # Upstream regularizes all kernels and only hidden-layer biases.
        params = []
        for network in [self.branch, self.trunk, self.stretch]:
            params.extend(network.weights)
            params.extend(list(network.biases)[:-1])
        return sum(1e-12 * (p.abs().sum() + p.square().sum()) for p in params)


class KerasAdam(torch.optim.Optimizer):
    """Adam with Keras' epsilon-hat placement, rather than PyTorch's epsilon."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state.update(step=0, first=torch.zeros_like(p), second=torch.zeros_like(p))
                state['step'] += 1
                state['first'].lerp_(p.grad, 1-beta1)
                state['second'].lerp_(p.grad.square(), 1-beta2)
                step = state['step']
                alpha = group['lr'] * math.sqrt(1-beta2**step) / (1-beta1**step)
                p.addcdiv_(state['first'], state['second'].sqrt().add_(group['eps']), value=-alpha)


def author_input_range(initial_states):
    """Match the pinned CustomNormalization.adapt min/max initialization.

    Its running extrema start at one, and 'range' does not subtract the min.
    Preserve this behavior explicitly, including constant-zero input channels.
    Output ranges use actual training-pool extrema, as in data/data.py.
    """
    import numpy as np
    lo = np.minimum(1.0, initial_states.min(axis=0))
    hi = np.maximum(1.0, initial_states.max(axis=0))
    span = hi-lo
    if not np.all(span > 0):
        raise ValueError('Upstream input normalization has a zero range')
    return span
