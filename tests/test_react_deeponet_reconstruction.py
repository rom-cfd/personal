"""Scientific preflight: data separation, time queries and recursive semantics."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kinetics.react_deeponet_experiment import (
    FifthRootScale, case_definitions, dense_rollout, publication_gate,
    representative_rollout, run, sample_windows,
)
from kinetics.react_deeponet_reconstruction import ReactDeepONet


def test_reported_architecture_has_time_and_trainable_shift():
    torch.manual_seed(11)
    model = ReactDeepONet()
    assert sum(p.numel() for p in model.parameters()) == 46186
    state = torch.randn((8, 5))
    a = model(state, torch.full((8, 1), 1e-6))
    b = model(state, torch.full((8, 1), 20e-6))
    assert a.shape == (8, 5)
    assert not torch.allclose(a, b)
    a.square().mean().backward()
    for network in [model.branch, model.trunk, model.shift]:
        assert sum(float(p.grad.abs().sum()) for p in network.parameters()) > 0


def test_fixed_split_is_disjoint_and_paper_curve_is_labelled_seen():
    cfg = json.loads(Path('configs/react_deeponet_reconstruction.json').read_text())
    cases = case_definitions(cfg)
    assert len(cases) == 68
    assert len({(c['T0'], c['phi']) for c in cases}) == 68
    original_curve = next(c for c in cases if c['T0'] == 1065 and c['phi'] == 1)
    assert original_curve['split'] == 'train' and original_curve['seen_training_curve']
    assert {c['phi'] for c in cases if c['split'] == 'test_extrapolation'} == {.6, .7, 1.3}


def test_fifth_root_preserves_trace_species_and_exposes_negative_decoder():
    training = np.array([[[1000., 0., .7], [2000., 1e-18, .7], [3000., .01, .7]]])
    scale = FifthRootScale.fit(training)
    np.testing.assert_allclose(scale.inverse(scale.forward(training)), training, rtol=2e-6, atol=1e-12)
    z = scale.forward(training[0, :1]).astype(float)
    z[0, 1] = -2
    assert scale.inverse(z)[0, 1] < 0
    assert scale.inverse(z)[0, 2] == .7
    with pytest.raises(ValueError, match='Negative'):
        scale.forward(np.array([[1000, -1e-12, .7]]))


def test_window_pairs_stay_within_one_trajectory_and_match_tau():
    data = torch.zeros((3, 30, 5))
    for i in range(3):
        data[i, :, 0] = i
        data[i, :, 1] = torch.arange(30)
    x, tau, y = sample_windows(data, 1000, 7, 1e-7, torch.Generator().manual_seed(2))
    assert torch.equal(x[:, 0], y[:, 0])
    np.testing.assert_allclose((y[:, 1]-x[:, 1]).numpy(), tau[:, 0].numpy()/1e-7, rtol=1e-6)
    assert torch.all((tau > 0) & (tau <= 7e-7))
    assert int(x[:, 1].max()) <= 22


class IdentityScale:
    def forward(self, value):
        return np.asarray(value, dtype=np.float32)

    def inverse(self, value):
        return np.asarray(value, dtype=float)

    def subset(self, indices):
        return self


class BiasedTimePropagator(torch.nn.Module):
    def forward(self, value, tau):
        result = value.clone()
        result[:, 0] += tau[:, 0]*1e6+1
        return result


def test_macro_feedback_accumulates_prediction_error():
    initial = np.array([1000, .02, .001, .2, .1])
    trace = representative_rollout(BiasedTimePropagator(), IdentityScale(), initial, 25, 20e-6)
    assert len(trace) == 26
    assert trace[-1, 0] == pytest.approx(1525)


def test_dense_interior_queries_are_not_extra_feedback_steps():
    initial = np.array([1000, .02, .001, .2, .1])
    trace, info = dense_rollout(BiasedTimePropagator(), None, initial, IdentityScale(),
        np.arange(5), np.array([], dtype=int), 200, 1e-7, 'representative_raw', 'h2o2.yaml')
    assert info['complete'] and info['completed_windows'] == 25
    assert len(trace) == 5001
    assert trace[1, 0] == pytest.approx(1001.1, abs=1e-4)
    assert trace[200, 0] == pytest.approx(1021)
    assert trace[201, 0] == pytest.approx(1022.1, abs=1e-4)
    assert trace[-1, 0] == pytest.approx(1525)


def test_duplicate_run_cannot_reset_the_training_budget(tmp_path):
    (tmp_path/'worker_started.json').write_text('{}')
    with pytest.raises(RuntimeError, match='no duplicate training'):
        run(tmp_path)
    assert not (tmp_path/'models').exists()


def test_publication_gate_requires_verified_expected_commit(tmp_path):
    p = tmp_path/'publication.json'
    p.write_text(json.dumps(dict(status='passed', remote_verified=False, branch='bw', commit_sha='a'*40, remote_sha='a'*40)))
    cfg = {'execution': {'publication_record': str(p), 'required_publication_sha': 'a'*40}}
    with pytest.raises(AssertionError):
        publication_gate(cfg)
