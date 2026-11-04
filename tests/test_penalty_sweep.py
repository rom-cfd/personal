import json
from pathlib import Path

import numpy as np
import pytest
import matplotlib.pyplot as plt
from kinetics.soft_constraint_training import SoftTrainer, FAMILIES
from kinetics.penalty_sweep_study import validate_config, deadlines
from scripts.report_penalty_sweep import (METHODS, TEAMS, WEIGHTS, normalized_metrics,
                                         profile_figure, table_figure)
from scripts.nrmse_util_reference import _nrmse_1d
from test_soft_constraint_training import setup


@pytest.mark.parametrize('family', FAMILIES)
def test_weights_share_fresh_initialization_features_and_minibatch_stream(family):
    x, y, cfg, matrix = setup(family)
    trainers = [SoftTrainer(x, y, cfg, 11, family, matrix, w) for w in (1., 10., 100., 1000.)]
    control = trainers[1]
    for trainer in trainers:
        for a, b in zip(trainer.predictor.model.get_weights(), control.predictor.model.get_weights()):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(trainer.features.numpy(), control.features.numpy())
        np.testing.assert_array_equal(trainer.targets.numpy(), control.targets.numpy())
        assert trainer.rng.bit_generator.state == control.rng.bit_generator.state
        assert trainer.update == int(trainer.optimizer.iterations.numpy()) == 0


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('weight', (1., 100., 1000.))
def test_each_requested_weight_has_exact_resume_and_finite_losses(tmp_path, family, weight):
    x, y, cfg, matrix = setup(family)
    whole = SoftTrainer(x, y, cfg, 11, family, matrix, weight); whole.advance(5)
    part = SoftTrainer(x, y, cfg, 11, family, matrix, weight); part.advance(2); part.full_train_loss()
    cp = part.save_checkpoint(tmp_path / f'{family}_{int(weight)}', 'test')
    restored = SoftTrainer.restore(cp, x, y, cfg, 11, family, matrix, weight); restored.advance(5)
    for a, b in zip(whole.predictor.model.get_weights(), restored.predictor.model.get_weights()):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(whole.optimizer.variables, restored.optimizer.variables):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    assert whole.rng.bit_generator.state == restored.rng.bit_generator.state
    loss = restored.full_train_loss()
    assert np.isfinite([loss['full_train_loss'], loss['data_loss'], loss['element_loss']]).all()
    assert loss['full_train_loss'] == pytest.approx(loss['data_loss'] + weight * loss['element_loss'], rel=1e-14)
    assert restored.mu == weight and restored.signature['mu'] == weight
    with pytest.raises(ValueError, match='provenance'):
        SoftTrainer.restore(cp, x, y, cfg, 11, family, matrix, 10.)


def test_scope_and_absolute_budget_resume_do_not_add_compute(tmp_path):
    cfg = json.loads(Path('configs/penalty_weight_sweep.json').read_text()); validate_config(cfg)
    with pytest.raises(ValueError):
        validate_config({**cfg, 'lambdas': [1., 10., 100., 1000.]})
    with pytest.raises(ValueError):
        validate_config({**cfg, 'updates_per_member': 20000})
    deadlines(tmp_path, cfg); before = (tmp_path / 'budget.json').read_bytes()
    deadlines(tmp_path, cfg); assert (tmp_path / 'budget.json').read_bytes() == before
    budget = json.loads(before)
    assert budget['authorized_updates'] == 1200000
    assert budget['expires_unix'] - budget['started_unix'] == cfg['pipeline_wall_safety_seconds']


def sample_metrics(empty=False):
    truth = np.array([[1., .1, .001, .001, .2, .001, .02, .001, .001, .7],
                      [1.1, .09, .002, .002, .19, .002, .03, .002, .002, .7],
                      [1.2, .08, .003, .003, .18, .003, .04, .003, .003, .7]])
    data = dict(states=np.stack([truth, truth]), dt=np.array(2e-6), initials=np.array([[1000., 101325., 1.]] * 2))
    traces = {method: {i: truth * np.r_[1.1, np.ones(9)] for i in (0, 1)} for method, _, _ in METHODS}
    traces['lambda1000_ann10'][1] = traces['lambda1000_ann10'][1][:2]
    if empty:
        traces['lambda1_p_only10'][0] = traces['lambda1_p_only10'][0][:2]
    return data, traces


def test_util_default_and_explicit_global_vs_within_team_completed_sets():
    data, traces = sample_metrics()
    frame, summary, within_summary, common, within = normalized_metrics(data, traces, [0, 1], 'repaired')
    assert common == [0] and within['ann10'] == [0] and within['p_only10'] == [0, 1]
    failed = frame[(frame.method == 'lambda1000_ann10') & (frame.trajectory == 1)]
    assert not failed.complete.any() and failed.nrmse.isna().all() and failed.full_rmse.isna().all()
    row = frame[(frame.method == 'lambda1_ann10') & (frame.trajectory == 0) & (frame.variable == 'T')].iloc[0]
    assert row.nrmse == _nrmse_1d(data['states'][0, :, 0], traces['lambda1_ann10'][0][:, 0])
    assert (frame.loc[frame.variable == 'N2'].dropna(subset=['nrmse']).nrmse == 0).all()
    assert set(summary.common_ICs) == {'[0]'}
    assert set(within_summary.loc[within_summary.team == 'p_only10', 'common_ICs']) == {'[0, 1]'}


def test_empty_global_set_is_na_without_discarding_valid_within_team_comparisons():
    data, traces = sample_metrics(True)
    _, summary, within_summary, common, within = normalized_metrics(data, traces, [0, 1], 'raw')
    assert common == [] and summary.median_nrmse.isna().all()
    assert within['ann10'] == [0] and within['p_only10'] == [1]
    assert within_summary.median_nrmse.notna().all()
    fig = table_figure(summary, common, 'raw'); assert fig.axes[0].tables; plt.close(fig)


def test_profile_has_all_seven_teams_ten_variables_and_six_curves():
    data, _ = sample_metrics()
    data['states'] = np.repeat(data['states'][:, :1], 5001, axis=1)
    traces = {method: {i: data['states'][i].copy() for i in (0, 1)} for method, _, _ in METHODS}
    fig = profile_figure(data, traces, 0, 'repaired')
    assert len(fig.axes) == 70 and all(len(ax.lines) == 6 for ax in fig.axes)
    assert all(ax.get_xlim() == (0., 10.) for ax in fig.axes)
    assert len(TEAMS) == 7 and WEIGHTS == (0, 1, 10, 100, 1000)
    plt.close(fig)
