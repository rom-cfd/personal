import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import matplotlib.pyplot as plt
from kinetics.soft_constraint_training import SoftTrainer, FAMILIES
from kinetics.weight_long_study import train_member, deadlines
from kinetics.penalty_sweep_study import weight_root
from scripts.run_weight_long_supervisor import run_supervised
from scripts.report_weight_long import PER_WEIGHT_METHODS, metrics_frame, summaries, profile_figure, table_figure
from test_soft_constraint_training import setup
from test_deeponet_long_training import POLICY


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('weight', (1., 100., 1000.))
def test_continuation_preserves_adam_rng_history_and_recovers_an_orphan_checkpoint(tmp_path, family, weight):
    x, y, model_cfg, matrix = setup(family)
    parent = SoftTrainer(x, y, model_cfg, 11, family, matrix, weight); parent.advance(3); parent.full_train_loss()
    origin = parent.save_checkpoint(tmp_path / 'original', 'short_endpoint')
    expected = SoftTrainer(x, y, model_cfg, 11, family, matrix, weight); expected.advance(10)
    root = tmp_path / 'long'; root.mkdir(); (root / 'run.json').write_text('{}\n')
    snapshot = dict(model_configs={family: model_cfg}, element_matrix=matrix.tolist(),
        origin_members=[dict(mu=weight, family=family, seed=11, updates=3, checkpoint=str(origin))])
    if family == 'tp_phi':
        pairs = dict(x=np.c_[x[:, :10], x[:, -1]], y=y,
                     parent=np.c_[np.arange(len(x)), np.zeros(len(x), int)])
        data = dict(initials=np.c_[x[:, 0], x[:, -1], x[:, 10]])
    else:
        pairs = dict(x=x, y=y); data = {}
    cfg = dict(plateau=POLICY, observe_every=2, checkpoint_every=2)
    first = train_member(root, cfg, snapshot, pairs, data, weight, family, 11, float('inf'), maximum_update=8)
    orphan = SoftTrainer.restore(first['checkpoint'], x, y, model_cfg, 11, family, matrix, weight)
    orphan.advance(9); orphan.full_train_loss()
    observed = weight_root(root, weight) / 'checkpoints' / family / 'seed11/update_0000009_orphan'
    orphan.save_checkpoint(observed, 'observed_checkpoint')
    second = train_member(root, cfg, snapshot, pairs, data, weight, family, 11, float('inf'), maximum_update=10)
    assert second['resume_start_update'] == 9 and second['added_updates'] == 1 and second['total_added_updates'] == 7
    restored = SoftTrainer.restore(second['checkpoint'], x, y, model_cfg, 11, family, matrix, weight)
    for a, b in zip(expected.predictor.model.get_weights(), restored.predictor.model.get_weights()):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(expected.optimizer.variables, restored.optimizer.variables):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    assert expected.rng.bit_generator.state == restored.rng.bit_generator.state
    assert restored.plateau_monitor['origin_update'] == 3 and restored.plateau_monitor['enabled']
    assert restored.full_train_history[0] == parent.full_train_history[0]
    assert restored.mu == weight and restored.update == 10
    (root / 'run.json').write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match='lineage'):
        train_member(root, cfg, snapshot, pairs, data, weight, family, 11, float('inf'), maximum_update=11)


def test_absolute_budget_survives_resume_without_new_time(tmp_path):
    cfg = dict(evaluation_reserve_seconds=3600)
    deadlines(tmp_path, cfg); original = (tmp_path / 'budget.json').read_bytes()
    deadlines(tmp_path, cfg); assert (tmp_path / 'budget.json').read_bytes() == original
    b = json.loads(original)
    assert b['expires_unix'] - b['started_unix'] == 21600
    assert b['training_expires_unix'] - b['started_unix'] == 18000


def test_resume_skips_finished_first_pass_members_before_allocating_remaining_models(tmp_path, monkeypatch):
    import kinetics.weight_long_study as module
    records = {(1., 'ann', 11): dict(mu=1., family='ann', seed=11, reason='time_limit', updates=10001)}
    calls = []
    def latest(root, w, f, s):
        return records.get((w, f, s))
    def fake_train(root, cfg, snapshot, pairs, data, w, f, s, deadline):
        calls.append((w, f, s)); records[w, f, s] = dict(mu=w, family=f, seed=s, reason='time_limit', updates=11000)
    monkeypatch.setattr(module, 'latest', latest); monkeypatch.setattr(module, 'train_member', fake_train)
    monkeypatch.setattr(module.time, 'perf_counter', lambda: 0.)
    cfg = dict(lambdas=[1., 100., 1000.]); snapshot = dict(base_config=dict(seeds=[11]))
    for w in cfg['lambdas']:
        weight_root(tmp_path, w).mkdir()
    module.train(tmp_path, cfg, snapshot, {}, {}, 20.)
    assert (1., 'ann', 11) not in calls and len(calls) == 11
    assert len(json.loads((tmp_path / 'final_members.json').read_text())) == 12


def test_supervised_worker_runs_in_its_own_os_session_and_reports_exit(tmp_path):
    output = tmp_path / 'session.json'
    code = 'import os,json;from pathlib import Path;Path(' + repr(str(output)) + ').write_text(json.dumps([os.getpid(),os.getsid(0)]))'
    result = run_supervised([sys.executable, '-c', code], time.time() + 10, poll_seconds=.02)
    assert result['returncode'] == 0 and not result['timed_out']
    pid, session = json.loads(output.read_text()); assert pid == session and session != os.getsid(0)


def test_supervisor_stops_an_overrun_without_resetting_the_deadline():
    result = run_supervised([sys.executable, '-c', 'import time;time.sleep(10)'], time.time() + .1, poll_seconds=.02)
    assert result['timed_out'] and result['returncode'] != 0
    assert result['finished_unix'] - result['started_unix'] < 4


def sample():
    truth = np.array([[1000., .1, .001, .001, .2, .001, .02, .001, .001, .7],
                      [1100., .09, .002, .002, .19, .002, .03, .002, .002, .7],
                      [1200., .08, .003, .003, .18, .003, .04, .003, .003, .7]])
    data = dict(states=np.stack([truth, truth]), dt=np.array(2e-6),
                initials=np.array([[1000., 101325., 1.]] * 2))
    traces = {method: {i: truth.copy() for i in (0, 1)} for method, _, _ in PER_WEIGHT_METHODS}
    return data, traces


def test_failure_scores_and_global_vs_within_team_summaries_for_short_and_long():
    data, traces = sample(); traces['long_ann10'][1] = traces['long_ann10'][1][:2]
    frame = metrics_frame(data, traces, [0, 1], 'repaired', 100.)
    summary, within_summary, common, within = summaries(frame)
    assert common == [0] and within['ann10'] == [0] and within['p_only10'] == [0, 1]
    failed = frame[(frame.method == 'long_ann10') & (frame.trajectory == 1)]
    assert failed.nrmse.isna().all() and failed.full_rmse.isna().all()
    assert set(summary.common_ICs) == {'[0]'}
    assert set(within_summary.loc[within_summary.team == 'p_only10', 'common_ICs']) == {'[0, 1]'}
    fig = table_figure(summary, common, 'repaired', 100.); assert fig.axes[0].tables; plt.close(fig)


def test_7x10_profiles_include_baseline_same_lambda_short_and_long():
    data, _ = sample(); data['states'] = np.repeat(data['states'][:, :1], 5001, axis=1)
    traces = {method: {i: data['states'][i].copy() for i in (0, 1)} for method, _, _ in PER_WEIGHT_METHODS}
    fig = profile_figure(data, traces, 0, 'raw', 1000.)
    assert len(fig.axes) == 70 and all(len(ax.lines) == 4 for ax in fig.axes)
    assert any('lambda=1000' in text.get_text() for text in fig.legends[0].get_texts())
    plt.close(fig)


def test_cross_weight_table_uses_explicit_cohorts_for_all_six_stages():
    import pandas as pd
    data, traces = sample(); frames = []
    for w in (1., 100., 1000.):
        p = metrics_frame(data, traces, [0, 1], 'repaired', w); p = p[p.stage != 'baseline'].copy()
        p['method'] = p['method'].map(lambda m: f'lambda{int(w)}_{m}'); frames.append(p)
    frame = pd.concat(frames, ignore_index=True)
    summary, _, common, within = summaries(frame)
    assert len(frame.method.unique()) == 42 and common == [0, 1] and within['ann10'] == [0, 1]
    fig = table_figure(summary, common, 'repaired'); assert fig.axes[0].tables; plt.close(fig)
