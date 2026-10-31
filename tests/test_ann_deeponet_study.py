import json
import numpy as np
import pandas as pd
import pytest
from kinetics import ann_deeponet_study as study
from kinetics.data import SPECIES
from kinetics.mixed_ensemble import MixedMean
from test_mixed_ensemble import member, state


def test_pair_audit_rejects_split_leakage_and_skipped_targets():
    states = np.repeat(state()[None, ...], 2, axis=0)
    states = np.repeat(states, 4, axis=1)
    states[:, :, 0] += np.arange(4)
    data = dict(states=states, pressures=np.array([101325., 101325.]), dt=2e-6,
                species=np.array(SPECIES), split=np.array(['train', 'test']))
    pairs = dict(x=np.c_[states[0, :2], [101325., 101325.]], y=states[0, 1:3],
                 parent=np.array([[0, 0], [0, 1]]), dt=2e-6)
    study.validate_pairs(data, pairs, 2e-6, 6e-6)
    with pytest.raises(ValueError, match='non-training'):
        study.validate_pairs(data, {**pairs, 'parent': np.array([[1, 0], [1, 1]])}, 2e-6, 6e-6)
    with pytest.raises(AssertionError):
        study.validate_pairs(data, {**pairs, 'y': states[0, 2:4]}, 2e-6, 6e-6)


def test_training_never_starts_without_budget():
    with pytest.raises(SystemExit) as error:
        study.main(['--phase', 'train'])
    assert error.value.code == 2


def test_new_output_namespace_and_resume_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(study.subprocess, 'check_output', lambda *a, **k: 'test-head')
    cfg = dict(output='results/ann_deeponet_mean10/test')
    root = study.prepare_run(cfg, {'data': 'original'})
    with pytest.raises(FileExistsError):
        study.prepare_run(cfg, {'data': 'original'})
    assert study.prepare_run(cfg, {'data': 'original'}, True) == root
    with pytest.raises(ValueError, match='changed'):
        study.prepare_run(cfg, {'data': 'different'}, True)
    with pytest.raises(ValueError, match='preserve'):
        study.prepare_run(dict(output='results/committee100_long10ms'), {})


def test_raw_invalid_diagnostics_are_not_normalized():
    p = MixedMean([member()], ['ann'])
    trace = np.repeat(state(), 3, axis=0)
    trace[1, 1:] *= 2
    info = study.physical_diagnostics(trace, trace.copy(), 101325., trace[0], p.physics, float('inf'))
    assert info['thermo_invalid_points'] == 1
    assert info['entropy_valid_step_count'] == 0


def test_small_evaluation_runs_three_independent_teams_in_both_modes(tmp_path, monkeypatch):
    cfg = dict(seeds=list(range(10)), dt=2e-6, horizon=6e-6,
               modes=['raw', 'repaired'], one_step_chunk_size=4)
    q = np.repeat(state()[None, ...], 2, axis=0)
    q = np.repeat(q, 4, axis=1)
    data = dict(states=q, pressures=np.array([101325., 101325.]), split=np.array(['val', 'val']),
                element_matrix=study.MixedMean([member()], ['ann']).physics.matrix,
                equilibrium=q[:, 0], dt=2e-6)
    fine = tmp_path / 'reference.csv'
    pd.DataFrame(dict(trajectory=[0, 1], reference_dt=[2e-7] * 2, tau400=[np.nan] * 2)).to_csv(fine, index=False)
    records = [dict(seed=i, path='test', group_seconds=1., group_size=1) for i in range(10)]
    fingerprint = dict(ann_members=records, reference_ignition=dict(val=dict(path=str(fine), sha256=study.sha256(fine))))
    monkeypatch.setattr(study, 'registered_members', lambda *a: records)
    p = member()
    monkeypatch.setattr(study.Predictor, 'load', lambda *a: p)
    monkeypatch.setattr(study.ThermoDeepONet, 'load', lambda *a: p)
    study.evaluate(tmp_path, cfg, data, fingerprint, 'val', float('inf'))
    directories = list((tmp_path / 'evaluations').iterdir())
    assert len(directories) == 6
    for directory in directories:
        info = json.loads((directory / 'complete.json').read_text())
        assert info['member_count'] == (20 if info['method'] == 'mixed20' else 10)
        one = json.loads((directory / 'one_step.json').read_text())
        assert one['pair_count'] == 6
        events = pd.read_csv(directory / 'events.csv')
        assert events.valid_count.eq(info['member_count']).all()
        with np.load(directory / 'traces.npz') as f:
            np.testing.assert_allclose(f['trajectory_0'], q[0], atol=1e-8)
    # Execute the real reporting/plotting path on these tiny traces, not a full
    # research experiment. It must label the incomplete 2ms metrics as NaN.
    from kinetics.ann_deeponet_report import report
    study.write_json(tmp_path / 'run.json', dict(fingerprint=fingerprint,
                                              config=dict(updates_per_member=10000)))
    study.write_json(tmp_path / 'deeponet_members.json', [dict(parameters=9687, train_seconds=1.)] * 10)
    result = report(tmp_path, data)
    summary = pd.read_csv(result / 'summary.csv')
    assert len(summary) == 12
    assert summary[summary.horizon == .002].temperature_rmse_completed_only.isna().all()
    assert summary[summary.horizon == 6e-6].completion_fraction.eq(1).all()
    assert len(list(result.glob('*.png'))) == 20
    assert (result / 'complete.json').exists()
    # Resumption must not silently trust a modified completed evaluation.
    (directories[0] / 'one_step.json').write_text('{}')
    with pytest.raises(ValueError, match='output changed'):
        study.evaluate(tmp_path, cfg, data, fingerprint, 'val', float('inf'))
