import json
from pathlib import Path
import time

import numpy as np
import pytest
import matplotlib.pyplot as plt
from kinetics.gamma_sweep_training import GammaTrainer, make_predictor, weights_hash
from kinetics.gamma_sweep_study import validate_config, deadlines, member_deadline
from kinetics.gamma_sweep_evaluation import validation_choices
from kinetics.soft_constraint_training import FAMILIES, LOADERS
from kinetics.p_only_training import PressureTrainer
from kinetics.transforms import Standardizer, Transform
from kinetics.mixed_ensemble import require_shared_preprocessing
from scripts.report_gamma_sweep import GAMMAS, METHODS, normalized_metrics, profile_figure, table_figure, selected_comparison, training_outputs
from scripts.nrmse_util_reference import _nrmse_1d
from test_soft_constraint_training import setup


@pytest.mark.parametrize('family', FAMILIES)
def test_all_requested_gammas_refit_exact_train_only_scales_and_keep_initial_weights(family):
    x, y, cfg, _ = setup(family)
    initial = None
    for gamma in GAMMAS:
        p = make_predictor(family, gamma, x, y, cfg, 11)
        assert p.transform.gamma == p.spec['gamma'] == gamma
        a = np.c_[x[:, 0]/1000, x[:, 1:10]**gamma/gamma]
        b = np.c_[y[:, 0]/1000, y[:, 1:10]**gamma/gamma]
        expected = Standardizer().fit(np.r_[a, b])
        inputs = Standardizer().fit(np.c_[expected.forward(a), np.log(x[:, -1])])
        for actual, wanted in ((p.norm.mean, expected.mean), (p.norm.scale, expected.scale),
                               (p.input_norm.mean, inputs.mean), (p.input_norm.scale, inputs.scale)):
            np.testing.assert_array_equal(actual, wanted)
        np.testing.assert_array_equal(p.train_target, (expected.forward(b)-expected.forward(a)).astype('float32'))
        if initial is None:
            initial = weights_hash(p)
        assert weights_hash(p) == initial
        if family == 'tp_phi':
            np.testing.assert_array_equal(p.phi_norm.mean, x[:, 10:11].mean(axis=0))
        if gamma == .1:
            old = LOADERS[family](dict(kind='boxcox', gamma=.1, epsilon=1e-8), x, y, cfg, 11)
            np.testing.assert_array_equal(old.train_features, p.train_features)
            assert weights_hash(old) == weights_hash(p)


@pytest.mark.parametrize('family,parameters', [('ann',9738),('p_only',9719),('tp',9687),('tp_phi',9719)])
def test_production_architectures_unchanged_for_linear_species_control(family, parameters):
    x, y, cfg, _ = setup(family, production=True)
    p = make_predictor(family, 1., x, y, cfg, 11)
    assert p.model.count_params() == parameters
    np.testing.assert_array_equal(p.transform.forward(x[:, 1:10]), x[:, 1:10])
    np.testing.assert_array_equal(p.model(p.train_features).numpy(), np.zeros((len(x), 10)))


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('gamma', [.05, 1.])
def test_exact_adam_rng_history_continuation_and_saved_gamma(tmp_path, family, gamma):
    x, y, cfg, _ = setup(family)
    whole = GammaTrainer(x, y, cfg, 11, family, gamma); whole.advance(5)
    part = GammaTrainer(x, y, cfg, 11, family, gamma); part.advance(2); part.full_train_loss()
    cp = part.save_checkpoint(tmp_path / f'{family}_{gamma}', 'test')
    restored = GammaTrainer.restore(cp, x, y, cfg, 11, family, gamma); restored.advance(5)
    for a, b in zip(whole.predictor.model.get_weights(), restored.predictor.model.get_weights()):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(whole.optimizer.variables, restored.optimizer.variables):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    assert whole.rng.bit_generator.state == restored.rng.bit_generator.state
    assert whole.history == restored.history
    assert restored.predictor.transform.gamma == gamma
    assert restored.signature['loss'] == 'transformed_residual_MSE_only'
    assert not restored.plateau_monitor['enabled']
    assert np.isfinite(restored.full_train_loss())
    with pytest.raises(ValueError, match='provenance'):
        GammaTrainer.restore(cp, x, y, cfg, 11, family, .3)


@pytest.mark.parametrize('family', FAMILIES)
def test_pure_mse_matches_original_optimizer_without_soft_objective(monkeypatch, family):
    import kinetics.soft_constraint_training as soft
    def forbidden(*a, **k):
        raise AssertionError('Physical penalty must not enter this experiment')
    monkeypatch.setattr(soft, 'SoftObjective', forbidden)
    x, y, cfg, _ = setup(family)
    p = make_predictor(family, .3, x, y, cfg, 11)
    old = PressureTrainer(x, y, cfg, 11, predictor=p); old.advance(4)
    new = GammaTrainer(x, y, cfg, 11, family, .3); new.advance(4)
    for a, b in zip(old.predictor.model.get_weights(), new.predictor.model.get_weights()):
        np.testing.assert_array_equal(a, b)
    assert old.rng.bit_generator.state == new.rng.bit_generator.state


def test_same_gamma_ensembles_share_preprocessing_and_cross_gamma_is_rejected():
    x, y, cfg, _ = setup('ann')
    a = make_predictor('ann', .2, x, y, cfg, 11)
    px, py, pcfg, _ = setup('p_only')
    b = make_predictor('p_only', .2, px, py, pcfg, 11)
    require_shared_preprocessing([a, b])
    c = make_predictor('p_only', .3, px, py, pcfg, 11)
    with pytest.raises(ValueError, match='same transformation'):
        require_shared_preprocessing([a, c])


def example_profiles():
    truth = np.array([[1000., .1,.001,.001,.2,.001,.02,.001,.001,.675],
                      [1100., .09,.002,.002,.19,.002,.03,.002,.002,.68],
                      [1200., .08,.003,.003,.18,.003,.04,.003,.003,.685]])
    truth[:, -1] = .7
    data = dict(states=np.stack([truth]*3), split=np.array(['val','val','test']), dt=np.array(2e-6), initials=np.array([[1000.,101325.,1.]]*3))
    traces = {m: {i: truth*np.r_[1.1,np.ones(9)] for i in (0,1)} for m, _, _ in METHODS}
    return data, traces


def test_metric_util_and_incomplete_common_cohorts():
    data, traces = example_profiles()
    traces['gamma_1_ann10'][1] = traces['gamma_1_ann10'][1][:2]
    frame, summary, within, common, sets = normalized_metrics(data, traces, [0,1], 'val', 'repaired')
    assert common == [0] and sets['ann10'] == [0] and sets['p_only10'] == [0,1]
    row = frame[(frame.method == 'gamma_0.2_ann10') & (frame.trajectory == 0) & (frame.variable == 'T')].iloc[0]
    assert row.nrmse == _nrmse_1d(data['states'][0,:,0], traces['gamma_0.2_ann10'][0][:,0])
    failed = frame[(frame.method == 'gamma_1_ann10') & (frame.trajectory == 1)]
    assert failed.nrmse.isna().all() and failed.full_rmse.isna().all()
    assert (frame.loc[frame.variable == 'N2', 'nrmse'].dropna() == 0).all()
    traces['gamma_0.05_p_only10'][0] = traces['gamma_0.05_p_only10'][0][:2]
    _, summary, _, common, _ = normalized_metrics(data, traces, [0,1], 'val', 'raw')
    assert common == [] and summary.median_nrmse.isna().all()
    fig = table_figure(summary, common, 'val', 'raw'); assert fig.axes[0].tables; plt.close(fig)


def test_validation_selection_uses_completion_then_paired_temperature_and_no_test_data():
    data, traces = example_profiles()
    for method, _, gamma in METHODS:
        if gamma == .2:
            traces[method] = {i:data['states'][i]*np.r_[1.01,np.ones(9)] for i in (0,1)}
        if gamma == .05:
            traces[method] = {i:data['states'][i][:2] for i in (0,1)}
    chosen = validation_choices(data, traces, list(GAMMAS))
    assert all(r['chosen_gamma'] == .2 and r['common_ICs'] == [0,1] for r in chosen['choices'].values())
    data['states'][2] *= 100
    assert validation_choices(data, traces, list(GAMMAS)) == chosen
    frame, *_ = normalized_metrics(data, traces, [0,1], 'val', 'repaired')
    selected = selected_comparison(frame, chosen)
    assert len(selected) == 70 and set(selected.common_IC_count) == {2}


def test_profiles_have_seven_rows_ten_variables_and_seven_curves():
    data, _ = example_profiles(); data['states'] = np.repeat(data['states'][:,:1], 5001, axis=1)
    traces = {m:{i:data['states'][i].copy() for i in (0,1)} for m, _, _ in METHODS}
    fig = profile_figure(data, traces, 0, 'val', 'repaired')
    assert len(fig.axes) == 70 and all(len(ax.lines) == 7 for ax in fig.axes)
    assert all(ax.get_xlim() == (0.,10.) for ax in fig.axes)
    plt.close(fig)


def test_training_report_accepts_seeded_observations_and_short_endpoints(tmp_path):
    cp=tmp_path/'checkpoint';cp.mkdir()
    (cp/'state.json').write_text(json.dumps({'full_train_history':[
        {'seed':11,'update':0,'full_train_loss':2.},
        {'seed':11,'update':1000,'full_train_loss':1.}]}))
    output=tmp_path/'report';output.mkdir()
    training_outputs(tmp_path,output,[dict(checkpoint=str(cp),gamma=.2,family='ann',seed=11,
        updates=1000,reason='member_time_limit',reused=False)])
    import pandas as pd
    history=pd.read_csv(output/'observed_full_train_MSE.csv')
    np.testing.assert_array_equal(history.loss_ratio_to_initial,[1.,.5])
    assert (output/'training_loss_ratios.pdf').is_file()


def test_fixed_scope_and_absolute_budget_do_not_reset(tmp_path):
    cfg = json.loads(Path('configs/gamma_sweep.json').read_text()); validate_config(cfg)
    with pytest.raises(ValueError):
        validate_config({**cfg, 'loss':'soft_constraint'})
    with pytest.raises(ValueError):
        validate_config({**cfg, 'gammas':[0.,.1,.2]})
    now = time.time()
    budget = dict(seconds=cfg['total_budget_seconds'], expires_unix=now-1, training_expires_unix=now-2)
    p = tmp_path/'budget.json'; p.write_text(json.dumps(budget)); before=p.read_bytes()
    a,b = deadlines(tmp_path,cfg); assert a <= time.perf_counter() and b <= time.perf_counter()
    assert p.read_bytes() == before


def test_each_model_gets_150_seconds_and_a_resume_cannot_refresh_it(tmp_path):
    cfg = json.loads(Path('configs/gamma_sweep.json').read_text())
    a = tmp_path/'seed11'; b = tmp_path/'seed22'
    limit_a, first = member_deadline(a, cfg)
    limit_b, second = member_deadline(b, cfg)
    assert first['seconds'] == second['seconds'] == 150
    assert 149 <= limit_a-time.perf_counter() <= 150
    assert 149 <= limit_b-time.perf_counter() <= 150
    clock = a/'budget.json'; before = clock.read_bytes()
    _, restored = member_deadline(a, cfg)
    assert restored == first and clock.read_bytes() == before
    expired = {**first,'expires_unix':time.time()-1}; clock.write_text(json.dumps(expired))
    limit, after = member_deadline(a,cfg)
    assert limit <= time.perf_counter() and after == expired


def test_supervisor_expired_clock_stays_expired_without_importing_training(tmp_path):
    import os
    import subprocess
    import sys
    root = tmp_path/'run'; root.mkdir(); (root/'run.json').write_text('{}')
    attempt = root/'process_launches'/'attempt_test'; attempt.mkdir(parents=True)
    cfg = dict(output=str(root),total_budget_seconds=21600,training_budget_seconds=10800)
    config = tmp_path/'config.json'; config.write_text(json.dumps(cfg))
    budget = root/'budget.json'
    budget.write_text(json.dumps(dict(seconds=21600,started_unix=time.time()-21601,
        expires_unix=time.time()-1,training_expires_unix=time.time()-3601)))
    before = budget.read_bytes()
    env = dict(os.environ,TF_ENABLE_ONEDNN_OPTS='0',CUDA_VISIBLE_DEVICES='-1',
               OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result = subprocess.run([sys.executable,'-m','scripts.run_gamma_sweep_supervisor',
        '--config',str(config),'--attempt',str(attempt)],capture_output=True,text=True,env=env,timeout=10)
    assert result.returncode == 124 and budget.read_bytes() == before
    status = json.loads((attempt/'exit.json').read_text())
    assert status['timed_out'] and status['returncode'] == 124
    assert 'tensorflow' not in result.stderr.lower()
