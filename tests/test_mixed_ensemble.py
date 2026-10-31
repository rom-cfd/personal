from types import SimpleNamespace
import numpy as np
import pytest
from kinetics.mixed_ensemble import MixedMean, rollout
from kinetics.study import Predictor, tf
from kinetics.transforms import Standardizer, Transform


def member(a=0., b=0.):
    inputs = tf.keras.Input((11,))
    layer = tf.keras.layers.Dense(10)
    model = tf.keras.Model(inputs, layer(inputs))
    weights = np.zeros((11, 10), np.float32)
    weights[0, 0] = a / 1000
    bias = np.zeros(10, np.float32)
    bias[0] = b / 1000
    layer.set_weights([weights, bias])
    return SimpleNamespace(model=model, spec=dict(kind='boxcox'), transform=Transform('boxcox', 1.),
        norm=Standardizer(np.zeros(10), np.ones(10)),
        input_norm=Standardizer(np.zeros(11), np.ones(11)), features=lambda x: x,
        encode_state=lambda q: np.c_[q[:, 0] / 1000, q[:, 1:]])


def state():
    return np.array([[1000., *([1 / 9] * 9)]])


def test_mixed_team_has_its_own_self_fed_state_not_average_of_separate_rollouts():
    ann, deep = member(b=10), member(a=.1)
    a = MixedMean([ann], ['ann'], 'raw')
    d = MixedMean([deep], ['deeponet'], 'raw')
    m = MixedMean([ann, deep], ['ann', 'deeponet'], 'raw')
    pressure = np.array([101325.])
    traces = [rollout(p, state(), pressure, 2, 2e-6)[0][0] for p in (a, d, m)]
    np.testing.assert_allclose(traces[2][:, 0], [1000, 1055, 1112.75], atol=1e-4)
    assert not np.isclose(traces[2][-1, 0], (traces[0][-1, 0] + traces[1][-1, 0]) / 2)
    with pytest.raises(TimeoutError):
        rollout(m, state(), pressure, 2, 2e-6, deadline=0)


def test_twenty_candidates_have_equal_weight_with_family_counts_and_whole_row_masking():
    base = member()
    p = MixedMean([base] * 20, ['ann'] * 10 + ['deeponet'] * 10)
    candidates = np.repeat(state()[None, ...], 20, axis=0)
    candidates[:10, :, 0] = 1000
    candidates[10:, :, 0] = 2000
    p.proposals = lambda x: candidates.copy()
    p.physics = SimpleNamespace(repair=lambda proposals, x: proposals)
    x = np.c_[state(), [101325.]]
    np.testing.assert_allclose(p.predict(x)[0, 0], 1500)
    # Nine ANN candidates have one invalid component. Their other coordinates
    # must not leak into the sum, nor should the two family means get 50/50 weight.
    candidates[:9, :, 1] = np.nan
    np.testing.assert_allclose(p.predict(x)[0, 0], (1000 + 10 * 2000) / 11)
    assert p.last_diagnostics['valid_ann'][0] == 1
    assert p.last_diagnostics['valid_deeponet'][0] == 10
    p.mode = 'raw'
    assert np.isnan(p.predict(x)).all()
    p.mode = 'repaired'
    candidates[:] = np.nan
    assert np.isnan(p.predict(x)).all()
    assert p.last_diagnostics['valid_count'][0] == 0


def test_shared_preprocessing_required():
    a, b = member(), member()
    b.norm.mean[0] = 1
    with pytest.raises(AssertionError):
        MixedMean([a, b], ['ann', 'deeponet'])


def test_repaired_ann_matches_legacy_committee(tmp_path):
    from kinetics.committee import Committee, fit_metric
    rng = np.random.default_rng(9)
    q = np.c_[rng.uniform(1000, 1300, 8), rng.dirichlet(np.ones(9), 8)]
    x = np.c_[q, np.full(8, 101325.)]
    cfg = dict(hidden_layers=[4], activation='tanh', learning_rate=.001)
    paths, models = [], []
    for seed in (11, 22):
        p = Predictor(dict(kind='boxcox', gamma=.1), x, q, cfg, seed)
        weights = p.model.get_weights()
        weights[-1] = rng.uniform(-.001, .001, 10).astype('float32')
        p.model.set_weights(weights)
        path = tmp_path / str(seed)
        p.save(path)
        paths.append(path)
        models.append(p)
    legacy = Committee(paths, [11, 22], fit_metric(q), rule='mean_after_repair',
                       collect_mean_diagnostics=False, physics_backend='vectorized_nasa')
    mixed = MixedMean(models, ['ann', 'ann'])
    actual, expected = mixed.predict(x), legacy.predict(x)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)
