import numpy as np
import pytest
from kinetics.deeponet import ThermoDeepONet, train_member
from kinetics.study import Predictor, tf


def training_case():
    rng = np.random.default_rng(43)
    state = np.c_[rng.uniform(900, 1600, 16), rng.dirichlet(np.ones(9), 16)]
    x = np.c_[state, rng.uniform(1, 5, 16) * 101325]
    y = state.copy()
    y[:, 0] += 20
    y[:, 1:] = .99 * state[:, 1:] + .01 / 9
    return x, y


def config():
    return dict(branch_layers=[64, 64], trunk_layers=[32], latent_width=7,
                learning_rate=.001, batch_size=8, input_partition='current_Y9__current_T_logP')


def test_thermo_model_parameter_count_and_exact_ann_preprocessing():
    x, y = training_case()
    spec = dict(kind='boxcox', gamma=.1)
    ann = Predictor(spec, x, y, dict(hidden_layers=[64, 64, 64], activation='tanh', learning_rate=.001), 11)
    deep = ThermoDeepONet(spec, x, y, config(), 11)
    assert ann.model.count_params() == 9738
    assert deep.model.count_params() == 9687
    np.testing.assert_array_equal(deep.train_features, ann.train_features)
    np.testing.assert_array_equal(deep.train_target, ann.train_target)
    np.testing.assert_array_equal(deep.model(deep.train_features).numpy(), np.zeros((16, 10)))


def test_current_temperature_and_pressure_are_trunk_queries_and_both_networks_learn(tmp_path):
    x, y = training_case()
    cfg = config()
    spec = dict(kind='boxcox', gamma=.1)
    initial = ThermoDeepONet(spec, x, y, cfg, 11)
    before = {layer.name: [w.copy() for w in layer.get_weights()] for layer in initial.model.layers}
    p, _, history = train_member(spec, 11, 4, x, y, cfg)
    for name in ('branch_0', 'thermo_basis'):
        assert any(not np.array_equal(a, b) for a, b in zip(before[name], p.model.get_layer(name).get_weights()))
    same_composition = np.repeat(x[:1], 3, axis=0)
    same_composition[1, 0] += 200
    same_composition[:, -1] = [101325, 101325, 5 * 101325]
    features = p.features(same_composition).astype('float32')
    partition = tf.keras.Model(p.model.input,
        [p.model.get_layer('branch_0').input, p.model.get_layer('trunk_0').input])
    branch, trunk = [v.numpy() for v in partition(features)]
    assert branch.shape == (3, 9) and trunk.shape == (3, 2)
    np.testing.assert_array_equal(branch, features[:, 1:10])
    np.testing.assert_array_equal(branch, np.repeat(branch[:1], 3, axis=0))
    np.testing.assert_array_equal(trunk, features[:, [0, 10]])
    residual = p.model(features).numpy()
    assert not np.allclose(residual[0], residual[1], atol=1e-9)
    assert not np.allclose(residual[0], residual[2], atol=1e-9)
    assert history[-1]['update'] == 4
    path = tmp_path / 'thermo_seed11'
    p.save(path)
    loaded = ThermoDeepONet.load(path)
    np.testing.assert_allclose(loaded.predict(x), p.predict(x), rtol=1e-12, atol=1e-12, equal_nan=True)
    with pytest.raises(FileExistsError):
        p.save(path)


def test_members_have_independent_seeded_training_and_timeout():
    x, y = training_case()
    cfg = {**config(), 'branch_layers': [4], 'trunk_layers': [3], 'latent_width': 2}
    spec = dict(kind='boxcox', gamma=.1)
    first, _, _ = train_member(spec, 11, 3, x, y, cfg)
    other, _, _ = train_member(spec, 22, 3, x, y, cfg)
    repeat, _, _ = train_member(spec, 11, 3, x, y, cfg)
    for a, b in zip(first.model.get_weights(), repeat.model.get_weights()):
        np.testing.assert_array_equal(a, b)
    assert any(not np.array_equal(a, b) for a, b in zip(first.model.get_weights(), other.model.get_weights()))
    with pytest.raises(TimeoutError):
        train_member(spec, 11, 3, x, y, cfg, deadline=0)
