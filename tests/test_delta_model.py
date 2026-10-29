import numpy as np
from kinetics.delta_model import DeltaPredictor, DeltaCommittee
from kinetics.transforms import Transform


def training_data():
    rng = np.random.default_rng(9)
    x = np.c_[rng.uniform(900, 1600, 20), rng.dirichlet(np.ones(9), 20), np.full(20, 101325.)]
    delta = np.c_[np.linspace(-1, 10, 20), rng.uniform(-.001, .001, (20, 9))]
    return x, x[:, :10]+delta


def test_signed_boxcox_delta_roundtrip_and_not_difference_of_transforms():
    t = Transform('boxcox', .1, signed=True)
    values = np.array([-1., -1e-30, -1e-300, 0., 1e-300, 1e-30, 1.])
    actual = t.inverse(t.forward(values))
    np.testing.assert_allclose(actual, values, atol=0, rtol=1e-12)
    assert t.forward(np.array([-.001]))[0] < 0
    assert not np.allclose(t.forward(np.array([.1]))-t.forward(np.array([.2])), t.forward(np.array([-.1])))


def test_delta_training_target_and_serialization(tmp_path):
    x, y = training_data()
    cfg = dict(hidden_layers=[4], activation='tanh')
    p = DeltaPredictor(dict(gamma=.1), x, y, cfg, 11)
    expected = p.output_norm.forward(p.encode_delta(y-x[:, :10]))
    np.testing.assert_allclose(p.train_target, expected, rtol=1e-6, atol=1e-6)
    physical_delta = p.decode(expected)
    np.testing.assert_allclose(physical_delta, y-x[:, :10], atol=1e-14)
    path = tmp_path/'delta'; p.save(path); q = DeltaPredictor.load(path)
    np.testing.assert_array_equal(p.predict(x), q.predict(x))


def test_unconstrained_single_and_mean_preserve_negative_candidates():
    # Supply deterministic proposals to test the runtime aggregation itself.
    p = DeltaCommittee.__new__(DeltaCommittee)
    p.seeds = np.array([11, 22]); p.rule = 'mean'
    c = np.ones((2, 1, 10)); c[:, :, 0] = 1000; c[:, :, 1] = -.01
    p.candidates = lambda x: c
    result = p.predict(np.zeros((1, 11)))
    assert result[0, 1] == -.01
    assert result[0, 1:].sum() != 1
    assert p.last_diagnostics['valid_count'][0] == 2


def test_raw_rollout_keeps_finite_nonphysical_states_and_records_violation():
    from kinetics.delta_evaluation import raw_rollouts
    class Dummy:
        def predict(self, x):
            self.last_diagnostics = dict(selected_seed=np.full(len(x), 11), valid_count=np.ones(len(x)))
            out = x[:, :10].copy(); out[:, 0] += 10; out[:, 1] -= .2
            return out
    initial = np.r_[1000., np.full(9, 1/9)]
    truth = np.tile(initial, (1, 4, 1))
    rows, _, _, traces = raw_rollouts(Dummy(), truth, np.array([101325.]), np.array([7]), np.ones((1, 9)), 2e-6)
    assert len(traces[0]) == 4
    np.testing.assert_allclose(traces[0][:, 0], [1000, 1010, 1020, 1030])
    assert traces[0][-1, 1] < 0
    assert rows[0]['complete'] and not rows[0]['physically_admissible_complete']
    assert rows[0]['negative_beyond_tolerance_fraction'] == .75


def test_raw_numerical_failure_does_not_become_a_successful_or_clipped_step():
    from kinetics.delta_evaluation import raw_rollouts
    class Explosive:
        def predict(self, x):
            self.last_diagnostics = dict(selected_seed=np.full(len(x), 11), valid_count=np.ones(len(x)))
            out = x[:, :10].copy(); out[:, 0] = 1e7
            return out
    initial = np.r_[1000., np.full(9, 1/9)]
    truth = np.tile(initial, (1, 4, 1))
    rows, _, events, traces = raw_rollouts(Explosive(), truth, np.array([101325.]), np.array([7]), np.ones((1, 9)), 2e-6)
    np.testing.assert_array_equal(traces[0], initial[None])
    assert rows[0]['status'] == 'numeric_magnitude_guard'
    assert not rows[0]['complete'] and not rows[0]['prediction_ignites']
    assert events[0]['temperature'] == 1e7 and not events[0]['accepted']


def test_ignition_crossing_in_first_invalid_step_is_not_physically_admissible():
    from kinetics.delta_evaluation import raw_rollouts
    class InvalidIgnition:
        def predict(self, x):
            self.last_diagnostics = dict(selected_seed=np.full(len(x), 11), valid_count=np.ones(len(x)))
            out = x[:, :10].copy(); out[:, 0] += 1000; out[:, 1] = -.01
            return out
    truth = np.tile(np.r_[1000., np.full(9, 1/9)], (1, 3, 1))
    rows, _, _, _ = raw_rollouts(InvalidIgnition(), truth, np.array([101325.]), np.array([7]), np.ones((1, 9)), 2e-6)
    assert rows[0]['prediction_ignites']
    assert not rows[0]['ignition_physically_admissible']
