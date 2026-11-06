"""TF physics-in-the-loop step == numpy WangHardCorrection rollout step (guard on)."""
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path('results/zhang_5p3m_gelu_ens/g0.1')
SOURCE = Path('results/long10ms_data/trajectories.npz')


@pytest.mark.skipif(not (ROOT / 'seed11/model.keras').exists() or not SOURCE.exists(), reason='needs a trained parent')
def test_tf_step_matches_numpy_rollout_step():
    import tensorflow as tf
    from kinetics.zhang_model import ZhangCommittee
    from kinetics.zhang_wang import WangHardCorrection
    from kinetics.tf_physics import TFWangPhysics
    info = json.loads((ROOT / 'preprocessing/preprocessing.json').read_text())
    metric = json.loads((ROOT / 'preprocessing/metric.json').read_text())['metric']
    with np.load(SOURCE) as d:
        S, P = d['states'], d['pressures']
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(S), 3000); j = rng.integers(0, S.shape[1], 3000)
    x = np.c_[S[i, j], P[i]]
    wang = WangHardCorrection(ZhangCommittee([str(ROOT / 'seed11/model')], [11], metric, 'single'))
    ref = wang.predict_full(x)
    base = wang.predictor.base
    xs = base.features(x).astype('float32')
    phys = TFWangPhysics(info)
    z = phys.encode(tf.constant(x))
    np.testing.assert_allclose(z.numpy(), base.encode(x), rtol=1e-12, atol=1e-12)
    h0 = phys.enthalpy(tf.constant(x[:, 0]), tf.constant(x[:, 1:10]))
    pred = wang.predictor.forward(xs)[0]   # the exact output the numpy rollout corrected (same kernel, same rounding)
    zn, ok, diag = phys.step(z, pred, h0)
    ok = ok.numpy(); finite = np.isfinite(ref).all(axis=1)
    assert (ok == finite).mean() > 0.999, (ok.sum(), finite.sum())
    both = ok & finite
    assert both.sum() > 2000
    np.testing.assert_allclose(diag['y'].numpy()[both], ref[both, 1:10], atol=2e-9)
    np.testing.assert_allclose(diag['t'].numpy()[both], ref[both, 0], atol=2e-6)
    # the carried float64 state decodes back to the corrected state, temperature included
    tb, _, yb = phys.decode(zn)
    np.testing.assert_allclose(yb.numpy()[both], ref[both, 1:10], rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(tb.numpy()[both], ref[both, 0], atol=2e-6)
    # gradients through two corrected steps are finite
    zv = z[:256]
    with tf.GradientTape() as tape:
        p = base.model(phys.features(zv), training=True)
        z1, okk, _ = phys.step(zv, p, h0[:256])
        p1 = base.model(phys.features(z1), training=True)
        z2, _, _ = phys.step(z1, p1, h0[:256])
        loss = tf.reduce_mean(tf.abs(base.model(phys.features(z2), training=True)))
    g = tape.gradient(loss, base.model.trainable_variables)
    assert all(np.isfinite(v.numpy()).all() for v in g)
