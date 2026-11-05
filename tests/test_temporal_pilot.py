import numpy as np
import pytest

from kinetics.temporal_pilot import (decode, nrmse, valid_starts, recursive_loss,
                                    tf, build_model, PilotPredictor, advance)


def test_horizon_uses_dense_index_and_excludes_only_out_of_bounds():
    parent=np.array([[1,0],[1,17],[2,99]])
    np.testing.assert_array_equal(valid_starts(parent,4,100),[0,1])
    np.testing.assert_array_equal(valid_starts(parent,1,100),[0,1,2])


def test_raw_inverse_domain_is_not_repaired():
    encoded=np.ones((1,10))
    encoded[0,1]=-.01
    assert np.isnan(decode(encoded)[0,1])
    assert decode(encoded,repaired=True)[0,1]==0


def test_util_nrmse_is_ratio_and_requires_matching_complete_profile():
    truth=np.array([[2.,4.],[2.,4.]])
    pred=truth+np.array([1.,2.])
    np.testing.assert_allclose(nrmse(truth,pred),[.5,.5])
    with pytest.raises(ValueError):
        nrmse(truth,pred[:1])


def test_multistep_uses_predictions_with_gradient_through_previous_steps():
    class Scale(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.a=self.add_weight(shape=(),initializer=tf.keras.initializers.Constant(.1))

        def call(self,x,training=False):
            return x[:,:10]*self.a

    model=Scale()
    with tf.GradientTape() as tape:
        loss=recursive_loss(model,tf.ones((1,10)),tf.zeros((1,1)),
            tf.ones((1,2,10)),tf.zeros(11),tf.ones(11))
    # Predicted states 1.1, 1.21; never restart from true 1 at step two.
    assert float(loss.numpy())==pytest.approx((.1**2+.21**2)/2,rel=2e-6)
    grad=tape.gradient(loss,model.a)
    assert float(grad.numpy())==pytest.approx(.1+.21*2*1.1,rel=2e-6)


def test_all_member_raw_mean_does_not_filter_invalid_member():
    predictor=PilotPredictor.__new__(PilotPredictor)
    predictor.mode='raw'
    predictor.models=[None,None]
    predictor.valid_min=2
    candidates=np.ones((2,1,10))
    candidates[0,0,2]=np.nan
    predictor.candidates=lambda x:candidates
    assert not np.isfinite(predictor.predict(np.ones((1,11)))).all()


def test_inference_feeds_previous_prediction_without_reference():
    class Increment:
        def predict(self,x):
            out=x[:,:10].copy()
            out[:,0]+=1
            return out
    initial=np.array([[1000.]+[1/9]*9])
    traces,failures=advance(Increment(),initial,np.array([101325.]),3)
    np.testing.assert_array_equal(traces[0][:,0],[1000.,1001.,1002.,1003.])
    assert failures==['completed']


def test_interval_model_reload_preserves_time_conditioning(tmp_path):
    model=build_model('interval',11)
    weights=model.get_weights()
    # Give the initially zero coefficient head nonzero weights.
    rng=np.random.default_rng(4)
    model.set_weights([rng.normal(0,.02,w.shape).astype(w.dtype) for w in weights])
    x=np.ones((3,12),np.float32)
    x[:,-1]=[-1,0,1]
    before=model(x).numpy()
    assert not np.allclose(before[0],before[2])
    path=tmp_path/'model.keras'
    model.save(path)
    after=tf.keras.models.load_model(path,compile=False)(x).numpy()
    np.testing.assert_array_equal(before,after)
