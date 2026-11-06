import json
from pathlib import Path

import numpy as np
import pytest

from kinetics.ann_multistep_study import (build_model,recursive_loss,nrmse,
                                       prepare_training,train_one,tf)
from kinetics.ann_multistep_extend import continue_one


def test_actual_architecture_is_ordinary_ANN_and_same_initialization():
    a=build_model('single_step',11)
    aw=[w.copy() for w in a.get_weights()]
    b=build_model('multistep',11)
    assert a.input_shape==(None,11) and a.output_shape==(None,10)
    assert a.count_params()==b.count_params()==9738
    assert [l.units for l in a.layers if isinstance(l,tf.keras.layers.Dense)]==[64,64,64,10]
    assert all(type(l).__name__ in ['InputLayer','Dense'] for l in a.layers)
    for x,y in zip(aw,b.get_weights()):np.testing.assert_array_equal(x,y)


def test_ANN_multistep_gradient_uses_predicted_state_at_later_steps():
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
    assert float(loss.numpy())==pytest.approx((.1**2+.21**2)/2,rel=2e-6)
    assert float(tape.gradient(loss,model.a).numpy())==pytest.approx(.1+.21*2*1.1,rel=2e-6)


def test_ANN_metric_is_user_RMSE_over_reference_RMS():
    reference=np.array([[2.,4.],[2.,4.]])
    np.testing.assert_allclose(nrmse(reference,reference+np.array([1.,2.])),[.5,.5])
    with pytest.raises(ValueError):nrmse(reference,reference[:1])


@pytest.mark.parametrize('method',['single_step','multistep'])
def test_ANN_checkpoint_continues_exact_Adam_state(tmp_path,method):
    prep=prepare_training('results/long10ms_data/trajectories.npz','results/long10ms_data/training_pairs.npz')
    short=tmp_path/'short';short.mkdir()
    old=train_one(method,11,prep,short,.001)
    long=tmp_path/'long';long.mkdir()
    new=continue_one(old,short,long,prep,1.)
    assert old['updates']==new['predecessor_updates']==1
    assert new['added_updates']>=1 and new['updates']==1+new['added_updates']
    assert new['optimizer_restored_exactly']
    assert new['parameters']==9738
    loaded=tf.keras.models.load_model(long/'models'/f'{method}_seed11'/'model.keras',compile=True)
    assert int(loaded.optimizer.iterations.numpy())==new['updates']
    assert loaded.input_shape[-1]==11
