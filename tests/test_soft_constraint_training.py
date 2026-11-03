import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from kinetics.data import mechanism
from kinetics.physics import element_matrix
from kinetics.study import tf
from kinetics.transforms import Standardizer,Transform
from kinetics.p_only_training import PressureTrainer
from kinetics.soft_constraint_training import (SoftTrainer,SoftObjective,make_predictor,
    signed_boxcox_inverse,FAMILIES)
from test_p_only_deeponet import case


def setup(family,production=False):
    x,y=case()
    cfg=dict(hidden_layers=[64,64,64] if production else [4,4],activation='tanh',
        branch_layers=[64,64] if production else [4],trunk_layers=[32] if production else [3],
        latent_width=7 if production else 2,batch_size=8,learning_rate=.001,
        input_partition={'ann':'T_Y9_logP','p_only':'current_T_Y9__logP',
            'tp':'current_Y9__current_T_logP','tp_phi':'current_Y9__current_T_logP_initial_phi'}[family])
    if family=='tp_phi':x=np.c_[x[:,:10],np.linspace(.5,2,len(x)),x[:,-1]]
    return x,y,cfg,element_matrix(mechanism())


@pytest.mark.parametrize('family,parameters',[('ann',9738),('p_only',9719),('tp',9687),('tp_phi',9719)])
def test_original_production_architecture_and_common_preprocessing(family,parameters):
    x,y,cfg,a=setup(family,True);p=make_predictor(family,x,y,cfg,11)
    assert p.model.count_params()==parameters
    np.testing.assert_array_equal(p.model(p.train_features).numpy(),np.zeros((len(x),10)))
    assert p.model.input_shape[-1]==(12 if family=='tp_phi' else 11)


def test_physical_element_L1_matches_uploaded_reduction_and_excludes_temperature():
    matrix=element_matrix(mechanism())
    p=SimpleNamespace(norm=Standardizer(np.zeros(10),np.ones(10)))
    objective=SoftObjective(p,matrix,10.)
    species=np.array([[.05,.001,.001,.2,.001,.02,.001,.001,.724],
                      [.02,.002,.002,.15,.002,.08,.002,.002,.74]])
    current=np.c_[np.ones(2),Transform('boxcox').forward(species)]
    truth=species.copy();truth[:,0]+=.02
    prediction=tf.zeros((2,10),tf.float32);target=tf.ones((2,10),tf.float32)*.03
    total,values=objective.components(prediction,target,tf.constant(current,tf.float64),tf.constant(truth))
    expected=np.mean(np.sum(np.abs((species-truth)@matrix.T),axis=1))
    assert float(values[2])==pytest.approx(expected,rel=1e-12,abs=1e-14)
    assert float(values[0])==pytest.approx(float(values[1])+10*expected,rel=1e-12)
    changed=tf.concat([tf.ones((2,1),tf.float32)*5,tf.zeros((2,9),tf.float32)],axis=1)
    _,second=objective.components(changed,target,tf.constant(current,tf.float64),tf.constant(truth))
    assert float(values[2])==float(second[2])


def test_signed_loss_inverse_matches_valid_boxcox_and_has_finite_gradients():
    z=np.array([0.,.3,2.,8.,10.])
    np.testing.assert_allclose(signed_boxcox_inverse(z).numpy(),Transform('boxcox').inverse(z),rtol=1e-13,atol=1e-30)
    variable=tf.Variable([-.3,0.,.3],dtype=tf.float64)
    with tf.GradientTape() as tape:loss=tf.reduce_sum(signed_boxcox_inverse(variable))
    gradient=tape.gradient(loss,variable).numpy()
    assert np.isfinite(gradient).all() and gradient[0]>0 and gradient[2]>0 and gradient[1]==0
    assert float(signed_boxcox_inverse(tf.constant(-.3,tf.float64)))<0
    assert np.isnan(Transform('boxcox').inverse([-.3]))[0]


@pytest.mark.parametrize('family',FAMILIES)
def test_mu0_exactly_reproduces_original_scalar_adam_and_minibatch_stream(family):
    x,y,cfg,a=setup(family)
    predictor=make_predictor(family,x,y,cfg,11)
    original=PressureTrainer(x,y,cfg,11,predictor=predictor);original.advance(7)
    soft=SoftTrainer(x,y,cfg,11,family,a,0.);soft.advance(7)
    for first,second in zip(original.predictor.model.get_weights(),soft.predictor.model.get_weights()):
        np.testing.assert_array_equal(first,second)
    for first,second in zip(original.optimizer.variables,soft.optimizer.variables):
        np.testing.assert_array_equal(first.numpy(),second.numpy())
    np.testing.assert_array_equal(original.rng.integers(0,len(x),8,dtype=np.int32),
                                 soft.rng.integers(0,len(x),8,dtype=np.int32))


@pytest.mark.parametrize('family',FAMILIES)
def test_mu10_checkpoint_resume_is_exact_for_weights_adam_and_rng(tmp_path,family):
    x,y,cfg,a=setup(family)
    whole=SoftTrainer(x,y,cfg,11,family,a,10.);whole.advance(8)
    partial=SoftTrainer(x,y,cfg,11,family,a,10.);partial.advance(3);partial.full_train_loss()
    cp=partial.save_checkpoint(tmp_path/family,'interrupted')
    resumed=SoftTrainer.restore(cp,x,y,cfg,11,family,a,10.);resumed.advance(8)
    for first,second in zip(whole.predictor.model.get_weights(),resumed.predictor.model.get_weights()):
        np.testing.assert_array_equal(first,second)
    for first,second in zip(whole.optimizer.variables,resumed.optimizer.variables):
        np.testing.assert_array_equal(first.numpy(),second.numpy())
    np.testing.assert_array_equal(whole.rng.integers(0,len(x),8,dtype=np.int32),
                                 resumed.rng.integers(0,len(x),8,dtype=np.int32))
    assert resumed.update==int(resumed.optimizer.iterations.numpy())==8
    assert resumed.mu==10 and resumed.full_train_history[0]['element_loss']>=0
    with pytest.raises(ValueError,match='provenance'):
        SoftTrainer.restore(cp,x,y,cfg,11,family,a,1.)
    with pytest.raises(FileExistsError):resumed.save_checkpoint(cp,'overwrite')


def test_soft_checkpoint_hash_tampering_is_rejected(tmp_path):
    x,y,cfg,a=setup('tp');trainer=SoftTrainer(x,y,cfg,11,'tp',a,10.);trainer.advance(2)
    cp=trainer.save_checkpoint(tmp_path/'checkpoint','test')
    with (cp/'adam.npz').open('ab') as handle:handle.write(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        SoftTrainer.restore(cp,x,y,cfg,11,'tp',a,10.)


def test_member_runner_records_components_and_fixed_update_reason(tmp_path):
    from kinetics.soft_constraint_study import train_member
    x,y,config,a=setup('tp')
    cfg=dict(observe_every=2,checkpoint_every=2,updates_per_member=4,mu=10.)
    snapshot=dict(model_configs={'tp':config},element_matrix=a.tolist())
    record=train_member(tmp_path,cfg,snapshot,dict(x=x,y=y),{},'tp',11,float('inf'))
    assert record['updates']==record['optimizer_iteration']==4
    assert record['reason']=='fixed_update_budget' and record['mu']==10
    assert set(record['full_train'])>={'data_loss','element_loss','full_train_loss'}
    state=json.loads(Path(record['checkpoint'],'state.json').read_text())
    assert state['signature']['mu']==10 and state['plateau_monitor']['enabled'] is False
