import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest
from kinetics.tp_phi_deeponet import PhiDeepONet
from kinetics.tp_phi_training import PhiTrainer, PhiSerialSteps, new_monitor
from kinetics.tp_phi_evaluation import PhiMean, rollout, evaluate_arm
from kinetics.study import Predictor, tf
from kinetics.p_only_training import sha256
from test_p_only_deeponet import case, config
from test_deeponet_long_training import POLICY


def phi_case(tiny=True):
    x,y=case()
    return np.c_[x[:,:10],np.linspace(.5,2,len(x)),x[:,-1]],y,{**config(tiny),
        'input_partition':'current_Y9__current_T_logP_initial_phi'}


def test_parameter_count_and_exact_common_ann_preprocessing():
    x,y,cfg=phi_case(False)
    p=PhiDeepONet(dict(kind='boxcox',gamma=.1),x,y,cfg,11)
    ann=Predictor(dict(kind='boxcox',gamma=.1),x,y,
        dict(hidden_layers=[64,64,64],activation='tanh',learning_rate=.001),11)
    assert p.model.count_params()==9719
    np.testing.assert_array_equal(p.train_features[:,:11],ann.train_features)
    np.testing.assert_array_equal(p.train_target,ann.train_target)
    np.testing.assert_array_equal(p.model(p.train_features).numpy(),np.zeros((len(x),10)))
    np.testing.assert_allclose(p.phi_norm.mean,[x[:,10].mean()])
    with pytest.raises(ValueError,match='Explicit'):
        p.features(np.c_[x[:,:10],x[:,-1]])


def test_partition_routes_current_T_logP_and_fixed_phi_to_trunk_only():
    x,y,cfg=phi_case()
    t=PhiTrainer(x,y,cfg,11);p=t.predictor
    before=p.model.get_layer('thermo_phi_basis').get_weights()
    t.advance(4)
    assert any(not np.array_equal(a,b) for a,b in zip(before,p.model.get_layer('thermo_phi_basis').get_weights()))
    probe=np.repeat(x[:1],4,axis=0)
    probe[1,0]+=100;probe[2,-1]*=1.5;probe[3,10]+=.3
    f=p.features(probe).astype('float32')
    partition=tf.keras.Model(p.model.input,[p.model.get_layer('branch_0').input,p.model.get_layer('trunk_0').input])
    branch,trunk=[a.numpy() for a in partition(f)]
    np.testing.assert_array_equal(branch,f[:,1:10])
    np.testing.assert_array_equal(trunk,np.c_[f[:,:1],f[:,10:12]])
    for i in (1,2,3):
        np.testing.assert_array_equal(branch[0],branch[i])
        assert not np.array_equal(trunk[0],trunk[i])


def test_scalar_serial_and_adam_rng_resume_are_array_equal(tmp_path):
    x,y,cfg=phi_case()
    a=PhiTrainer(x,y,cfg,11);a.advance(8)
    b=PhiTrainer(x,y,cfg,11);b.plateau_monitor=new_monitor(POLICY)
    engine=PhiSerialSteps(b);engine.advance(3)
    cp=b.save_checkpoint(tmp_path/'checkpoint','interrupted')
    c=PhiTrainer.restore(cp,x,y,cfg,11)
    engine=PhiSerialSteps(c);engine.advance(8)
    for aa,cc in zip(a.predictor.model.get_weights(),c.predictor.model.get_weights()):
        np.testing.assert_array_equal(aa,cc)
    for aa,cc in zip(a.optimizer.variables,c.optimizer.variables):
        np.testing.assert_array_equal(aa.numpy(),cc.numpy())
    np.testing.assert_array_equal(a.rng.integers(0,len(x),8,dtype=np.int32),c.rng.integers(0,len(x),8,dtype=np.int32))
    np.testing.assert_array_equal(a.predictor.predict(x),c.predictor.predict(x))
    np.testing.assert_allclose(a.full_train_loss(),engine.full_train_loss(),rtol=1e-12)
    assert int(c.optimizer.iterations.numpy())==c.update==8
    with pytest.raises(ValueError,match='provenance'):
        PhiTrainer.restore(cp,x,y,{**cfg,'batch_size':16},11)
    with pytest.raises(FileExistsError):
        c.save_checkpoint(cp,'overwrite')


def test_mixed_proposals_use_each_model_same_current_state_and_phi():
    x,y,cfg=phi_case()
    t=PhiTrainer(x,y,cfg,11);t.advance(3)
    ann=Predictor(dict(kind='boxcox',gamma=.1),x,y,
        dict(hidden_layers=[4],activation='tanh',learning_rate=.001),11)
    p=PhiMean([ann,t.predictor],['ann','deeponet'],'raw')
    candidates=p.proposals(x)
    np.testing.assert_array_equal(candidates[0],ann.predict(x))
    np.testing.assert_array_equal(candidates[1],t.predictor.predict(x))


def test_rollout_preserves_initial_phi_when_composition_drifts_and_IC_fails():
    x,y,cfg=phi_case()
    class Fake:
        def __init__(self):self.calls=[]
        def predict(self,x):
            self.calls.append(x.copy());out=x[:,:10].copy();out[:,1:]*=1.2
            if len(self.calls)==1:out[0]=np.nan
            self.last_diagnostics={k:np.ones(len(x),dtype=int) for k in ('valid_count','valid_ann','valid_deeponet')}
            return out
    p=Fake()
    traces,events=rollout(p,x[:2,:10],x[:2,-1],3,2e-6,phis=x[:2,10])
    assert len(traces[0])==1 and len(traces[1])==4
    np.testing.assert_array_equal(p.calls[0][:,10],x[:2,10])
    for call in p.calls[1:]:
        assert call.shape==(1,12) and call[0,10]==x[1,10] and call[0,-1]==x[1,-1]
    assert p.calls[-1][0,1]!=x[1,1]


def test_tiny_arm_writes_full_metrics_and_constant_phi(tmp_path):
    x,y,cfg=phi_case()
    t=PhiTrainer(x,y,cfg,11);t.advance(2)
    reference=np.repeat(x[:2,None,:10],3,axis=1)
    p=PhiMean([t.predictor],['deeponet'])
    data=dict(states=reference,pressures=x[:2,-1],split=np.array(['test','test']),
        initials=np.c_[x[:2,0],x[:2,-1],x[:2,10]],equilibrium=reference[:,0],element_matrix=p.physics.matrix)
    fine=tmp_path/'fine.csv'
    pd.DataFrame(dict(trajectory=[0,1],reference_dt=[2e-7]*2,tau400=[np.nan]*2)).to_csv(fine,index=False)
    fp=dict(reference_ignition={'test':dict(path=str(fine),sha256=sha256(fine))})
    base=dict(dt=2e-6,horizon=4e-6,one_step_chunk_size=2)
    info=evaluate_arm(tmp_path,'phi',[t.predictor],['deeponet'],'repaired','test',data,base,fp,float('inf'))
    marker=next((tmp_path/'evaluations').glob('*/*/complete.json'))
    for name,h in info['output_sha256'].items():assert sha256(marker.parent/name)==h
    frame=pd.read_csv(marker.parent/'trajectories.csv')
    assert frame.loc[np.isclose(frame.horizon,4e-6),'complete'].all()
    assert json.loads((marker.parent/'one_step.json').read_text())['pair_count']==4


def test_member_runner_keeps_matched_endpoint_and_resumes_long(tmp_path):
    from kinetics.tp_phi_study import train_member
    x,y,model=phi_case()
    cfg=dict(matched_updates=3,checkpoint_every=1000,plateau=POLICY)
    snapshot=dict(model_config=model)
    pair=dict(x=x,y=y)
    a=train_member(tmp_path,cfg,snapshot,pair,11,'matched',float('inf'))
    assert a['updates']==3 and a['reason']=='matched_update_budget'
    b=train_member(tmp_path,cfg,snapshot,pair,11,'long',time.perf_counter()-1)
    assert b['updates']==3 and b['reason']=='time_limit'
    state=json.loads(Path(b['checkpoint'],'state.json').read_text())
    assert state['rng_state']==json.loads(Path(a['checkpoint'],'state.json').read_text())['rng_state']
    assert state['optimizer_iteration']==3
    assert Path(a['checkpoint'],'complete.json').is_file()
