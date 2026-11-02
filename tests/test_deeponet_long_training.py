import numpy as np
import pytest
import json
from pathlib import Path
import pandas as pd
from kinetics.deeponet_long_training import (SerialSteps,check_plateau,new_monitor,
                                          restore_trainer,fresh_tp)
from kinetics.p_only_training import PressureTrainer
from test_p_only_deeponet import case,config


POLICY = dict(minimum_updates=50000,check_every=5000,window_updates=20000,
              observe_every=1000,relative_thresholds=[.01,.01,.03],sustain_updates=10000)


def test_serial_block_preserves_scalar_adam_weights_and_minibatch_stream():
    x,y = case();cfg=config()
    scalar=PressureTrainer(x,y,cfg,11);scalar.advance(7)
    fast=PressureTrainer(x,y,cfg,11);fast.plateau_monitor=new_monitor(POLICY)
    engine=SerialSteps(fast);engine.advance(7)
    for a,b in zip(scalar.predictor.model.get_weights(),fast.predictor.model.get_weights()):
        np.testing.assert_array_equal(a,b)
    for a,b in zip(scalar.optimizer.variables,fast.optimizer.variables):
        np.testing.assert_array_equal(a.numpy(),b.numpy())
    np.testing.assert_array_equal(scalar.rng.integers(0,len(x),8,dtype=np.int32),
                                  fast.rng.integers(0,len(x),8,dtype=np.int32))
    np.testing.assert_allclose(scalar.full_train_loss(),engine.full_train_loss(),rtol=1e-12)


@pytest.mark.parametrize('family',['p_only','tp'])
def test_long_checkpoint_actual_resume_for_both_families(tmp_path,family):
    x,y=case();cfg=config()
    if family=='tp':cfg={**cfg,'input_partition':'current_Y9__current_T_logP'}
    make=lambda: fresh_tp(x,y,cfg,11) if family=='tp' else PressureTrainer(x,y,cfg,11)
    a=make();a.plateau_monitor=new_monitor(POLICY)
    engine=SerialSteps(a);engine.advance(8)
    b=make();b.plateau_monitor=new_monitor(POLICY)
    SerialSteps(b).advance(3)
    checkpoint=b.save_checkpoint(tmp_path/'checkpoint','interrupted')
    c=restore_trainer(checkpoint,x,y,cfg,11,family)
    assert c.plateau_monitor['policy']==POLICY
    SerialSteps(c).advance(8)
    for aa,cc in zip(a.predictor.model.get_weights(),c.predictor.model.get_weights()):
        np.testing.assert_array_equal(aa,cc)
    for aa,cc in zip(a.optimizer.variables,c.optimizer.variables):
        np.testing.assert_array_equal(aa.numpy(),cc.numpy())
    np.testing.assert_array_equal(a.rng.integers(0,len(x),8,dtype=np.int32),
                                  c.rng.integers(0,len(x),8,dtype=np.int32))
    assert c.update==int(c.optimizer.iterations.numpy())==8


def history(function,maximum=100000):
    return [dict(update=u,full_train_loss=function(u)) for u in range(0,maximum+1,1000)]


def test_plateau_requires_minimum_two_observed_windows_and_sustained_checks():
    h=history(lambda u:1.)
    m=new_monitor(POLICY)
    assert not check_plateau(h,45000,m)
    assert not check_plateau(h,50000,m)
    assert not check_plateau(h,55000,m)
    assert check_plateau(h,60000,m)
    assert len(m['checks'])==3
    assert m['checks'][-1]['window_counts']==[20,20]
    assert check_plateau(h,60000,m) and len(m['checks'])==3


def test_declining_loss_missing_observations_and_nonconsecutive_checks_do_not_pass():
    m=new_monitor(POLICY)
    assert not check_plateau(history(lambda u:np.exp(-u/100000)),50000,m)
    assert not m['checks'][-1]['passed']
    flat=history(lambda u:1.)
    m=new_monitor(POLICY)
    check_plateau(flat,50000,m)
    assert not check_plateau(flat,60000,m)
    assert m['passing_since']==60000
    missing=[r for r in flat if r['update']!=49000]
    assert not check_plateau(missing,65000,m)
    assert m['passing_since'] is None


def test_zero_loss_relative_change_is_explicit_and_finite():
    flat=history(lambda u:0.)
    m=new_monitor(POLICY)
    check_plateau(flat,50000,m)
    assert m['checks'][-1]['relative_changes']==[0.,0.,0.]
    rising=history(lambda u:0. if u<=30000 else 1.)
    m=new_monitor(POLICY)
    assert not check_plateau(rising,50000,m)
    assert m['checks'][-1]['relative_changes']==[None,None,None]


def test_timed_mean_preserves_original_raw_and_repaired_values():
    from kinetics.deeponet_long_evaluation import TimedMean
    from kinetics.mixed_ensemble import MixedMean
    from test_mixed_ensemble import member,state
    models=[member(b=10),member(a=.1)]
    x=np.c_[state(),[101325.]]
    for mode in ('raw','repaired'):
        ordinary=MixedMean(models,['ann','deeponet'],mode)
        measured=TimedMean(models,['ann','deeponet'],mode)
        np.testing.assert_array_equal(ordinary.predict(x),measured.predict(x))
        assert measured.timing['neural']>0 and measured.timing['decode']>0


def test_common_completed_intersection_excludes_failed_prefixes():
    from kinetics.deeponet_long_report import common_completed
    a=pd.DataFrame(dict(trajectory=[1,2,3],horizon=[.01]*3,complete=[True,True,False]))
    b=pd.DataFrame(dict(trajectory=[1,2,3],horizon=[.01]*3,complete=[False,True,True]))
    assert common_completed({'a':a,'b':b},.01)==[2]
    assert common_completed({'a':a,'b':b},.002)==[]


def test_tiny_evaluation_arm_writes_verified_prefix_metrics_and_separate_timings(tmp_path):
    from kinetics.deeponet_long_evaluation import evaluate_arm
    from kinetics.ann_deeponet_study import verified_evaluation
    from kinetics.mixed_ensemble import MixedMean
    from kinetics.p_only_training import sha256
    from test_mixed_ensemble import member,state
    reference=np.repeat(state()[None,...],2,axis=0)
    reference=np.repeat(reference,3,axis=1)
    thermo=MixedMean([member()],['ann']).physics
    data=dict(states=reference,pressures=np.full(2,101325.),split=np.array(['test','test']),
              equilibrium=reference[:,0].copy(),element_matrix=thermo.matrix)
    fine=tmp_path/'fine.csv'
    pd.DataFrame(dict(trajectory=[0,1],reference_dt=[2e-7]*2,tau400=[np.nan]*2)).to_csv(fine,index=False)
    fingerprint=dict(reference_ignition={'test':dict(path=str(fine),sha256=sha256(fine))})
    cfg=dict(dt=2e-6,horizon=4e-6,one_step_chunk_size=2)
    info=evaluate_arm(tmp_path,'test_team',[member()],['ann'],'raw','test',data,cfg,fingerprint,float('inf'))
    marker=list((tmp_path/'evaluations').glob('*/*/complete.json'))[0]
    assert verified_evaluation(marker.parent)==info
    timing=json.loads((marker.parent/'timing.json').read_text())
    assert timing['rollout_components']['neural']>0
    assert timing['one_step_components']['neural']>0
    frame=pd.read_csv(marker.parent/'trajectories.csv')
    assert frame.loc[np.isclose(frame.horizon,4e-6),'complete'].all()
    assert frame.loc[np.isclose(frame.horizon,.002),'full_temperature_rmse'].isna().all()


@pytest.mark.parametrize('family',['p_only','tp'])
def test_member_runner_caps_and_resumes_without_resetting_state(tmp_path,family):
    from kinetics.deeponet_long_study import train_member
    x,y=case();model_cfg=config()
    if family=='tp':model_cfg={**model_cfg,'input_partition':'current_Y9__current_T_logP'}
    snapshot=dict(base_config=model_cfg,p_only_config=model_cfg,p_only_origins={})
    if family=='p_only':
        initial=PressureTrainer(x,y,model_cfg,11);initial.advance(3)
        origin=initial.save_checkpoint(tmp_path/'origin','synthetic_origin')
        snapshot['p_only_origins']['11']=dict(checkpoint=str(origin))
    cfg=dict(maximum_updates=8,checkpoint_every=5,plateau=POLICY)
    record=train_member(tmp_path,cfg,snapshot,dict(x=x,y=y),family,11,float('inf'))
    assert record['reason']=='update_limit' and record['updates']==8
    resumed=train_member(tmp_path,{**cfg,'maximum_updates':10},snapshot,dict(x=x,y=y),family,11,float('inf'))
    assert resumed['updates']==resumed['optimizer_iteration']==10
    assert resumed['mode']=='continuation_within_long_run'
    assert Path(record['checkpoint'],'complete.json').exists()
