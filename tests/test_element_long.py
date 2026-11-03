import json
from pathlib import Path
import numpy as np
import pytest

from kinetics.element_long_training import joint_monitor,joint_check
from kinetics.element_long_study import train_member
from kinetics.soft_constraint_training import SoftTrainer
from test_soft_constraint_training import setup
from test_deeponet_long_training import POLICY


def history(data=lambda u:1.,element=lambda u:.1):
    return [dict(update=u,data_loss=data(u),element_loss=element(u),
        full_train_loss=data(u)+10*element(u)) for u in range(0,100001,1000)]


def test_joint_plateau_needs_all_components_and_three_sustained_checks():
    monitor=joint_monitor(POLICY,'run-sha',10000,1.)
    h=history()
    assert not joint_check(h,45000,monitor)
    assert not joint_check(h,50000,monitor)
    assert not joint_check(h,55000,monitor)
    assert joint_check(h,60000,monitor)
    assert monitor['checks'][-1]['component_plateaus']==dict(total=True,data=True,element=True)
    copy=json.loads(json.dumps(monitor))
    assert joint_check(h,60000,copy) and len(copy['checks'])==3


def test_constant_total_with_cancelling_data_and_element_trends_does_not_pass():
    h=history(data=lambda u:1+u/100000,element=lambda u:.4-u/1000000)
    monitor=joint_monitor(POLICY,'run-sha',10000,1.)
    for update in (50000,55000,60000):assert not joint_check(h,update,monitor)
    assert monitor['components']['total']['plateau']
    assert not monitor['components']['data']['plateau']
    assert not monitor['components']['element']['plateau']


def test_missing_one_component_observation_prevents_joint_plateau():
    h=history();del h[49]['element_loss']
    monitor=joint_monitor(POLICY,'run-sha',10000,1.)
    for u in (50000,55000,60000):assert not joint_check(h,u,monitor)
    assert monitor['components']['total']['plateau'] and monitor['components']['data']['plateau']
    assert not monitor['components']['element']['plateau']


@pytest.mark.parametrize('family',['ann','p_only','tp','tp_phi'])
def test_member_continuation_preserves_original_adam_rng_and_monitor_lineage(tmp_path,family):
    x,y,cfg,matrix=setup(family)
    parent=SoftTrainer(x,y,cfg,11,family,matrix,10.);parent.advance(3);parent.full_train_loss()
    cp=parent.save_checkpoint(tmp_path/'original','short_endpoint')
    expected=SoftTrainer(x,y,cfg,11,family,matrix,10.);expected.advance(10)
    root=tmp_path/'long';root.mkdir();(root/'run.json').write_text('{}\n')
    snapshot=dict(model_configs={family:cfg},element_matrix=matrix.tolist(),
        origin_members=[dict(family=family,seed=11,checkpoint=str(cp))])
    if family=='tp_phi':
        raw=np.c_[x[:,:10],x[:,-1]];data=dict(initials=np.c_[x[:,0],x[:,-1],x[:,10]])
        pairs=dict(x=raw,y=y,parent=np.c_[np.arange(len(x)),np.zeros(len(x),int)])
    else:pairs=dict(x=x,y=y);data={}
    config=dict(plateau=POLICY,observe_every=2,checkpoint_every=2)
    first=train_member(root,config,snapshot,pairs,data,family,11,float('inf'),maximum_update=8)
    second=train_member(root,config,snapshot,pairs,data,family,11,float('inf'),maximum_update=10)
    assert second['updates']==second['optimizer_iteration']==10 and second['origin_update']==8
    restored=SoftTrainer.restore(second['checkpoint'],x,y,cfg,11,family,matrix,10.)
    assert restored.plateau_monitor['enabled'] and restored.plateau_monitor['origin_update']==3
    for a,b in zip(expected.predictor.model.get_weights(),restored.predictor.model.get_weights()):
        np.testing.assert_array_equal(a,b)
    for a,b in zip(expected.optimizer.variables,restored.optimizer.variables):
        np.testing.assert_array_equal(a.numpy(),b.numpy())
    np.testing.assert_array_equal(expected.rng.integers(0,len(x),8,dtype=np.int32),
                                  restored.rng.integers(0,len(x),8,dtype=np.int32))
    assert Path(first['checkpoint'],'complete.json').is_file() and Path(cp,'complete.json').is_file()
