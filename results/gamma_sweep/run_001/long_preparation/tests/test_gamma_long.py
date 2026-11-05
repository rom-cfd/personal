import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from kinetics.gamma_long_training import MemberAllowance,bridge_legacy,start_long_monitor,restore_long
from kinetics.gamma_long_study import train,validate_config
from kinetics.gamma_sweep_training import GammaTrainer
from kinetics.p_only_training import PressureTrainer,sha256
from kinetics.tp_phi_training import PhiTrainer
from kinetics.deeponet_long_training import new_monitor,check_plateau
from scripts.report_gamma_long import paired_summaries,paired_errors,stage_figure
from scripts.report_gamma_sweep import METHODS,normalized_metrics
from test_soft_constraint_training import setup


class Clock:
    def __init__(self):self.wall=1000.;self.perf=10.
    def time(self):return self.wall
    def perf_counter(self):return self.perf
    def time_ns(self):return int(self.wall*1e9)
    def advance(self,seconds):self.wall+=seconds;self.perf+=seconds


def test_allowance_accumulates_active_visits_excludes_queue_and_charges_interruptions(tmp_path,monkeypatch):
    import kinetics.gamma_long_training as module
    clock=Clock();monkeypatch.setattr(module,'time',clock)
    path=tmp_path/'budget.json';a=MemberAllowance(path,150)
    assert a.begin('bootstrap')==160.;clock.advance(12);a.close()
    clock.advance(1000);assert a.remaining==138
    assert a.begin('long')==clock.perf+138;clock.advance(20);a.close()
    b=MemberAllowance(path,150);assert b.remaining==118
    b.begin('interrupted');clock.advance(200)
    c=MemberAllowance(path,150);assert c.remaining==0 and c.begin('retry') is None
    assert c.state['consumed_seconds']==150
    with pytest.raises(ValueError,match='allowance'):MemberAllowance(path,300)


@pytest.mark.parametrize('family',['p_only','tp_phi'])
def test_legacy_bridge_preserves_every_slot_rng_feature_and_next_updates(tmp_path,family):
    x,y,cfg,_=setup(family)
    old=(PhiTrainer if family=='tp_phi' else PressureTrainer)(x,y,cfg,11)
    old.advance(4);old.full_train_loss();cp=old.save_checkpoint(tmp_path/'legacy','test')
    record=dict(checkpoint=str(cp),seed=11,updates=4)
    bridged=bridge_legacy(record,x,y,family)
    for a,b in zip(old.optimizer.variables,bridged.optimizer.variables):np.testing.assert_array_equal(a.numpy(),b.numpy())
    assert old.rng.bit_generator.state==bridged.rng.bit_generator.state
    np.testing.assert_array_equal(old.tf_rng.state.numpy(),bridged.tf_rng.state.numpy())
    assert old.history==bridged.history and old.full_train_history==bridged.full_train_history
    old.advance(7);bridged.advance(7)
    for a,b in zip(old.predictor.model.get_weights(),bridged.predictor.model.get_weights()):np.testing.assert_array_equal(a,b)
    assert bridged.signature['loss']=='transformed_residual_MSE_only'


def test_scalar_monitor_and_long_restore_keep_exact_state(tmp_path):
    cfg=json.loads(Path('configs/gamma_long.json').read_text());policy=cfg['plateau']
    dummy=SimpleNamespace(elapsed_seconds=7.)
    start_long_monitor(dummy,policy,'run_sha',dict(checkpoint='origin',origin_kind='test'))
    assert dummy.plateau_monitor['enabled'] and 'components' not in dummy.plateau_monitor
    history=[dict(update=u,full_train_loss=1e-5) for u in range(0,60001,1000)]
    assert not check_plateau(history,50000,dummy.plateau_monitor)
    assert not check_plateau(history,55000,dummy.plateau_monitor)
    assert check_plateau(history,60000,dummy.plateau_monitor)
    x,y,model_cfg,_=setup('ann')
    whole=GammaTrainer(x,y,model_cfg,11,'ann',.3);whole.advance(7)
    part=GammaTrainer(x,y,model_cfg,11,'ann',.3);part.advance(4)
    part.plateau_monitor=new_monitor(policy,origin_update=4,origin_seconds=0.)
    part.plateau_monitor['long_run_sha256']='run_sha'
    cp=part.save_checkpoint(tmp_path/'long','observed_checkpoint')
    restored=restore_long(cp,x,y,dict(seed=11,family='ann',gamma=.3),'run_sha');restored.advance(7)
    for a,b in zip(whole.predictor.model.get_weights(),restored.predictor.model.get_weights()):np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='another run'):restore_long(cp,x,y,dict(seed=11,family='ann',gamma=.3),'other')


def test_exhausted_allowance_keeps_latest_observed_long_checkpoint(tmp_path):
    x,y,model_cfg,_=setup('ann')
    trainer=GammaTrainer(x,y,model_cfg,11,'ann',.2);trainer.advance(10000);trainer.full_train_loss()
    short=trainer.save_checkpoint(tmp_path/'short','short')
    root=tmp_path/'run';root.mkdir();(root/'run.json').write_text('{}')
    cfg=json.loads(Path('configs/gamma_long.json').read_text())
    start_long_monitor(trainer,cfg['plateau'],sha256(root/'run.json'),dict(checkpoint=str(short),origin_kind='test'))
    trainer.advance(11000);trainer.full_train_loss()
    parent=root/'checkpoints/gamma_0.2/ann/seed11'
    long_cp=trainer.save_checkpoint(parent/'update_0011000_test','observed_checkpoint')
    budget=root/'member_allowances/gamma_0.2/ann/seed11.json';budget.parent.mkdir(parents=True)
    budget.write_text(json.dumps(dict(seconds=150,consumed_seconds=150,visits=[],open_visit=None)))
    origin=dict(gamma=.2,family='ann',seed=11,updates=10000,checkpoint=str(short),path=str(short/'model'),
                sha256=sha256(short/'model.keras'),parameters=trainer.predictor.model.count_params(),origin_kind='test')
    train(root,cfg,{},[origin],None,None,float('inf'))
    r=json.loads((root/'final_members.json').read_text())[0]
    assert r['updates']==11000 and r['added_updates']==1000 and r['checkpoint']==str(long_cp)
    assert not r['borrowed_short_endpoint'] and r['reason']=='member_time_limit'


def examples():
    truth=np.array([[1000.,.1,.001,.001,.2,.001,.02,.001,.001,.7],
                    [1100.,.09,.002,.002,.19,.002,.03,.002,.002,.7],
                    [1200.,.08,.003,.003,.18,.003,.04,.003,.003,.7]])
    data=dict(states=np.stack([truth]*2),dt=np.array(2e-6),initials=np.array([[1000.,101325.,1.]]*2))
    traces={m:{i:truth.copy() for i in (0,1)} for m,_,_ in METHODS}
    return data,traces


def test_paired_cohorts_and_same_gamma_comparisons_do_not_mix_different_ics():
    data,short=examples();long={m:{i:p.copy() for i,p in profiles.items()} for m,profiles in short.items()}
    short['gamma_1_ann10'][1]=short['gamma_1_ann10'][1][:2]
    long['gamma_0.05_p_only10'][0]=long['gamma_0.05_p_only10'][0][:2]
    frames={s:normalized_metrics(data,t,[0,1],'test','repaired')[0] for s,t in [('short',short),('long',long)]}
    joined,summary,within,common,sets=paired_summaries(frames)
    assert len(joined)==1680 and common==[] and summary.median_nrmse.isna().all()
    assert sets['ann10']==[0] and sets['p_only10']==[1]
    pair=paired_errors(frames['short'],frames['long'])
    p=pair[(pair.team=='ann10')&(pair.gamma==.2)&(pair.variable=='T')].iloc[0]
    assert p.common_IC_count==2 and p.short_median_NRMSE==p.long_median_NRMSE==0


def test_short_and_long_figures_share_y_scales_and_seven_curves():
    data,short=examples();data['states']=np.repeat(data['states'][:,:1],5001,axis=1)
    short={m:{i:data['states'][i].copy() for i in (0,1)} for m,_,_ in METHODS}
    long={m:{i:p*np.r_[1.1,np.ones(9)] for i,p in profiles.items()} for m,profiles in short.items()}
    all_traces={s+'_'+m:p for s,t in [('short',short),('long',long)] for m,p in t.items()}
    a=stage_figure(data,short,all_traces,0,'test','repaired','short')
    b=stage_figure(data,long,all_traces,0,'test','repaired','long')
    assert len(a.axes)==len(b.axes)==70
    for x,y in zip(a.axes,b.axes):
        assert x.get_ylim()==y.get_ylim() and len(x.lines)==len(y.lines)==7
    plt.close(a);plt.close(b)


def test_long_config_keeps_requested_scalar_policy_and_allowance():
    cfg=json.loads(Path('configs/gamma_long.json').read_text());validate_config(cfg)
    with pytest.raises(ValueError):validate_config({**cfg,'member_budget_seconds':300})
    with pytest.raises(ValueError):validate_config({**cfg,'loss':'MSE_plus_physics'})
