import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
from kinetics.p_only_deeponet import PressureDeepONet
from kinetics.p_only_training import PressureTrainer, learning_rate
from kinetics.study import Predictor, tf


def case():
    rng = np.random.default_rng(43)
    state = np.c_[rng.uniform(900,1600,24),rng.dirichlet(np.ones(9),24)]
    x = np.c_[state,rng.uniform(1,5,24)*101325]
    y = state.copy()
    y[:,0] += 20
    y[:,1:] = .99*state[:,1:] + .01/9
    return x,y


def config(tiny=True):
    return dict(input_partition='current_T_Y9__logP', branch_layers=[4] if tiny else [64,64],
        trunk_layers=[3] if tiny else [32], latent_width=2 if tiny else 7,
        learning_rate=.001,batch_size=8)


def test_production_parameter_count_exact_preprocessing_and_initial_residual():
    x,y = case()
    p = PressureDeepONet(dict(kind='boxcox',gamma=.1),x,y,config(False),11)
    ann = Predictor(dict(kind='boxcox',gamma=.1),x,y,
        dict(hidden_layers=[64,64,64],activation='tanh',learning_rate=.001),11)
    assert p.model.count_params() == 9719
    np.testing.assert_array_equal(p.train_features,ann.train_features)
    np.testing.assert_array_equal(p.train_target,ann.train_target)
    np.testing.assert_array_equal(p.model(p.train_features).numpy(),np.zeros((24,10)))


def test_current_temperature_in_branch_and_only_pressure_in_trunk_both_learn():
    x,y = case()
    trainer = PressureTrainer(x,y,config(False),11)
    p = trainer.predictor
    before = {n:[w.copy() for w in p.model.get_layer(n).get_weights()]
              for n in ('branch_0','pressure_basis')}
    trainer.advance(4)
    for n in before:
        assert any(not np.array_equal(a,b) for a,b in zip(before[n],p.model.get_layer(n).get_weights()))
    probe = np.repeat(x[:1],3,axis=0)
    probe[1,0] += 200
    probe[2,-1] *= 2
    f = p.features(probe).astype('float32')
    partition = tf.keras.Model(p.model.input,
        [p.model.get_layer('branch_0').input,p.model.get_layer('trunk_0').input])
    branch,trunk = [v.numpy() for v in partition(f)]
    np.testing.assert_array_equal(branch,f[:,:10])
    np.testing.assert_array_equal(trunk,f[:,10:11])
    np.testing.assert_array_equal(branch[0],branch[2])
    np.testing.assert_array_equal(trunk[0],trunk[1])
    assert branch[0,0] != branch[1,0]
    assert trunk[0,0] != trunk[2,0]
    residual = p.model(f).numpy()
    assert not np.array_equal(residual[0],residual[1])
    assert not np.array_equal(residual[0],residual[2])


def test_model_reload_and_refuses_wrong_partition_family_or_overwrite(tmp_path):
    x,y = case()
    t = PressureTrainer(x,y,config(),11)
    t.advance(3)
    path = tmp_path/'p_only'
    t.predictor.save(path)
    loaded = PressureDeepONet.load(path)
    np.testing.assert_array_equal(loaded.model(loaded.features(x).astype('float32')).numpy(),
                                  t.predictor.model(t.predictor.features(x).astype('float32')).numpy())
    with pytest.raises(FileExistsError):
        t.predictor.save(path)
    with pytest.raises(ValueError,match='Branch'):
        PressureDeepONet(dict(kind='boxcox'),x,y,{**config(),'input_partition':'current_Y9__current_T_logP'},11)
    info = json.loads(path.with_suffix('.json').read_text())
    info['family'] = 'temperature_pressure_deeponet'
    path.with_suffix('.json').write_text(json.dumps(info))
    with pytest.raises(ValueError,match='P-only'):
        PressureDeepONet.load(path)


def test_checkpoint_resume_in_fresh_process_matches_uninterrupted_adam_and_rng(tmp_path):
    x,y = case()
    cfg = config()
    uninterrupted = PressureTrainer(x,y,cfg,11)
    uninterrupted.advance(3)
    uninterrupted.full_train_loss()
    uninterrupted.advance(7)
    interrupted = PressureTrainer(x,y,cfg,11)
    interrupted.advance(3)
    interrupted.full_train_loss()
    checkpoint = interrupted.save_checkpoint(tmp_path/'checkpoint','interrupted_test')
    np.savez(tmp_path/'data.npz',x=x,y=y)
    code = '''
import json,sys,numpy as np
from kinetics.p_only_training import PressureTrainer
data=np.load(sys.argv[1]);cfg=json.loads(sys.argv[4])
t=PressureTrainer.restore(sys.argv[2],data['x'],data['y'],cfg,11)
t.advance(7)
arrays={f'w{i}':v for i,v in enumerate(t.predictor.model.get_weights())}
arrays.update({f'a{i}':v.numpy() for i,v in enumerate(t.optimizer.variables)})
arrays['next_batch']=t.rng.integers(0,len(data['x']),8,dtype=np.int32)
np.savez(sys.argv[3],**arrays)
'''
    env = {**os.environ,'TF_ENABLE_ONEDNN_OPTS':'0','CUDA_VISIBLE_DEVICES':'-1',
           'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    result = subprocess.run([sys.executable,'-c',code,str(tmp_path/'data.npz'),str(checkpoint),
        str(tmp_path/'resumed.npz'),json.dumps(cfg)],env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0,result.stderr
    with np.load(tmp_path/'resumed.npz') as resumed:
        for i,w in enumerate(uninterrupted.predictor.model.get_weights()):
            np.testing.assert_array_equal(resumed[f'w{i}'],w)
        for i,v in enumerate(uninterrupted.optimizer.variables):
            np.testing.assert_array_equal(resumed[f'a{i}'],v.numpy())
        np.testing.assert_array_equal(resumed['next_batch'],uninterrupted.rng.integers(0,len(x),8,dtype=np.int32))
    assert int(uninterrupted.optimizer.iterations.numpy()) == 7


def test_resume_rejects_changed_data_config_and_corrupted_state(tmp_path):
    x,y = case()
    t = PressureTrainer(x,y,config(),11)
    t.advance(2)
    ckpt = t.save_checkpoint(tmp_path/'checkpoint','test')
    with pytest.raises(FileExistsError):
        t.save_checkpoint(ckpt,'test')
    with pytest.raises(ValueError,match='mismatch'):
        PressureTrainer.restore(ckpt,x,y+1e-10,config(),11)
    with pytest.raises(ValueError,match='mismatch'):
        PressureTrainer.restore(ckpt,x,y,{**config(),'batch_size':9},11)
    with (ckpt/'adam.npz').open('ab') as handle:
        handle.write(b'corrupted')
    with pytest.raises(ValueError,match='hash'):
        PressureTrainer.restore(ckpt,x,y,config(),11)


def test_seeds_are_independent_reproducible_and_deadline_stops_without_update():
    x,y = case()
    a = PressureTrainer(x,y,config(),11)
    a.advance(3)
    b = PressureTrainer(x,y,config(),22)
    b.advance(3)
    c = PressureTrainer(x,y,config(),11)
    c.advance(3)
    for aa,cc in zip(a.predictor.model.get_weights(),c.predictor.model.get_weights()):
        np.testing.assert_array_equal(aa,cc)
    assert any(not np.array_equal(aa,bb) for aa,bb in zip(a.predictor.model.get_weights(),b.predictor.model.get_weights()))
    state = json.dumps(c.rng.bit_generator.state)
    assert c.advance(4,deadline=0) == 'time_limit'
    assert c.update == int(c.optimizer.iterations.numpy()) == 3
    assert json.dumps(c.rng.bit_generator.state) == state


@pytest.mark.parametrize('update,expected',[(1,.001),(6000,.001),(6001,.0003),
    (9000,.0003),(9001,.0001),(10000,.0001),(50000,.0001),(250000,.0001)])
def test_lr_does_not_shift_with_longer_target_or_resume(update,expected):
    assert learning_rate(update) == expected


def test_real_p_only_member_accepts_mixed_team_current_state():
    from kinetics.mixed_ensemble import MixedMean, rollout
    x,y = case()
    t = PressureTrainer(x,y,config(),11)
    t.advance(3)
    ann = Predictor(dict(kind='boxcox',gamma=.1),x,y,
        dict(hidden_layers=[4],activation='tanh',learning_rate=.001),22)
    team = MixedMean([ann]*10+[t.predictor]*10,['ann']*10+['deeponet']*10,'raw')
    seen = []
    original = team.proposals
    def observe(current):
        seen.append(current.copy())
        return original(current)
    team.proposals = observe
    traces,events = rollout(team,x[:1,:10],x[:1,-1],2,2e-6)
    assert len(traces[0]) == 3
    np.testing.assert_array_equal(seen[1][0,:10],traces[0][1])
    assert events[-1]['valid_ann'] == events[-1]['valid_deeponet'] == 10


def test_training_budget_expiry_and_resume_cannot_create_a_fresh_window(tmp_path,monkeypatch):
    from kinetics import p_only_study
    budget = dict(seconds=1.,started_unix=time.time()-10,expires_unix=time.time()-9)
    (tmp_path/'training_budget.json').write_text(json.dumps(budget))
    def forbidden(*args,**kwargs):
        raise AssertionError('Expired budget must not initialize a training member')
    monkeypatch.setattr(p_only_study,'PressureTrainer',forbidden)
    assert p_only_study.train(tmp_path,{},dict(seeds=[11]),1.) == 'time_limit'
    assert json.loads((tmp_path/'training_budget.json').read_text()) == budget
    with pytest.raises(ValueError,match='original training budget'):
        p_only_study.train(tmp_path,{},dict(seeds=[11]),600.)
    assert not (tmp_path/'checkpoints').exists()
