import numpy as np
from kinetics.zhang_model import ZhangPredictor,ZhangCommittee
from kinetics.batched_training import train_group


def data():
    rng=np.random.default_rng(18)
    state=np.c_[rng.uniform(1000,1500,12),rng.dirichlet(np.ones(9),12)]
    x=np.c_[state,np.full(12,101325.)]
    y=state.copy();y[:,0]+=2;y[:,1]*=.99;y[:,6]+=state[:,1]*.01
    return x,y


def cfg():return dict(hidden_layers=[4],activation='tanh',learning_rate=.001,batch_size=8,loss='mae')


def test_literal_full_state_increment_and_separate_zscores(tmp_path):
    x,y=data();p=ZhangPredictor(dict(gamma=.1),x,y,cfg(),11)
    assert p.model.input_shape[-1]==p.model.output_shape[-1]==11
    encoded=np.c_[x[:,0],x[:,-1]/101325,(x[:,1:10]**.1-1)/.1]
    target=np.c_[y[:,0]-x[:,0],np.zeros(len(x)),(y[:,1:]**.1-x[:,1:10]**.1)/.1]
    np.testing.assert_allclose(p.encode(x),encoded)
    np.testing.assert_allclose(p.input_norm.mean,encoded.mean(axis=0))
    np.testing.assert_allclose(p.output_norm.mean,target.mean(axis=0),atol=1e-14)
    np.testing.assert_allclose(p.decode(p.output_norm.forward(target),x),np.c_[y,x[:,-1]],atol=1e-10)
    p.save(tmp_path/'p');q=ZhangPredictor.load(tmp_path/'p')
    np.testing.assert_allclose(p.predict_full(x),q.predict_full(x),atol=1e-12)


def test_no_decoder_clipping_normalization_or_temperature_pressure_closure():
    x,y=data();p=ZhangPredictor(dict(gamma=.5),x,y,cfg(),11)
    delta=np.zeros((len(x),11));delta[:,0]=100;delta[:,1]=.1;delta[:,2]=.5
    raw=p.decode(p.output_norm.forward(delta),x)
    np.testing.assert_allclose(raw[:,0],x[:,0]+100)
    np.testing.assert_allclose(raw[:,-1],x[:,-1]+.1*101325)
    assert np.max(np.abs(raw[:,1:10].sum(axis=1)-1))>.01
    delta[:,2]=-100
    invalid=p.decode(p.output_norm.forward(delta),x)
    assert np.isnan(invalid[:,1]).all()  # Undefined inverse fails; never clamp G.


def test_grouped_l1_training_has_eleven_outputs_and_no_physics_calls(tmp_path,monkeypatch):
    x,y=data();members,_,_=train_group(dict(gamma=.1),[11,22],3,x,y,cfg(),predictor_class=ZhangPredictor)
    paths=[]
    for i,p in enumerate(members):
        path=tmp_path/f'member{i}';p.save(path);paths.append(path)
    from kinetics.committee import fit_metric
    p=ZhangCommittee(paths,[11,22],fit_metric(x[:,:10]),'mean')
    import kinetics.wang_physics
    monkeypatch.setattr(kinetics.wang_physics.WangThermo,'repair',lambda *a: (_ for _ in ()).throw(AssertionError('physics called')))
    out=p.predict_full(x)
    assert out.shape==(len(x),11)
    p.save(tmp_path/'method.json');loaded=ZhangCommittee.load(tmp_path/'method.json')
    np.testing.assert_allclose(out,loaded.predict_full(x),atol=1e-12,equal_nan=True)


def test_raw_rollout_self_feeds_pressure_and_does_not_filter_finite_nonphysical_states():
    from kinetics.zhang_raw_study import raw_rollouts
    class Dummy:
        def predict_full(self,x):
            out=x.copy();out[:,0]+=10;out[:,1]+=.5;out[:,-1]+=100
            self.last_diagnostics=dict(selected_seed=np.full(len(x),11),valid_count=np.ones(len(x),int))
            return out
    truth=np.c_[np.full(4,1000.),np.ones((4,9))/9][None]
    traces,status,events=raw_rollouts(Dummy(),truth,np.array([101325.]),np.array([4]),2e-6,lambda:None)
    np.testing.assert_array_equal(traces[0][:,-1],[101325,101425,101525,101625])
    assert traces[0][-1,1]>1 and len(traces[0])==4 and status==['completed']


def test_undefined_inverse_stops_without_replacing_state():
    from kinetics.zhang_raw_study import raw_rollouts
    class Dummy:
        def predict_full(self,x):
            self.last_diagnostics=dict(selected_seed=np.full(len(x),-1),valid_count=np.zeros(len(x),int))
            return np.full_like(x,np.nan)
    truth=np.c_[np.full(4,1000.),np.ones((4,9))/9][None]
    traces,status,events=raw_rollouts(Dummy(),truth,np.array([101325.]),np.array([4]),2e-6,lambda:None)
    assert len(traces[0])==1 and status==['undefined_BCT_inverse_or_nonfinite']
    assert not events[0]['accepted']
