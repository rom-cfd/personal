import numpy as np
from kinetics.zhang_model import ZhangPredictor,ZhangCommittee
from kinetics.zhang_wang import WangHardCorrection
from kinetics.wang_physics import WangThermo
from kinetics.enthalpy_temperature import load_predictor
from kinetics.committee import fit_metric
from kinetics.data import mechanism


def committee(tmp_path,bias=.05):
    rng=np.random.default_rng(23)
    state=np.c_[rng.uniform(1000,1500,12),rng.dirichlet(np.ones(9),12)]
    x=np.c_[state,np.full(12,101325.)];y=state.copy();y[:,0]+=2;y[:,1]*=.99;y[:,6]+=state[:,1]*.01
    p=ZhangPredictor(dict(gamma=.1),x,y,dict(hidden_layers=[4],activation='tanh',learning_rate=.001),11)
    w=p.model.get_weights();w[-1][:]=bias;p.model.set_weights(w);p.save(tmp_path/'m')
    return ZhangCommittee([tmp_path/'m'],[11],fit_metric(state),'single'),x


def test_corrected_rollout_state_conserves_elements_and_enthalpy(tmp_path):
    c,x=committee(tmp_path);p=WangHardCorrection(c);out=p.predict_full(x)
    raw=c.predict_full(x);thermo=WangThermo(.1)
    expected=thermo.repair(raw[None,:,:10],x)[0]
    np.testing.assert_allclose(out[:,:10],expected,atol=1e-12,equal_nan=True)
    np.testing.assert_array_equal(out[:,-1],raw[:,-1])
    live=np.isfinite(out).all(axis=1);assert live.any()
    np.testing.assert_allclose(out[live,1:10]@thermo.matrix.T,x[live,1:10]@thermo.matrix.T,atol=1e-10)
    g=mechanism()
    for a,b in zip(x[live],out[live]):
        g.TPY=a[0],a[-1],a[1:10];h=g.enthalpy_mass;g.HPY=h,a[-1],b[1:10]
        np.testing.assert_allclose(b[0],g.T,atol=1e-4)


def test_guard_only_changes_out_of_range_species_and_mode_reloads(tmp_path):
    c,x=committee(tmp_path,bias=-30.)  # Drives BCT below the image of Y=0.
    assert np.isnan(c.predict_full(x)[:,1:10]).any()
    guarded=WangHardCorrection(c).raw_candidates(x)[0]
    assert np.isfinite(guarded).all() and (guarded[:,1:10]>=0).all() and (guarded[:,1:10]<=1).all()
    assert np.isnan(WangHardCorrection(c,guard=False).raw_candidates(x)[0][:,1:10]).any()
    for guard in (True,False):
        p=WangHardCorrection(c,guard);p.save(tmp_path/f'method{guard}.json')
        q=load_predictor(tmp_path/f'method{guard}.json')
        assert isinstance(q,WangHardCorrection) and q.guard==guard
        np.testing.assert_allclose(p.predict_full(x),q.predict_full(x),atol=1e-12,equal_nan=True)


def test_committee_rules_select_or_average_corrected_candidates(tmp_path):
    from kinetics.committee import choose_medoid
    rng=np.random.default_rng(29)
    state=np.c_[rng.uniform(1000,1500,12),rng.dirichlet(np.ones(9),12)]
    x=np.c_[state,np.full(12,101325.)];y=state.copy();y[:,0]+=2;y[:,1]*=.99;y[:,6]+=state[:,1]*.01
    paths=[]
    for seed,bias in [(11,.03),(22,.05),(33,.08)]:
        m=ZhangPredictor(dict(gamma=.1),x,y,dict(hidden_layers=[4],activation='tanh',learning_rate=.001),seed)
        w=m.model.get_weights();w[-1][:]=bias;m.model.set_weights(w);m.save(tmp_path/f'm{seed}');paths.append(tmp_path/f'm{seed}')
    metric=fit_metric(state);matrix=WangThermo(.1).matrix
    medoid=WangHardCorrection(ZhangCommittee(paths,[11,22,33],metric,'medoid'))
    c=medoid.candidates(x);t=c[:,:,:10].copy();t[~np.isfinite(c).all(axis=-1)]=np.nan
    expected,choice,_=choose_medoid(t,metric,'scaled')
    out=medoid.predict_full(x);np.testing.assert_allclose(out[:,:10],expected,atol=1e-12,equal_nan=True)
    assert set(medoid.last_diagnostics['selected_seed'][choice>=0])<={11,22,33}
    mean=WangHardCorrection(ZhangCommittee(paths,[11,22,33],metric,'mean')).predict_full(x)
    live=np.isfinite(mean).all(axis=1);assert live.any()
    np.testing.assert_allclose(mean[live,1:10]@matrix.T,x[live,1:10]@matrix.T,atol=1e-10)
