import numpy as np
from kinetics.committee import choose_medoid,fit_metric


def candidates(values):
    out=np.zeros((len(values),1,10));out[:,:,0]=1000
    out[:,0,1]=values
    return out


def test_medoid_is_real_candidate_and_resists_outlier():
    c=candidates([0,.01,.02,.03,.9]);metric=fit_metric(np.r_[c[:,0],c[:,0]+1e-8])
    out,index,stats=choose_medoid(c,metric)
    assert index[0]==2
    np.testing.assert_array_equal(out,c[index[0]])
    assert out[0,1]!=np.mean(c[:,0,1])
    assert stats['valid_count'][0]==5


def test_invalid_candidates_and_deterministic_ties():
    c=candidates([.1,.2,np.nan]);metric=fit_metric(c[:2,0])
    out,index,stats=choose_medoid(c,metric)
    assert index[0]==0 and stats['valid_count'][0]==2
    c[:]=np.nan
    out,index,stats=choose_medoid(c,metric)
    assert index[0]==-1 and np.all(np.isnan(out)) and stats['valid_count'][0]==0


def test_scale_balances_minor_species_and_is_training_only():
    train=np.array([[1000,*([.1]*9)],[2000,*([.2]*9)]])
    metric=fit_metric(train)
    assert metric['fit_split']=='train'
    assert np.isfinite(list(metric['species_scale'])).all()
    c=candidates([.1,.2,.3]);c[2,0,2]=.1
    for kind in ('scaled','softlog'):
        out,index,_=choose_medoid(c,metric,kind)
        assert any(np.array_equal(out,c[i]) for i in range(3))


def test_compiled_committee_matches_individual_networks(tmp_path):
    from kinetics.study import Predictor
    from kinetics.committee import Committee
    rng=np.random.default_rng(55)
    state=np.c_[rng.uniform(1000,1500,8),rng.dirichlet(np.ones(9),8)]
    x=np.c_[state,np.full(8,101325.)];paths=[];individual=[]
    cfg=dict(hidden_layers=[4],activation='tanh',learning_rate=.001)
    for seed in (11,22):
        p=Predictor(dict(kind='boxcox',gamma=.1),x,state,cfg,seed)
        weights=p.model.get_weights();weights[-1]=np.full_like(weights[-1],.001*seed)
        p.model.set_weights(weights);path=tmp_path/f'model_seed{seed}';p.save(path)
        paths.append(path);individual.append(p.predict(x))
    committee=Committee(paths,[11,22],fit_metric(state))
    np.testing.assert_allclose(committee.proposals(x,guard=False),np.stack(individual),rtol=1e-5,atol=1e-8,equal_nan=True)


def test_logged_rollout_is_self_fed_and_switches_are_counted():
    from kinetics.committee_study import run_rollouts
    class Dummy:
        def predict(self,x):
            self.last_diagnostics=dict(selected_seed=np.where(x[:,0]>=1010,22,11),valid_count=np.full(len(x),10))
            p=x[:,:10].copy();p[:,0]+=10
            return p
    truth=np.c_[np.linspace(1000,5000,5),np.ones((5,9))/9][None,...]
    summary,rows,species,events,traces=run_rollouts(Dummy(),truth,np.array([101325]),np.array([5]),np.ones((1,9)),2e-6)
    np.testing.assert_allclose(traces[0][:,0],[1000,1010,1020,1030,1040])
    assert sum(e['switched'] for e in events)==1
    assert all(e['trajectory']==5 for e in events)


def test_100_member_chunking_preserves_exact_selection():
    rng=np.random.default_rng(99)
    c=np.concatenate([rng.uniform(1000,2500,(100,130,1)),rng.dirichlet(np.ones(9),size=(100,130))],axis=-1)
    c[2,:5]=np.nan
    metric=fit_metric(c[0,10:])
    dense,idx,stats=choose_medoid(c,metric,batch_chunk=None)
    chunked,idx_chunk,stats_chunk=choose_medoid(c,metric,batch_chunk=7)
    np.testing.assert_array_equal(idx,idx_chunk)
    np.testing.assert_array_equal(dense,chunked)
    for key in stats:np.testing.assert_allclose(stats[key],stats_chunk[key],rtol=1e-14,atol=1e-14)
