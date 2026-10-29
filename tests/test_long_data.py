import numpy as np
from kinetics.long_data import equilibrium_mask,select_transitions


def test_quiet_induction_is_not_chemical_equilibrium():
    initial=np.r_[1000,np.ones(9)/9]
    eq=np.r_[2500,np.r_[.2,np.full(8,.1)]]
    states=np.tile(initial,(500,1))
    assert not equilibrium_mask(states,eq).any()
    selected,mask,activity=select_transitions(states,eq,max_active=40,max_equilibrium=3)
    assert len(selected)==40 and selected[0]==0 and selected[-1]==len(states)-2
    assert not mask[selected].any()


def test_equilibrium_cap_keeps_anchors_and_full_targets_stay_one_dt_ahead():
    eq=np.r_[2500,np.ones(9)/9]
    states=np.tile(eq,(5001,1));states[:40,0]=np.linspace(1000,2490,40)
    selected,mask,activity=select_transitions(states,eq,max_active=40,max_equilibrium=8)
    assert mask[selected].sum()<=8
    assert selected[0]==0 and selected[-1]==4999
    # Selection thins input times only; target for j is j+1, never next selected j.
    target_indices=selected+1
    assert np.all(target_indices-selected==1)
    assert np.any(np.diff(selected)>1)
    np.testing.assert_array_equal(states[target_indices[-1]],states[-1])


def test_trace_species_disequilibrium_prevents_premature_thinning():
    eq=np.r_[2500,np.ones(9)/9]
    states=np.tile(eq,(100,1));states[:,2]+=.01;states[:,1]-=.01
    assert not equilibrium_mask(states,eq).any()


def test_long_generator_preserves_splits_source_and_one_step_targets(tmp_path):
    import hashlib
    from kinetics.data import generate as original_generate
    from kinetics.long_data import generate
    source=tmp_path/'source.npz';original_generate(source,trajectories=10,steps=4,seed=57)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    target=tmp_path/'long';generate(source,target,horizon=8e-6,max_active=4,max_equilibrium=1)
    old=np.load(source);full=np.load(target/'trajectories.npz');pairs=np.load(target/'training_pairs.npz')
    np.testing.assert_array_equal(full['split'],old['split'])
    np.testing.assert_allclose(full['states'][:,0],old['states'][:,0],rtol=0,atol=0)
    for x,y,(i,j) in zip(pairs['x'],pairs['y'],pairs['parent']):
        assert full['split'][i]=='train'
        np.testing.assert_allclose(x[:10],full['states'][i,j])
        np.testing.assert_allclose(y,full['states'][i,j+1])
    assert float(pairs['dt'])==float(full['dt'])==2e-6
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
