import numpy as np
from kinetics.committee import choose_min_correction, Committee


def test_minimum_is_correction_not_smallest_physical_step():
    metric=dict(temperature_scale=100.,species_scale=[1.]*9)
    raw=np.zeros((3,2,10));raw[...,0]=1000
    raw[1,:,0]=2000  # Large physical step can still need very little correction.
    corrected=raw.copy()
    corrected[0,:,0]+=20
    corrected[1,:,0]+=1
    corrected[2,:,0]+=30
    # At the second state a different member is the minimum.
    corrected[0,1,0]=raw[0,1,0]
    out,choice,stats=choose_min_correction(raw,corrected,metric)
    np.testing.assert_array_equal(choice,[1,0])
    np.testing.assert_array_equal(out,corrected[choice,np.arange(2)])
    np.testing.assert_allclose(stats['correction_norm'],[.01/np.sqrt(10),0])


def test_invalid_exclusion_tie_and_all_invalid():
    metric=dict(temperature_scale=100.,species_scale=[1.]*9)
    raw=np.zeros((3,3,10));corrected=raw.copy()
    corrected[0,0]=np.nan
    corrected[:,2]=np.nan
    out,choice,stats=choose_min_correction(raw,corrected,metric)
    np.testing.assert_array_equal(choice,[1,0,-1])
    np.testing.assert_array_equal(stats['valid_count'],[2,3,0])
    assert np.isnan(out[2]).all() and np.isnan(stats['correction_norm'][2])


def test_units_are_normalized_before_comparing_corrections():
    metric=dict(temperature_scale=100.,species_scale=[.001]*9)
    raw=np.zeros((2,1,10));corrected=raw.copy()
    corrected[0,0,0]=1.  # 1 K / 100 K is less than .001 / .001.
    corrected[1,0,1]=.001
    _,choice,_=choose_min_correction(raw,corrected,metric)
    assert choice[0]==0


def test_predict_applies_correction_and_changes_members_each_step():
    p=Committee.__new__(Committee)
    p.rule='min_correction';p.seeds=np.array([11,22]);p.distance='scaled'
    p.metric=dict(temperature_scale=100.,species_scale=[1.]*9)
    p.timing=dict(neural_decode=0.,candidate_repair=0.,selection=0.)
    def proposals(x,guard=True):
        return np.stack([x[:,:10]+1,x[:,:10]+2])
    def repair(raw,x):
        result=raw.copy()
        result[0,:,0]+=np.where(x[:,0]<1010,0,10)
        result[1,:,0]+=np.where(x[:,0]<1010,10,0)
        return result
    p.proposals=proposals;p.repair=repair
    x=np.zeros((2,11));x[:,0]=[1000,1020]
    expected=repair(proposals(x),x)[[0,1],np.arange(2)]
    np.testing.assert_array_equal(p.predict(x),expected)
    np.testing.assert_array_equal(p.last_diagnostics['selected_seed'],[11,22])
