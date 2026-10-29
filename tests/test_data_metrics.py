import numpy as np
import pytest
from kinetics.data import mechanism, generate, samples, SPECIES
from kinetics.metrics import evaluate


def test_nine_species_and_conservation(tmp_path):
    gas=mechanism()
    assert gas.species_names==SPECIES and gas.n_reactions==28
    path=tmp_path/'data.npz'
    generate(path,trajectories=10,steps=4,seed=1)
    d=np.load(path)
    assert set(d['split'])=={'train','val','test'}
    ids=[set(samples(d,s)[2]) for s in ('train','val','test')]
    assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
    y=d['states'][:,:,1:]
    np.testing.assert_allclose(y.sum(axis=-1),1,atol=1e-10)
    elements=y@d['element_matrix'].T
    assert np.max(np.abs(elements-elements[:,:1]))<1e-10
    assert np.max(d['states'][:,:,0]-d['states'][:,:1,0])>0
    with pytest.raises(FileExistsError):generate(path)


def test_metrics_report_raw_violations():
    true=np.array([[1000,.2,.8],[1001,.3,.7]])
    current=np.c_[true,np.ones(2)]
    matrix=np.eye(2)
    exact=evaluate(true,true,current,matrix)
    assert exact['rmse']==0 and exact['selection_score']==0
    bad=true.copy();bad[0,1]=-.1
    result=evaluate(true,bad,current,matrix)
    assert result['negative_fraction']==.25
    assert result['mass_sum_mae']>0 and result['elemental_mae']>0
    bad[1,1]=np.nan
    assert evaluate(true,bad,current,matrix)['invalid_row_fraction']==.5
