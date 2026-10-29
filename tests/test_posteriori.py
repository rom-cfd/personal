import numpy as np
from kinetics.posteriori import ignition, metrics
from kinetics.study import rollouts


def test_ignition_crossing_and_no_ignition():
    trace=np.c_[np.arange(1000,1900,100),np.ones((9,9))/9]
    result=ignition(trace,1e-6,1000)
    assert np.isclose(result['tau400'],4e-6)
    assert np.isnan(ignition(trace[:3],1e-6,1000)['tau400'])


def test_incomplete_profiles_cannot_count_as_full_accuracy():
    truth=np.c_[np.linspace(1000,2000,10),np.ones((10,9))/9]
    trace=truth[:3].copy()
    m,_=metrics(trace,truth,1e-6,np.ones((1,9)),9)
    assert not m['complete'] and np.isnan(m['full_species_rmse'])
    assert m['prefix_species_rmse']==0
    assert m['ignition_status']=='integration_failed_before_ignition'
    assert np.isclose(m['coverage'],2/9)


def test_rollout_is_self_fed_without_teacher_forcing():
    class Dummy:
        def predict(self,x):
            p=x[:,:10].copy();p[:,0]+=10
            return p
    states=np.c_[np.linspace(1000,5000,5),np.ones((5,9))/9][None,:]
    data=dict(states=states,pressures=np.array([101325]),split=np.array(['test']),element_matrix=np.ones((1,9)))
    _,traces=rollouts(Dummy(),data,'test')
    np.testing.assert_allclose(traces[0][1][:,0],[1000,1010,1020,1030,1040])
