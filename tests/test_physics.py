import numpy as np
from kinetics.physics import PhysicalPredictor, project_species, element_matrix
from kinetics.data import mechanism


def test_projection_positive_conserving_and_identity():
    rng=np.random.default_rng(46);a=element_matrix(mechanism())
    for _ in range(50):
        current=rng.dirichlet(np.ones(9)*.4)
        proposal=current+rng.normal(0,.03,9)
        p=project_species(proposal,current,a)
        assert p.min()>=0
        np.testing.assert_allclose(a@p,a@current,atol=1e-9)
        np.testing.assert_allclose(p.sum(),1,atol=1e-9)
        same=project_species(current,current,a)
        np.testing.assert_allclose(same,current,atol=1e-12)


def test_physics_wrapper_conserves_enthalpy_and_does_not_integrate_chemistry():
    class Dummy:
        def predict(self,x):
            p=x[:,:10].copy();p[:,0]+=1000;p[:,1:]*=1.01;p[:,2]-=.001
            return p
    g=mechanism();g.TP=1200,101325;g.set_equivalence_ratio(1,'H2','O2:1,N2:3.76')
    y=g.Y.copy();h=g.enthalpy_mass;x=np.r_[g.T,y,g.P][None,:]
    pred=PhysicalPredictor(Dummy()).predict(x)[0]
    assert np.all(np.isfinite(pred)) and pred[1:].min()>=0
    np.testing.assert_allclose(element_matrix(g)@pred[1:],element_matrix(g)@y,atol=1e-9)
    g.TPY=pred[0],x[0,-1],pred[1:]
    np.testing.assert_allclose(g.enthalpy_mass,h,rtol=1e-9)
