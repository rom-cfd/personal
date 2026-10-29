import numpy as np
from kinetics.vector_physics import VectorThermo,project_batch
from kinetics.physics import project_species
from kinetics.data import mechanism


def test_nasa_h_cp_matches_cantera_including_polynomial_switches():
    v=VectorThermo();g=mechanism()
    ts=np.r_[np.linspace(200,6000,50),999.999999,1000.,1000.000001]
    h,cp=v.species_h_cp(ts)
    for k,t in enumerate(ts):
        g.TP=t,101325
        import cantera as ct
        np.testing.assert_allclose(h[k],g.standard_enthalpies_RT*ct.gas_constant*t/g.molecular_weights,atol=2e-7,rtol=2e-13)
        np.testing.assert_allclose(cp[k],g.standard_cp_R*ct.gas_constant/g.molecular_weights,atol=1e-8,rtol=2e-13)


def test_vector_projection_matches_scalar_projection_and_conserves():
    rng=np.random.default_rng(73);v=VectorThermo()
    current=rng.dirichlet(np.ones(9)*.4,size=100)
    proposal=current+rng.normal(0,.01,current.shape)
    actual,falls=project_batch(proposal,current,v.matrix)
    expected=np.array([project_species(p,c,v.matrix) for p,c in zip(proposal,current)])
    np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-7)
    np.testing.assert_allclose(actual@v.matrix.T,current@v.matrix.T,atol=1e-10)
    assert actual.min()>=0


def test_vector_thermal_root_matches_cantera_hpy():
    v=VectorThermo();g=mechanism();rng=np.random.default_rng(13)
    y=rng.dirichlet(np.ones(9),size=100);target_t=rng.uniform(300,4000,100)
    hs,_=v.species_h_cp(target_t);h=np.sum(y*hs,axis=1)
    t=v.temperatures(y,np.full(100,1200.),h,np.full(100,101325.))
    np.testing.assert_allclose(t,target_t,atol=2e-6,rtol=1e-9)
    for k in range(100):
        g.TP=1200,101325;g.HPY=h[k],101325,y[k]
        np.testing.assert_allclose(t[k],g.T,atol=2e-5,rtol=1e-8)
