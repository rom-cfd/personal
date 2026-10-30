import numpy as np
from kinetics.enthalpy_temperature import EnthalpyTemperature
from kinetics.data import mechanism


def test_temperature_only_closure_matches_cantera_and_preserves_raw_species():
    p=EnthalpyTemperature(None);g=mechanism();rng=np.random.default_rng(81)
    y=rng.dirichlet(np.ones(9),20);x=np.c_[rng.uniform(1200,2500,20),y,np.full(20,101325.)]
    # Non-unit mass sums exercise Cantera normalization without changing ANN Y.
    proposed=np.c_[np.full(20,9999.),y*rng.uniform(.9,1.1,(20,1)),x[:,-1]]
    out=p.close(proposed,x)
    np.testing.assert_array_equal(out[:,1:],proposed[:,1:])
    for k in range(20):
        g.TPY=x[k,0],x[k,-1],x[k,1:10];h=g.enthalpy_mass
        g.HPY=h,x[k,-1],proposed[k,1:10]
        np.testing.assert_allclose(out[k,0],g.T,atol=2e-5,rtol=1e-8)
        hs,_=p.thermo.species_h_cp(out[k:k+1,0]);yn=out[k,1:10]/out[k,1:10].sum()
        np.testing.assert_allclose(hs[0]@yn,h,atol=2e-5,rtol=1e-11)


def test_invalid_composition_fails_without_clipping_or_projection():
    p=EnthalpyTemperature(None);x=np.c_[[1200.],np.full((1,9),1/9),[101325.]]
    proposed=x.copy();proposed[0,1]=-1e-4
    out=p.close(proposed,x)
    assert np.isnan(out[0,0]);np.testing.assert_array_equal(out[:,1:],proposed[:,1:])
