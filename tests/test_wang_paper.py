import numpy as np
from kinetics.wang_paper import boxcox,inverse_boxcox,paper_projection,PaperPredictor,BiasCorrectedAdam
from kinetics.study import tf
from kinetics.data import mechanism
from kinetics.physics import element_matrix


def test_literal_boxcox_and_inverse_without_clipping():
    y=np.array([0.,1e-40,1e-10,.01,1.,1.1])
    np.testing.assert_allclose(boxcox(y),(y**.1-1)/.1,atol=0)
    np.testing.assert_allclose(inverse_boxcox(boxcox(y)),y,rtol=1e-10,atol=0)
    assert np.isnan(inverse_boxcox(np.array([-10.1]))[0])
    assert inverse_boxcox(boxcox(np.array([1.1])))[0]>1
    # Literal subtraction has the paper formula's finite precision loss.
    assert inverse_boxcox(boxcox(np.array([1e-300])))[0]==0


def test_projection_matches_paper_matrix_formula():
    rng=np.random.default_rng(4);a=element_matrix(mechanism())
    current=rng.dirichlet(np.ones(9));v=current*1.001;b=a*v**.9
    expected=v+v**.9*(b.T@np.linalg.inv(b@b.T)@(a@(current-v)))
    result=paper_projection(v,current,a)
    np.testing.assert_allclose(result,expected,atol=1e-14)
    np.testing.assert_allclose(a@result,a@current,atol=1e-14)


def test_adam_bias_corrections_and_epsilon_placement():
    variable=tf.Variable([.5,-.3],dtype=tf.float64)
    optimizer=BiasCorrectedAdam([variable]);expected=np.array([.5,-.3]);m=np.zeros(2);v=np.zeros(2)
    # Tiny gradients expose epsilon-placement differences.
    for t,g in enumerate([np.array([1e-10,-2e-10]),np.array([-1e-9,3e-10])],1):
        m=.9*m+.1*g;v=.999*v+.001*g*g
        expected-=1e-4*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
        optimizer.apply([tf.constant(g)],[variable],1e-4)
        np.testing.assert_allclose(variable.numpy(),expected,atol=2e-12,rtol=0)


def test_species_only_direct_target_and_energy_recovery_serialization(tmp_path):
    gas=mechanism();gas.TP=1200.,101325.;gas.set_equivalence_ratio(1.,'H2','O2:1,N2:3.76')
    # Positive trace species avoid a degenerate matrix in this constructed test.
    y=gas.Y+1e-8;y/=y.sum();x=np.r_[1200.,y,101325.][None]
    model=PaperPredictor(x)
    assert model.model.input_shape[-1]==11 and model.model.output_shape[-1]==9
    assert model.model.count_params()==1624009
    np.testing.assert_array_equal(model.targets(np.c_[x[:,0]+100,y[None]]),boxcox(y[None]).astype('float32'))
    weights=model.model.get_weights();weights[-2][:]=0;weights[-1][:]=boxcox(y)
    model.model.set_weights(weights)
    result=model.predict(x)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(model.matrix@result[0,1:],model.matrix@y,atol=1e-12)
    gas.TPY=x[0,0],x[0,-1],y;h0=gas.enthalpy_mass
    gas.TPY=result[0,0],x[0,-1],result[0,1:]
    np.testing.assert_allclose(gas.enthalpy_mass,h0,rtol=1e-9,atol=1e-4)
    path=tmp_path/'paper';model.save(path);restored=PaperPredictor.load(path)
    np.testing.assert_array_equal(result,restored.predict(x))
