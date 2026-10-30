import json
from pathlib import Path
import numpy as np
import cantera as ct
from kinetics.multiscale_data import sample_states,integrate_labels
from kinetics.data import mechanism,SPECIES


def config():
    return json.loads(Path('configs/zhang_multiscale_data.json').read_text())


def test_independent_log_sampling_ranges_and_mass():
    cfg=config();x,f=sample_states(300,25,np.random.default_rng(31),cfg)
    assert np.all((x[:,0]>=800)&(x[:,0]<=3100))
    assert np.all((x[:,-1]>=.5*ct.one_atm)&(x[:,-1]<=2*ct.one_atm))
    assert np.all((x[:,9]>=.2)&(x[:,9]<=.9))
    np.testing.assert_allclose(x[:,1:10].sum(axis=1),1,atol=1e-14)
    weights=x[:,1:9]/f[:,None]
    assert weights.min()>=1e-25 and weights.max()<=1
    for species in ('H2','O2'):
        assert weights[:,SPECIES.index(species)].min()>=1e-5
    assert (x[:,2]<1e-16).any()  # genuinely samples the trace-radical scales.
    coarse,_=sample_states(300,1,np.random.default_rng(31),cfg)
    assert (coarse[:,2]<1e-16).sum()==0


def test_hot_two_steps_are_adjacent_fresh_Cantera_labels():
    gas=mechanism();gas.TP=1200,ct.one_atm;gas.set_equivalence_ratio(1,'H2','O2:1,N2:3.76')
    state=np.r_[gas.T,gas.Y,gas.P]
    x,y,q=integrate_labels(state,2,2e-6,gas)
    np.testing.assert_array_equal(x[1,:10],y[0])
    assert x.shape==(2,11) and y.shape==(2,10) and np.isfinite(q).all()
    reference=mechanism();reference.TPY=state[0],state[-1],state[1:10]
    reactor=ct.IdealGasConstPressureReactor(reference,clone=True)
    net=ct.ReactorNet([reactor]);net.rtol=1e-10;net.atol=1e-20;net.advance(4e-6)
    np.testing.assert_allclose(y[1],np.r_[reactor.T,np.maximum(reactor.phase.Y,0)],rtol=1e-8,atol=1e-10)
    assert np.max(np.abs(y[1]-y[0]))>0
