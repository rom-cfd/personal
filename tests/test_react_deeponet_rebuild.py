"""Validate the newly recovered method's mathematical and data semantics."""
import json
from pathlib import Path
import cantera as ct
import numpy as np
import pytest
import torch
from kinetics.react_deeponet_diagnostics import GatedMLP
from kinetics.react_deeponet_v2_model import DetailedReactDeepONet,recursive_endpoint,sample_batch
from kinetics.react_deeponet_rebuild import definitions
from kinetics.react_deeponet_rebuild import mathematical_representative
from kinetics.react_deeponet_experiment import FifthRootScale,valid_state


def test_gated_port_matches_independent_numpy_formula():
    torch.manual_seed(12);m=GatedMLP([5,8,8,4]);x=torch.randn((9,5));a=x.numpy()
    transform=lambda layer,value: value@layer.weight.detach().numpy().T+(layer.bias.detach().numpy() if layer.bias is not None else 0)
    u=np.tanh(transform(m.u,a));v=np.tanh(transform(m.v,a));h=a
    for layer in m.gates:
        g=np.tanh(transform(layer,h));h=g*u+(1-g)*v
    expected=transform(m.output,h)
    np.testing.assert_allclose(m(x).detach().numpy(),expected,atol=1e-6,rtol=1e-5)


def test_complete_network_has_paramnet_and_finite_zero_time():
    torch.manual_seed(11);m=DetailedReactDeepONet();x=torch.randn((8,5));t=torch.full((8,1),20e-6)
    a=m(x,t,torch.full((8,1),.8));b=m(x,t,torch.full((8,1),1.1))
    assert sum(p.numel() for p in m.parameters())==42236
    assert not torch.allclose(a,b)
    assert torch.isfinite(m(x,torch.zeros_like(t),torch.ones_like(t))).all()
    a.square().mean().backward()
    for network in [m.branch,m.trunk,m.shift,m.parameter]:assert sum(float(p.grad.abs().sum()) for p in network.parameters() if p.grad is not None)>0


def test_refinement_gradient_reaches_all_five_self_fed_steps():
    class Linear(torch.nn.Module):
        def __init__(self):super().__init__();self.a=torch.nn.Parameter(torch.tensor(1.1,dtype=torch.float64))
        def forward(self,x,t,p):return self.a*x+t
    m=Linear();x=torch.tensor([[.4]],dtype=torch.float64);t=torch.tensor([[.03]],dtype=torch.float64)
    y=recursive_endpoint(m,x,t,t,5);y.sum().backward()
    a=1.1;expected=a**5*.4+.03*sum(a**j for j in range(5));derivative=5*a**4*.4+.03*sum(j*a**(j-1) for j in range(1,5))
    assert float(y)==pytest.approx(expected)
    assert float(m.a.grad)==pytest.approx(derivative)


def test_refinement_targets_stay_within_ic_and_include_identity_labels():
    states=torch.zeros((3,50,5))
    for i in range(3):states[i,:,0]=i;states[i,:,1]=torch.arange(50)
    phi=torch.tensor([.8,1.,1.2]);x,t,p,y=sample_batch(states,phi,1000,5,5,torch.Generator().manual_seed(2))
    assert torch.equal(x[:,0],y[:,0])
    np.testing.assert_allclose((y[:,1]-x[:,1]).numpy(),t[:,0].numpy()/1e-7*5,atol=1e-5)
    assert torch.allclose(p[:,0],phi[x[:,0].long()])
    zero=t[:,0]==0;assert int(zero.sum())>0
    assert torch.equal(x[zero],y[zero])


def test_paper_grid_and_fresh_tests_are_frozen_and_disjoint():
    cfg=json.loads(Path('configs/react_deeponet_v2.json').read_text());cases=definitions(cfg)
    assert len({(c['T0'],c['phi']) for c in cases})==len(cases)
    assert {s:sum(c['split']==s for c in cases) for s in ['train','val','paper_test','fresh_test']}==dict(train=99,val=12,paper_test=111,fresh_test=16)


def test_burke_inert_removal_preserves_air_chemistry():
    root=Path('third_party/burke_2012');a=ct.Solution(str(root/'h2_burke2012_full.yaml'));b=ct.Solution(str(root/'h2_burke2012_H2_air10.yaml'))
    assert b.n_species==10 and b.n_reactions==25
    for gas in [a,b]:gas.TP=1100,ct.one_atm;gas.set_equivalence_ratio(1,'H2','O2:1,N2:3.76')
    np.testing.assert_allclose(a.net_production_rates[[a.species_index(n) for n in b.species_names]],b.net_production_rates,rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(a.enthalpy_mass,b.enthalpy_mass,rtol=1e-13)


def test_finite_unconstrained_curves_do_not_imply_valid_raw_states():
    class NegativeSpecies(torch.nn.Module):
        def forward(self,state,tau,phi):
            output=state.clone();output[:,1]=-1.1;return output
    scale=FifthRootScale([900**.2,0,0,0,0],[3000**.2,.25**.2,.01**.2,.3**.2,.3**.2])
    initial=np.array([1000,.05,.0001,.2,.1]);prediction,physical=mathematical_representative(NegativeSpecies(),scale,initial,1.)
    assert len(prediction)==10001 and np.isfinite(prediction).all()
    assert not physical and np.min(prediction[:,1])<0
    assert not valid_state(prediction[-1])
