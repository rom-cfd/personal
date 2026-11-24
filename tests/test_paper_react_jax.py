"""Checks of feedback, data ownership, and source-network semantics."""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from kinetics.paper_react_jax import initialize,network,model,cases_for,encode,decode,nrmse,draw

def config():return json.loads(Path('configs/paper_react_2023_jax.json').read_text())

def test_constant_canonical_gate_and_independent_numpy_reference():
    p=initialize([2,3,3,4],random.PRNGKey(7));x=np.array([[.1,.2],[.8,-.3]],np.float32)
    dense,(uw,ub),(vw,vb)=jax.device_get(p);u=np.tanh(x@uw+ub);v=np.tanh(x@vw+vb);h=x.copy()
    for w,_ in dense[:-1]:g=np.tanh(h@w);h=g*u+(1-g)*v
    expected=h@dense[-1][0]+dense[-1][1]
    np.testing.assert_allclose(network(p,jnp.asarray(x)),expected,rtol=2e-6,atol=1e-7)
    # The credited base implementation deliberately does not use gate biases.
    edited=([ (w,b+100) if i<len(dense)-1 else (w,b) for i,(w,b) in enumerate(dense)],(uw,ub),(vw,vb))
    np.testing.assert_allclose(network(edited,jnp.asarray(x)),expected,rtol=2e-6,atol=1e-7)

def test_no_overlap_and_declared_seen_paper_example():
    rows=cases_for(config());pairs=[(x['T0'],x['phi']) for x in rows]
    assert len(set(pairs))==len(pairs)
    assert sum(x['split']=='train' for x in rows)==25
    assert any(x['split']=='train' and x['T0']==1065 and x['phi']==1 for x in rows)
    assert all(x['phi'] in [.6,.7,1.3] for x in rows if x['split']=='paper_test')

def test_serialized_normalization_and_rms_ratio():
    a=np.array([[1000.,0.],[2000.,1e-5]]);lo=(a**.2).min(0);span=(a**.2).max(0)-lo
    np.testing.assert_allclose(decode(encode(a,lo,span),lo,span),a,rtol=1e-6,atol=1e-14)
    np.testing.assert_allclose(nrmse(a,a*1.05),[.05,.05],rtol=1e-12)

def test_window_labels_are_future_states_not_increments():
    states=jnp.arange(2*500*5,dtype=jnp.float32).reshape(2,500,5)
    _,(x,t,y)=draw(random.PRNGKey(8),states,128,200,1e-7)
    assert np.all(np.asarray(t)>0) and np.all(np.asarray(t)<=20e-6+1e-12)
    np.testing.assert_allclose(np.asarray(y-x),np.repeat(np.round(np.asarray(t)/1e-7)*5,5,axis=1),atol=1e-3)

def test_five_dot_segments_and_time_gradient():
    p=tuple(initialize(s,random.PRNGKey(k)) for k,s in enumerate([[5,80,80,200],[1,80,80,200],[5,10,10,1]]))
    x=jnp.full((2,5),.2);t=jnp.array([[1e-6],[20e-6]])
    assert model(p,x,t).shape==(2,5)
    g=jax.grad(lambda z:jnp.sum(model(p,x,z)))(t)
    assert np.isfinite(g).all() and np.linalg.norm(g)>0
