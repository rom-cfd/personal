import numpy as np
import torch

from kinetics.react_deeponet_experiment import FifthRootScale
from kinetics.react_stepper_reproduction import feedback,score,state_valid

class AffineOperator(torch.nn.Module):
    def __init__(self): super().__init__(); self.calls=[]
    def forward(self,state,tau,phi):
        self.calls.append(state.clone())
        return state+.01

class ConstantReconstruction(torch.nn.Module):
    def forward(self,state): return torch.zeros((len(state),6))

def setup():
    lo=np.zeros(11);hi=np.ones(11);hi[0]=10
    scale=FifthRootScale(lo,hi)
    state=scale.inverse(np.zeros((1,11)))
    state[:,0]=1000
    return scale,state

def test_every_next_input_is_the_predicted_endpoint():
    scale,state=setup();operator=AffineOperator()
    trace,info=feedback(operator,ConstantReconstruction(),scale,state,[1.],2,6)
    assert len(operator.calls)==3
    for i in range(1,3):
        torch.testing.assert_close(operator.calls[i],operator.calls[0]+i*.01)
    assert info['first_invalid_step'][0]==-1
    assert np.all(trace[0,2:,0]>trace[0,0,0])

def test_negative_interior_is_exposed_but_never_fed_back():
    scale,state=setup()
    class InteriorNegative(AffineOperator):
        def forward(self,s,t,p):
            out=s.clone();out[t[:,0]<1.5e-7,1]=-1.1
            return out
    trace,info=feedback(InteriorNegative(),ConstantReconstruction(),scale,state,[1.],2,4)
    assert trace[0,1,1]<0
    assert not state_valid(trace[0]).all()
    assert info['raw_boundary_valid'][0]

def test_negative_endpoint_is_a_failed_actual_solver():
    scale,state=setup()
    class BadEnd(AffineOperator):
        def forward(self,s,t,p):
            out=s.clone();out[:,1]=-1.1
            return out
    _,info=feedback(BadEnd(),ConstantReconstruction(),scale,state,[1.],2,4)
    assert not info['raw_boundary_valid'][0]
    assert info['first_invalid_step'][0]==1

def test_normalized_error_uses_cantera_rms_and_zero_rms_is_undefined():
    truth=np.array([[2.,0.],[2.,0.]])
    rmse,rms,ratio=score(truth,np.array([[3.,1.],[3.,1.]]))
    assert ratio[0]==.5 and np.isnan(ratio[1])
    assert rmse[0]==1 and rms[0]==2
