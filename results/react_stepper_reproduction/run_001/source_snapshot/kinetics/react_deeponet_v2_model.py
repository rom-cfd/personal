"""Rebuild the detailed React-DeepONet method in Kumar thesis chapter 3.

The original private code is unavailable. Widths, modified MLP, ParamNet,
Adamax/decay, and final-endpoint recursive refinement come from that source.
The three-way coefficient product and the finite zero-time log convention are
explicit reconstruction choices and must not be presented as verified code.
"""
import numpy as np
import torch
from torch import nn
from .react_deeponet_diagnostics import GatedMLP


class DetailedReactDeepONet(nn.Module):
    def __init__(self, tau_min=1e-7, tau_max=40e-6, phi_min=.75, phi_max=1.15):
        super().__init__()
        self.branch=GatedMLP([5,50,50,50,200])
        self.trunk=GatedMLP([1,50,50,50,200])
        self.shift=GatedMLP([5,10,10,1])
        self.parameter=GatedMLP([1,50,200])
        self.bias=nn.Parameter(torch.zeros(5))
        self.register_buffer('log_min',torch.tensor(np.log(tau_min),dtype=torch.float32))
        self.register_buffer('log_span',torch.tensor(np.log(tau_max/tau_min),dtype=torch.float32))
        self.register_buffer('phi_min',torch.tensor(phi_min,dtype=torch.float32))
        self.register_buffer('phi_span',torch.tensor(phi_max-phi_min,dtype=torch.float32))

    def forward(self,state,tau,phi):
        time_coordinate=2*(torch.log(torch.clamp_min(tau,1e-12))-self.log_min)/self.log_span-1
        location=time_coordinate+self.shift(state)
        condition=2*(phi-self.phi_min)/self.phi_span-1
        a=self.branch(state).reshape(-1,5,40)
        b=self.trunk(location).reshape(-1,5,40)
        p=self.parameter(condition).reshape(-1,5,40)
        return (a*b*p).sum(-1)+self.bias


class BoundPhiOperator(nn.Module):
    """Fixed known IC parameter for a trajectory; contains no reference states."""
    def __init__(self,model,phi):
        super().__init__();self.model=model;self.phi=float(phi)

    def forward(self,state,tau):
        return self.model(state,tau,torch.full((len(state),1),self.phi,dtype=state.dtype,device=state.device))


def recursive_endpoint(model,state,tau,phi,steps):
    for _ in range(steps):state=model(state,tau,phi)
    return state


def sample_batch(states,phis,n,maximum_offset,steps,generator,dt=1e-7):
    # Start points are restricted by the largest possible final l*tau, so all
    # endpoints remain inside the SAME trajectory, including zero-time samples.
    i=torch.randint(len(states),(n,),generator=generator)
    start=torch.randint(states.shape[1]-maximum_offset*steps,(n,),generator=generator)
    length=torch.randint(0,maximum_offset+1,(n,),generator=generator)
    return (states[i,start],length[:,None].float()*dt,phis[i,None],states[i,start+steps*length])
