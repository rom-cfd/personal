from pathlib import Path
import numpy as np
import pytest
import torch
from kinetics.react_stepper_refinement import (curriculum,sample_operator,endpoint_loss,sample_reconstruction,
    check_publication,operator_rank,combined_rank)

def test_curriculum_changes_only_at_declared_boundaries():
    assert [curriculum(v,2) for v in [1,2,3,4,5,12]]==[1,1,2,2,5,5]

def test_targets_stay_on_same_trajectory_at_final_recursive_time():
    x=torch.arange(3*3001,dtype=torch.float32).reshape(3,3001,1).expand(3,3001,5)
    p=torch.arange(3,dtype=torch.float32)
    for mode in ['uniform','fixed20']:
        a,t,phi,y=sample_operator(x,p,100,torch.Generator().manual_seed(9),mode,5)
        offset=torch.round(t[:,0]/1e-7)
        torch.testing.assert_close(y[:,0],a[:,0]+5*offset)
        assert ((a[:,0]//3001)==phi[:,0]).all() and (y[:,0]//3001==phi[:,0]).all()
        assert (offset>0).all()
        if mode=='fixed20':assert (offset==200).all()

def test_candidate_draws_match_case_and_origin():
    x=torch.arange(3*3001,dtype=torch.float32).reshape(3,3001,1).expand(3,3001,5);p=torch.arange(3,dtype=torch.float32)
    a=sample_operator(x,p,100,torch.Generator().manual_seed(8),'uniform',5)
    b=sample_operator(x,p,100,torch.Generator().manual_seed(8),'fixed20',5)
    torch.testing.assert_close(a[0],b[0]);torch.testing.assert_close(a[2],b[2])

def test_endpoint_loss_differentiates_through_all_predictions():
    class Multiply(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.1,dtype=torch.float64))
        def forward(self,x,t,p):return self.weight*x
    m=Multiply();x=torch.ones((3,5),dtype=torch.float64)
    loss=endpoint_loss(m,x,torch.zeros((3,1)),torch.ones((3,1)),torch.zeros_like(x),5);loss.backward()
    assert np.isclose(m.weight.grad.item(),10*1.1**9)

def test_early_sampling_increases_initial_time_coverage_without_new_states():
    x=torch.arange(10001,dtype=torch.float32).reshape(1,10001,1).expand(2,10001,11)
    a=sample_reconstruction(x,10000,torch.Generator().manual_seed(4),False)
    b=sample_reconstruction(x,10000,torch.Generator().manual_seed(4),True)
    assert .08<float((a[:,0]<=1000).float().mean())<.12
    assert .52<float((b[:,0]<=1000).float().mean())<.58
    assert b.min()>=0 and b.max()<=10000

def test_balanced_combination_rank_does_not_trade_away_extrapolation():
    a=dict(balanced_rep=.5,rep_both=18,balanced_full=.25,full_both=16,full=20,complete=24,median_worst5=.2,median_worst9=.3)
    b=dict(a,balanced_full=0.,full_both=18)
    assert combined_rank(a)<combined_rank(b)
    assert operator_rank(a)[0]==-.5

def test_missing_publication_never_opens_training_gate(tmp_path):
    with pytest.raises(FileNotFoundError):check_publication(tmp_path,tmp_path/'missing.json',dict(fixture=False))
