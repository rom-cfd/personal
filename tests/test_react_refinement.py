"""Semantics of continuation controls, unseen splits and strict scoring."""
from pathlib import Path
import json
import numpy as np
import torch
import pytest
from kinetics.react_refinement import fresh_cases,sample_targets,objective,profile_scores,ranking,arc_time_pool,sample_reconstruction


@pytest.mark.parametrize('family',['endpoint','all_steps'])
def test_recursive_gradient_matches_independent_closed_form(family):
    class Linear(torch.nn.Module):
        def __init__(self):super().__init__();self.a=torch.nn.Parameter(torch.tensor(.9,dtype=torch.float64))
        def forward(self,x,t,p):return self.a*x+t
    m=Linear();x=torch.tensor([[.3]],dtype=torch.float64);tau=torch.tensor([[.02]],dtype=torch.float64);targets=torch.tensor([[[.31]],[[.32]],[[.33]],[[.34]],[[.35]]],dtype=torch.float64)
    loss=objective(m,x,tau,tau,targets,family);loss.backward()
    values=[.9**k*.3+.02*sum(.9**j for j in range(k)) for k in range(1,6)]
    derivatives=[k*.9**(k-1)*.3+.02*sum(j*.9**(j-1) for j in range(1,k)) for k in range(1,6)]
    terms=[2*(y-(.30+.01*k))*dy for k,(y,dy) in enumerate(zip(values,derivatives),1)]
    assert float(m.a.grad)==pytest.approx(terms[-1] if family=='endpoint' else np.mean(terms),rel=1e-10,abs=1e-12)


def test_targets_keep_all_intermediate_times_in_same_trajectory():
    a=torch.zeros((3,60,5))
    for i in range(3):a[i,:,0]=i;a[i,:,1]=torch.arange(60)
    phi=torch.tensor([.8,1.,1.2]);x,t,p,y=sample_targets(a,phi,1000,5,torch.Generator().manual_seed(3))
    for k in range(5):
        assert torch.equal(x[:,0],y[k,:,0])
        torch.testing.assert_close(y[k,:,1]-x[:,1],t[:,0]/1e-7*(k+1),atol=1e-5,rtol=1e-5)
    assert torch.equal(p[:,0],phi[x[:,0].long()])
    zero=t[:,0]==0;assert zero.any()
    assert torch.equal(y[:,zero],x[zero].expand(5,-1,-1))


def test_partial_rollout_has_early_score_but_no_full_score():
    ref=np.ones((10001,11));p=ref[:1100].copy();d=profile_scores(ref,p)
    assert not d['complete'] and d['full'] is None and not d['pass_full'] and not d['pass_both']
    np.testing.assert_array_equal(d['early'],np.zeros(11))


def test_selection_balances_both_validation_domains_before_total_passes():
    a={'interpolation':dict(total=12,complete=12,pass_full=12,pass_both=12),'extrapolation':dict(total=12,complete=12,pass_full=0,pass_both=0)}
    b={'interpolation':dict(total=12,complete=12,pass_full=8,pass_both=8),'extrapolation':dict(total=12,complete=12,pass_full=3,pass_both=3)}
    assert ranking(b,[.1]*24)<ranking(a,[.01]*24)


def test_fresh_holdout_and_new_validation_never_overlap_old_cases():
    cfg=json.loads(Path('configs/react_refinement.json').read_text());old=json.loads(Path(cfg['source_root'],'cases.json').read_text());new=fresh_cases(cfg,old)
    assert new==fresh_cases(cfg,old)
    assert len(new)==44
    keys={(c['T0'],c['phi']) for c in new};assert len(keys)==44
    assert not keys & {(c['T0'],c['phi']) for c in old}
    assert {s:sum(c['split']==s for c in new) for s in ['val','test_interpolation','test_extrapolation']}==dict(val=12,test_interpolation=16,test_extrapolation=16)


def test_arc_pool_emphasizes_changes_and_preserves_boundaries_and_ic():
    a=np.zeros((2,101,11),dtype=np.float32);a[:,50:,1]=1;a[1,:,0]=2
    pool=arc_time_pool(a,16)
    assert torch.equal(pool[:,0],torch.zeros(2,dtype=torch.int64))
    assert torch.equal(pool[:,-1],torch.full((2,),100,dtype=torch.int64))
    assert int((pool==50).sum())>=24
    states=torch.from_numpy(a);x=sample_reconstruction(states,pool,1000,torch.Generator().manual_seed(7),True)
    assert set(x[:,0].tolist())=={0.,2.}
