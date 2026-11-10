import copy
import numpy as np
import torch
from kinetics.flex_model import FlexDeepONet, author_input_range, KerasAdam


def test_parameter_count_matches_paper_and_released_case3():
    model = FlexDeepONet()
    assert sum(p.numel() for p in model.parameters()) == 34038


def test_batched_networks_match_individual_mlp_and_time_transform():
    torch.manual_seed(11)
    model = FlexDeepONet()
    x = torch.randn(5, 19, dtype=torch.float64)
    t = torch.logspace(-7, 0, 5, dtype=torch.float64)
    output = model(x, t)
    def one(net, value, index):
        for i, (w,b) in enumerate(zip(net.weights,net.biases)):
            value=value@w[index]+b[index]
            if i<len(net.weights)-1:value=torch.tanh(value)
        return value
    stretch=torch.nn.functional.softplus(one(model.stretch,x,0))
    for index in range(19):
        b=one(model.branch,x,index)
        tr=one(model.trunk,torch.log(t*stretch[:,index])[:,None],index)
        assert torch.allclose(output[:,index],(b[:,:8]*tr).sum(1)+b[:,8],atol=2e-14,rtol=2e-13)


def test_no_feedback_or_other_query_dependency():
    torch.manual_seed(13)
    model=FlexDeepONet()
    x=torch.randn(4,19,dtype=torch.float64)
    t=torch.tensor([1e-7,1e-5,1e-3,1.],dtype=torch.float64)
    p=model(x,t)
    order=torch.tensor([3,1,0,2])
    assert torch.allclose(p[order],model(x[order],t[order]),atol=2e-14,rtol=2e-13)


def test_adam_checkpoint_continuation_is_exact():
    torch.manual_seed(14)
    model=FlexDeepONet()
    opt=KerasAdam(model.parameters())
    x=torch.randn(8,19,dtype=torch.float64)
    t=torch.logspace(-7,0,8,dtype=torch.float64)
    def step(m,o):
        o.zero_grad();loss=m(x,t).square().mean()+m.regularization();loss.backward();o.step()
    step(model,opt)
    other=copy.deepcopy(model)
    other_opt=KerasAdam(other.parameters());other_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    step(model,opt);step(other,other_opt)
    for a,b in zip(model.parameters(),other.parameters()):assert torch.equal(a,b)


def test_pinned_input_scaling_preserves_constant_zero_columns():
    x=np.zeros((4,19));x[:,0]=[1000,1200,1500,2000];x[:,1]=[.1,.2,.3,.4]
    span=author_input_range(x)
    assert span[0]==1999 and span[1]==.9
    assert np.all(span[2:]==1)
