"""One-update causal controls at the same published pretraining checkpoint.

Only existing training and validation archives are read. This is a numerical
transition diagnostic, never a held-out performance or reproduction claim.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import copy,json,time,shutil
import numpy as np
import torch
from kinetics.react_deeponet_v2_model import DetailedReactDeepONet,sample_batch
from kinetics.react_deeponet_rebuild import REP,validation
from kinetics.react_deeponet_experiment import FifthRootScale,write,csv_write,sha


def step_predictions(model,state,tau,phi,steps=5):
    outputs=[]
    for _ in range(steps):state=model(state,tau,phi);outputs.append(state)
    return torch.stack(outputs)


def main():
    source=Path('results/react_deeponet_rebuild/run_001');root=Path('results/react_refinement_transition/run_001')
    if root.exists():raise FileExistsError('Use a new diagnostic root')
    root.mkdir(parents=True);shutil.copyfile(__file__,root/'frozen_source.py')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);start=time.perf_counter()
    checkpoint=source/'models/pretrain_100000.pt';saved=torch.load(checkpoint,weights_only=False)
    metadata=json.loads((source/'preprocessing.json').read_text());scale=FifthRootScale(metadata['minimum'],metadata['maximum']).subset(REP)
    train=np.load(source/'data/train.npz')['states'];val=np.load(source/'data/val.npz')['states'];cases=json.loads((source/'cases.json').read_text())
    states=torch.from_numpy(scale.forward(train[:,:,REP]));v=torch.from_numpy(scale.forward(val[:,:,REP]));phi=torch.tensor([c['phi'] for c in cases if c['split']=='train'])
    vphi=torch.tensor([c['phi'] for c in cases if c['split']=='val']);vg=torch.Generator().manual_seed(19)
    fixed=sample_batch(v,vphi,8192,400,1,vg)
    generator=torch.Generator();generator.set_state(saved['generator_state'])
    i=torch.randint(len(states),(20000,),generator=generator);origin=torch.randint(states.shape[1]-2000,(20000,),generator=generator);offset=torch.randint(0,401,(20000,),generator=generator)
    a=states[i,origin];tau=offset[:,None].float()*1e-7;p=phi[i,None]
    targets=torch.stack([states[i,origin+(k+1)*offset] for k in range(5)])
    model=DetailedReactDeepONet();model.load_state_dict(saved['model']);_,before=validation(model,val,[c for c in cases if c['split']=='val'],scale,fixed)
    gradients={};gradient_stats={}
    for objective in ['single','endpoint','all_steps']:
        model.zero_grad(set_to_none=True)
        predictions=step_predictions(model,a,tau,p)
        loss=(predictions[0]-targets[0]).square().mean() if objective=='single' else ((predictions[-1]-targets[-1]).square().mean() if objective=='endpoint' else (predictions-targets).square().mean())
        loss.backward();g=torch.cat([q.grad.detach().flatten() for q in model.parameters()]);gradients[objective]=g
        gradient_stats[objective]=dict(loss=float(loss.detach()),gradient_L2=float(g.norm()),gradient_max=float(g.abs().max()))
    reference=gradients['single']
    for name,g in gradients.items():gradient_stats[name]['cosine_with_single']=float(torch.nn.functional.cosine_similarity(reference,g,dim=0))
    specs=[dict(name='endpoint_original_lr',objective='endpoint',scale=1.,reset=False),dict(name='endpoint_lr_div10',objective='endpoint',scale=.1,reset=False),dict(name='endpoint_lr_div100',objective='endpoint',scale=.01,reset=False),dict(name='endpoint_reset_lr_div100',objective='endpoint',scale=.01,reset=True),dict(name='all_steps_lr_div10',objective='all_steps',scale=.1,reset=False)]
    rows=[];initial=torch.cat([q.detach().flatten() for q in model.parameters()]);base_lr=.01*.9**(100000/5000)
    for spec in specs:
        model=DetailedReactDeepONet();model.load_state_dict(saved['model']);opt=torch.optim.Adamax(model.parameters(),lr=base_lr*spec['scale'])
        if not spec['reset']:opt.load_state_dict(copy.deepcopy(saved['optimizer']))
        opt.param_groups[0]['lr']=base_lr*spec['scale'];opt.zero_grad(set_to_none=True)
        predictions=step_predictions(model,a,tau,p);loss=(predictions[-1]-targets[-1]).square().mean() if spec['objective']=='endpoint' else (predictions-targets).square().mean()
        loss.backward();opt.step();delta=torch.cat([q.detach().flatten() for q in model.parameters()])-initial
        _,metric=validation(model,val,[c for c in cases if c['split']=='val'],scale,fixed)
        rows.append(dict(**spec,lr=base_lr*spec['scale'],weight_update_L2=float(delta.norm()),weight_update_max=float(delta.abs().max()),**metric))
        write(root/'status.json',dict(phase='single_update_diagnostic',completed=len(rows),total=len(specs),latest=rows[-1]))
    write(root/'analysis.json',dict(training_only_gradient_stats=gradient_stats,baseline_validation=before,one_update_controls=rows,
        source_checkpoint=str(checkpoint),source_checkpoint_sha256=sha(checkpoint),same_training_batch=True,batch_size=20000,test_read=False))
    csv_write(root/'one_update_controls.csv',rows)
    write(root/'complete.json',dict(status='passed',diagnostic_only=True,test_used=False,elapsed_seconds=time.perf_counter()-start,
        output_sha256={str(p.relative_to(root)):sha(p) for p in root.iterdir() if p.is_file() and p.name!='status.json'}))
    print((root/'analysis.json').read_text())


if __name__=='__main__':main()
