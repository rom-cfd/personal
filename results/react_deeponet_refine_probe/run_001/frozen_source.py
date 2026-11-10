"""Isolate recursive refinement from the same frozen operator weights."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
import torch
from .react_deeponet_diagnostics import create
from .react_deeponet_experiment import FifthRootScale,write,csv_write,sha,scores,dense_rollout,representative_rollout


def draw(states,n,steps,generator):
    ic=torch.randint(len(states),(n,),generator=generator)
    start=torch.randint(states.shape[1]-1000,(n,),generator=generator)
    h=torch.randint(1,201,(n,),generator=generator)
    return states[ic,start],h[:,None].float()*1e-7,states[ic,start+h],states[ic,start+5*h]


def unroll(model,x,t,steps):
    for _ in range(steps):x=model(x,t)
    return x


def run(root,updates,batch,budget):
    if root.exists():raise FileExistsError('New output required')
    root.mkdir(parents=True);shutil.copyfile(__file__,root/'frozen_source.py')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    source=Path('results/react_deeponet_reconstruction/run_001')
    initial=Path('results/react_deeponet_diagnostics/run_001/gated_adamax_1e2_decay/best.pt')
    saved=torch.load(initial,weights_only=False)
    cases=json.loads((source/'cases.json').read_text())
    with np.load(source/'trajectories.npz') as f:physical=f['states']
    train=physical[[r['case_id'] for r in cases if r['split']=='train']]
    val=physical[[r['case_id'] for r in cases if r['split']=='val']]
    p=json.loads((source/'preprocessing.json').read_text());fullscale=FifthRootScale(p['minimum'],p['maximum']);indices=np.array([0,1,2,4,6]);scale=fullscale.subset(indices)
    a=torch.from_numpy(scale.forward(train[:,:,indices]));b=torch.from_numpy(scale.forward(val[:,:,indices]))
    fixed=draw(b,4096,5,torch.Generator().manual_seed(667))
    started=time.perf_counter();deadline=started+budget;summary=[];arrays={}
    write(root/'config.json',dict(starting_checkpoint=str(initial),starting_sha256=sha(initial),starting_update=saved['update'],
        cases=['single_step_control','five_step_refinement'],updates=updates,batch=batch,lr=.001,decay_steps=5000,decay_rate=.9,
        optimizer_reset=True,optimizer='Adamax',loss='K1 control versus final K5 endpoint only, with BPTT through all predictions',
        trajectory_source=str(source),test_used=False,budget_seconds=budget))
    for label,k in [('single_step_control',1),('five_step_refinement',5)]:
        model=create(saved['spec']);model.load_state_dict(saved['model']);opt=torch.optim.Adamax(model.parameters(),lr=.001)
        gen=torch.Generator().manual_seed(171);history=[];best_rank=None;directory=root/label;directory.mkdir()
        for update in range(1,updates+1):
            if time.perf_counter()>deadline:raise TimeoutError('Refinement diagnosis budget exhausted')
            opt.param_groups[0]['lr']=.001*.9**((update-1)/5000)
            x,t,y1,y5=draw(a,batch,k,gen)
            opt.zero_grad(set_to_none=True);pred=unroll(model,x,t,k);loss=torch.mean((pred-(y1 if k==1 else y5))**2)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite refinement loss')
            loss.backward();opt.step()
            if update==1 or update%100==0:history.append(dict(update=update,loss=float(loss.detach()),lr=opt.param_groups[0]['lr']))
            if update==1 or update%500==0:
                complete=0;passed=0;errors=[]
                with torch.inference_mode():
                    fx,ft,fy1,fy5=fixed;mse1=float(torch.mean((model(fx,ft)-fy1)**2));mse5=float(torch.mean((unroll(model,fx,ft,5)-fy5)**2))
                for truth in val:
                    ref=truth[::200][:,indices];p=representative_rollout(model,scale,ref[0],25,20e-6)
                    if len(p)==len(ref):
                        complete+=1;e=float(np.max(scores(ref,p)));errors.append(e);passed+=int(e<=.1)
                rank=(-complete,-passed,float(np.median(errors)) if errors else float('inf'),mse5)
                if best_rank is None or rank<best_rank:
                    best_rank=rank;torch.save(dict(model=model.state_dict(),update=update,spec=saved['spec'],mse1=mse1,mse5=mse5),directory/'best.pt')
                write(root/'status.json',dict(phase=label,update=update,target_updates=updates,window_MSE=mse1,five_step_MSE=mse5,macro_complete=complete,macro_joint5=passed,elapsed_seconds=time.perf_counter()-started))
                csv_write(directory/'training.csv',history)
        best=torch.load(directory/'best.pt',weights_only=False);model.load_state_dict(best['model']);model.eval();rows=[]
        for i,truth in enumerate(val):
            pred,info=dense_rollout(model,None,truth[0],fullscale,indices,np.array([3,5,7,8,9,10]),200,1e-7,'representative_raw',source/'mechanism.yaml')
            values=scores(truth[:,indices],pred) if info['complete'] else None
            rows.append(dict(validation_case=i,complete=info['complete'],points=len(pred),worst5=float(np.max(values)) if values is not None else None,
                joint5=bool(values is not None and np.max(values)<=.1)))
            arrays[f'{label}_val{i}']=pred;arrays[f'reference_val{i}']=truth[:,indices]
        csv_write(directory/'validation.csv',rows)
        rr=dict(candidate=label,K=k,selected_update=best['update'],complete=sum(r['complete'] for r in rows),joint5=sum(r['joint5'] for r in rows),total=len(rows),
            median_worst5=float(np.median([r['worst5'] for r in rows if r['complete']])) if any(r['complete'] for r in rows) else None,window_MSE=best['mse1'],five_step_MSE=best['mse5'])
        summary.append(rr);csv_write(root/'summary.csv',summary);print(json.dumps(rr),flush=True)
    np.savez_compressed(root/'validation_traces.npz',**arrays)
    write(root/'complete.json',dict(status='passed',test_used=False,elapsed_seconds=time.perf_counter()-started,output_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='status.json'}))
    write(root/'status.json',dict(phase='complete',status='passed'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--updates',type=int,default=8000);p.add_argument('--batch',type=int,default=1024);p.add_argument('--budget',type=float,default=900)
    args=p.parse_args();run(args.root,args.updates,args.batch,args.budget)
