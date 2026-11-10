"""Controlled seen-IC sampling diagnostic; this is not held-out accuracy."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import json,time,shutil
import numpy as np
import torch
import cantera as ct
from kinetics.react_deeponet_v2_model import DetailedReactDeepONet,recursive_endpoint
from kinetics.react_deeponet_rebuild import REP,mathematical_representative
from kinetics.react_deeponet_experiment import FifthRootScale,write,csv_write,sha,scores


def main():
    root=Path('results/react_deeponet_sampling_probe/run_001')
    if root.exists():raise FileExistsError('Never restart a completed diagnostic')
    root.mkdir(parents=True);shutil.copyfile(__file__,root/'frozen_probe.py')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    source=Path('results/react_deeponet_single_ic/run_001')
    assert json.loads((source/'complete.json').read_text())['status']=='passed'
    checkpoint=source/'models/pretrain_010000.pt'
    metadata=json.loads((source/'preprocessing.json').read_text());scale=FifthRootScale(metadata['minimum'],metadata['maximum']).subset(REP)
    reference=np.load(source/'traces.npz')['reference'];states=torch.from_numpy(scale.forward(reference))
    write(root/'config.json',dict(capacity_diagnostic_only=True,seen_trajectory=True,test_used=False,
        shared_checkpoint=str(checkpoint),shared_checkpoint_sha256=sha(checkpoint),initial_lr=.001,
        optimizer='Adamax restored from the same checkpoint in both arms',updates_K1=5000,updates_K5=3000,batch=1024,
        sampler_control='uniform starts on the seen trajectory',sampler_treatment='30% uniform, 60% starts within first 0.1ms, 10% initial state',
        note='Only the training-start distribution differs. This ablation is not the paper training protocol or generalization.'))
    history=[];summary=[];started=time.perf_counter()
    for name,focused in [('uniform',False),('initial_transient_focused',True)]:
        folder=root/name;folder.mkdir();torch.manual_seed(11);model=DetailedReactDeepONet()
        saved=torch.load(checkpoint,weights_only=False);model.load_state_dict(saved['model'])
        opt=torch.optim.Adamax(model.parameters(),lr=.001);opt.load_state_dict(saved['optimizer'])
        generator=torch.Generator().manual_seed(123);active=0.;best=None
        for update in range(1,8001):
            if time.perf_counter()-started>900:raise TimeoutError('Sampling diagnostic total guard')
            k=1 if update<=5000 else 5;tick=time.perf_counter();n=1024
            # Draw the same arrays in both arms, even where the control ignores them.
            start=torch.randint(len(states)-400*k,(n,),generator=generator)
            early=torch.randint(min(1001,len(states)-400*k),(n,),generator=generator)
            mixture=torch.rand(n,generator=generator)
            if focused:
                start=torch.where(mixture<.6,early,start);start=torch.where(mixture>.9,torch.zeros_like(start),start)
            offset=torch.randint(0,401,(n,),generator=generator);tau=offset[:,None].float()*1e-7
            target=states[start+k*offset];opt.param_groups[0]['lr']=.001*.9**((update-1)/1000)
            opt.zero_grad(set_to_none=True);loss=torch.mean((recursive_endpoint(model,states[start],tau,torch.ones((n,1)),k)-target)**2)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite diagnostic loss')
            loss.backward();opt.step();active+=time.perf_counter()-tick
            if update==1 or update%100==0:history.append(dict(arm=name,update=update,K=k,loss=float(loss.detach()),lr=opt.param_groups[0]['lr'],active_seconds=active))
            if update==1 or update%1000==0:
                model.eval();prediction,physical=mathematical_representative(model,scale,reference[0],1.);model.train()
                error=scores(reference,prediction) if len(prediction)==len(reference) else None
                rank=(int(len(prediction)!=len(reference)),float(np.max(error)) if error is not None else float('inf'))
                if best is None or rank<best[0]:
                    best=(rank,update,error,physical);torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),update=update),folder/'selected.pt')
                    np.savez_compressed(folder/'traces.npz',reference=reference,prediction=prediction)
                write(root/'status.json',dict(arm=name,update=update,K=k,elapsed_seconds=time.perf_counter()-started,latest_worst_NRMSE=rank[1],best_worst_NRMSE=best[0][1]))
                csv_write(root/'training.csv',history)
        error=best[2]
        summary.append(dict(arm=name,selected_update=best[1],updates=8000,active_seconds=active,numerically_complete=best[0][0]==0,
            joint5_curve=bool(error is not None and np.max(error)<=.1),strict_state_valid=best[3],worst_NRMSE=best[0][1],
            **{v:float(error[i]) if error is not None else None for i,v in enumerate(['T','H2','H','O2','H2O'])}))
        csv_write(root/'summary.csv',summary)
    write(root/'complete.json',dict(status='passed',capacity_diagnostic_only=True,seen_trajectory=True,test_used=False,elapsed_seconds=time.perf_counter()-started,
        output_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
