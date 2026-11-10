"""Capacity/learning check on ONE seen trajectory; no generalization claim."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import json,time,shutil
import numpy as np
import torch
import cantera as ct
from kinetics.react_deeponet_rebuild import train_operator,mathematical_representative,macro,REP
from kinetics.react_deeponet_experiment import FifthRootScale,write,sha,scores

root=Path('results/react_deeponet_single_ic/run_001')
if root.exists():raise FileExistsError('A new capacity-check root is required')
root.mkdir(parents=True);(root/'models').mkdir();shutil.copyfile(__file__,root/'frozen_probe.py')
torch.set_num_threads(2);torch.set_num_interop_threads(1)
cfg=json.loads(Path('configs/react_deeponet_v2.json').read_text())
cfg['reported'].update(pretrain_updates=10000,refine_updates=4000,batch_size=1024,initial_lr=.003,decay_steps=1000,decay_rate=.8)
cfg['execution'].update(checkpoint_every=500,history_every=100)
write(root/'config.json',dict(**cfg,capacity_probe_only=True,validation_equals_training_trajectory=True,test_used=False,
    note='Accelerated learning-rate schedule is a numerical learning diagnostic, not the author training protocol'))
mechanism=Path('third_party/burke_2012/h2_burke2012_H2_air10.yaml')
gas=ct.Solution(str(mechanism));gas.TP=1080,ct.one_atm;gas.set_equivalence_ratio(1,'H2','O2:1,N2:3.76')
r=ct.IdealGasConstPressureReactor(gas,clone=False);net=ct.ReactorNet([r]);net.rtol=1e-10;net.atol=1e-20;net.max_time_step=1e-7
truth=np.empty((10001,11));truth[0]=np.r_[gas.T,gas.Y]
for j in range(1,10001):net.advance(j*1e-7);truth[j]=np.r_[gas.T,gas.Y]
assert truth[:,1:].min()>=-1e-14;truth[:,1:]=np.maximum(truth[:,1:],0)
scale=FifthRootScale.fit(truth[None,:,:]);write(root/'preprocessing.json',scale.json())
cases=[dict(case_id=0,split='train',T0=1080.,phi=1.),dict(case_id=1,split='val',T0=1080.,phi=1.)]
start=time.perf_counter();model=train_operator(root,cfg,truth[None,:,:],truth[None,:,:],cases,scale.subset(REP),start+600)
prediction,physical=mathematical_representative(model,scale.subset(REP),truth[0,REP],1.)
values=scores(truth[:,REP],prediction) if len(prediction)==len(truth) else None
np.savez_compressed(root/'traces.npz',reference=truth[:,REP],prediction=prediction)
selection=json.loads((root/'selection_frozen.json').read_text())
write(root/'complete.json',dict(status='passed',capacity_probe_only=True,seen_trajectory=True,test_used=False,
    elapsed_seconds=time.perf_counter()-start,complete=len(prediction)==len(truth),strict_state_validity=physical,
    NRMSE=values.tolist() if values is not None else None,joint5=bool(values is not None and np.max(values)<=.1),
    update_target_completed=selection['update_target_completed'],source_mechanism_sha256=sha(mechanism)))
print((root/'complete.json').read_text())
