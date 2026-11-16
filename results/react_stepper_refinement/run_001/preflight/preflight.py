from pathlib import Path
import os,time,json,copy,sys
sys.path.insert(0,'/workspace/personal')
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',
                  MPLCONFIGDIR='/tmp/react-stepper-refinement-mpl')
started=time.perf_counter()
import torch,numpy as np
from kinetics.react_stepper_refinement import (sample_operator,sample_reconstruction,curriculum,training_losses,load_operator)
from kinetics.react_deeponet_experiment import FifthRootScale
from scripts.monitor_react_stepper_refinement import inspect,sha

follow=Path('/workspace/react_stepper_refinement_followup/run_001');fixture=follow/'fixture_002'
assert not (follow/'preflight.json').exists()
report=inspect(fixture,external=False);assert report['verified_complete']
cfg=json.loads((fixture/'config.json').read_text())
raw=np.load(fixture/'data/train.npz')['states'];prep=json.loads((fixture/'preprocessing.json').read_text())
scale=FifthRootScale(prep['minimum'],prep['maximum']);encoded=torch.from_numpy(scale.forward(raw));native=encoded[:,:,[0,1,2,4,6]].contiguous()
cases=json.loads((fixture/'training_cases.json').read_text());phi=torch.tensor([c['phi'] for c in cases])
base=torch.load(fixture/'inputs/operator.pt',weights_only=False);promoted=json.loads((fixture/'promotion_frozen.json').read_text())['promoted']
continuity=[]
for spec in promoted:
    saved=torch.load(fixture/'models'/spec['name']/'last.pt',weights_only=False);n=saved['new_update']
    assert n==cfg['target_new_updates']
    for key,state in saved['optimizer']['state'].items():
        assert int(state['step'])==int(base['optimizer']['state'][key]['step'])+n
    gen=torch.Generator();gen.set_state(base['generator_state'])
    for u in range(1,n+1):sample_operator(native,phi,cfg['batch_size'],gen,spec['mode'],curriculum(u,cfg['curriculum_stage_updates']))
    assert torch.equal(gen.get_state(),saved['generator_state'])
    continuity.append(dict(candidate=spec['name'],additional_updates=n,Adam_state_continued=True,RNG_sequence_continued=True))
print('CHECKPOINT_CONTINUITY_PASSED',json.dumps(continuity),flush=True)
recon_base=torch.load(fixture/'inputs/reconstruction.pt',weights_only=False)
for name,early in [('cnet_uniform',False),('cnet_early50',True)]:
    saved=torch.load(fixture/'models'/name/'last.pt',weights_only=False);n=saved['new_update']
    for key,state in saved['optimizer']['state'].items():assert int(state['step'])==int(recon_base['optimizer']['state'][key]['step'])+n
    gen=torch.Generator();gen.set_state(recon_base['generator_state'])
    for _ in range(n):sample_reconstruction(encoded,cfg['reconstruction_batch'],gen,early)
    assert torch.equal(gen.get_state(),saved['generator_state'])
print('CNET_CONTINUITY_PASSED',flush=True)
torch.set_num_threads(2)
a=load_operator(fixture/'inputs/operator.pt');a.train();b=copy.deepcopy(a)
batch=sample_operator(native,phi,64,torch.Generator().manual_seed(123),'uniform',2)
loss_a=training_losses(a,False)[2](*batch);loss_a.backward()
print('COMPILING_ENDPOINT_K2',flush=True)
loss_b=training_losses(b,True)[2](*batch);loss_b.backward()
torch.testing.assert_close(loss_a,loss_b,rtol=1e-4,atol=1e-9)
maximum=0.
for (name,x),(other,y) in zip(a.named_parameters(),b.named_parameters()):
    assert name==other
    torch.testing.assert_close(x.grad,y.grad,rtol=2e-4,atol=1e-7)
    maximum=max(maximum,float(torch.max(torch.abs(x.grad-y.grad))))
charged=started_cost=0.
for name in ['fixture_001','fixture_002']:
    p=follow/name;charged+=json.loads((p/'prepared.json').read_text())['prepare_seconds']+json.loads((p/'complete.json').read_text())['elapsed_seconds']
record=dict(status='passed',elapsed_seconds=time.perf_counter()-started+charged+3.06,
    unit_tests=dict(status='passed',count=7,log_sha256=sha(follow/'unit_tests.log')),
    software_fixture=dict(status='passed',complete_sha256=sha(fixture/'complete.json'),verification=report['verification']),
    exact_checkpoint_continuation=continuity,reconstruction_Adam_RNG_continuation=True,
    compile_equivalence=dict(status='passed',K=2,loss_abs_difference=abs(float(loss_a.detach()-loss_b.detach())),maximum_gradient_difference=maximum),
    chemical_performance_claim=False)
(follow/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
