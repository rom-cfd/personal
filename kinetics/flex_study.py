"""Bounded, frozen protocol reproduction of the 0D flexDeepONet example."""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import shutil
import time
import traceback
import numpy as np
import torch
from kinetics.flex_model import FlexDeepONet, KerasAdam
from kinetics.flex_data import atomic, sha, stamp, prepare, FOLLOW

SOURCES=['kinetics/flex_model.py','kinetics/flex_data.py','kinetics/flex_study.py',
         'kinetics/flex_report.py','scripts/verify_flex_fidelity.py',
         'scripts/monitor_flex_reproduction.py','scripts/supervise_flex_reproduction.py',
         'tests/test_flex_reproduction.py']


def save_checkpoint(path,model,optimizer,rng,epoch,update,loss,active):
    path=Path(path);tmp=path.with_suffix('.pt.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                    epoch=epoch,update=update,validation_loss=loss,active_seconds=active),tmp)
    os.replace(tmp,path)


def freeze(root,budget_seconds):
    root=Path(root)
    assert not (root/'prepared.json').exists()
    assert json.loads((root/'reference_prepared.json').read_text())['status']=='passed'
    fidelity=json.loads((FOLLOW/'fidelity.json').read_text())
    assert fidelity['status']=='passed'
    (root/'preflight').mkdir(exist_ok=True)
    shutil.copyfile(FOLLOW/'fidelity.json',root/'preflight/fidelity.json')
    tests=FOLLOW/'unit_tests.log'
    if tests.exists():
        assert '5 passed' in tests.read_text()
        shutil.copyfile(tests,root/'preflight/unit_tests.log')
    train=np.load(root/'data/train.npz')
    truth=train['values'];identity=train['initial'][:,None,:]
    rms=np.sqrt(np.mean(truth**2,axis=1))
    nrmse=np.sqrt(np.mean((truth-identity)**2,axis=1))/rms
    atomic(root/'preflight/identity_baseline.json',dict(training_ICs=len(truth),
        joint9_curve_passes=int(np.all(nrmse[:,:9]<=.1,axis=1).sum()),
        median_NRMSE_by_variable=np.median(nrmse,axis=0).tolist(),
        minimum_temperature_rise_K=float(np.min(truth[:,:,0].max(1)-train['initial'][:,0])),
        minimum_H2_consumption=float(np.min(train['initial'][:,1]-truth[:,:,1].min(1))),
        note='Training-only no-change control. No external model-test results used.'))
    fixture=Path('/workspace/personal/results/flex_deeponet_reproduction/fixture_001')
    if fixture!=root and (fixture/'complete.json').exists():
        for name in ['complete.json','verification.json','report/complete.json']:
            target=root/'preflight'/('fixture_'+name.replace('/','_'))
            shutil.copyfile(fixture/name,target)
    config=json.loads((root/'config.json').read_text());config['budget_seconds']=budget_seconds
    atomic(root/'config.json',config)
    snapshot=root/'source_snapshot';snapshot.mkdir()
    repo=Path(__file__).resolve().parents[1]
    manifest={}
    for name in SOURCES+['kinetics/__init__.py']:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(repo/name,target);manifest[name]=sha(target)
    (snapshot/'scripts/__init__.py').write_text('')
    manifest['scripts/__init__.py']=sha(snapshot/'scripts/__init__.py')
    atomic(root/'source_manifest.json',manifest)
    inputs={str(p.relative_to(root)):sha(p) for p in sorted((root/'data').glob('*.npz'))}
    for name in ['config.json','cases.json','normalization.json','gri30.yaml','protocol_audit.json',
                 'source_manifest.json','reference_verification.json','preflight/fidelity.json']:
        inputs[name]=sha(root/name)
    atomic(root/'input_manifest.json',inputs)
    atomic(root/'prepared.json',dict(status='passed',at=stamp(),budget_seconds=budget_seconds,
        preparation_seconds=json.loads((root/'reference_prepared.json').read_text())['seconds'],
        preflight_seconds=fidelity['wall_seconds'],fixture=config['fixture'],
        source_manifest_sha256=sha(root/'source_manifest.json'),input_manifest_sha256=sha(root/'input_manifest.json')))
    atomic(root/'status.json',dict(phase='awaiting_publication',at=stamp()))


def verify_inputs(root):
    inputs=json.loads((root/'input_manifest.json').read_text())
    assert all(sha(root/name)==value for name,value in inputs.items())
    sources=json.loads((root/'source_manifest.json').read_text())
    assert all(sha(root/'source_snapshot'/name)==value for name,value in sources.items())


def run(root,fixture=False):
    root=Path(root).resolve()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    config=json.loads((root/'config.json').read_text())
    assert config['fixture']==fixture
    prepared=json.loads((root/'prepared.json').read_text())
    if fixture:
        budget=90.;deadline=time.monotonic()+budget
    else:
        publication=json.loads((FOLLOW/'publication.json').read_text())
        assert publication['status']=='passed' and publication['remote_verified'] is True
        assert publication['prepared_sha256']==sha(root/'prepared.json')
        launch=json.loads((FOLLOW/'launch.json').read_text())
        deadline=launch['deadline_monotonic']
    verify_inputs(root)
    started=time.monotonic()
    atomic(root/'worker_started.json',dict(pid=os.getpid(),at=stamp(),deadline_monotonic=deadline,
        command='frozen kinetics.flex_study run',source_manifest_sha256=sha(root/'source_manifest.json')))
    train=np.load(root/'data/train.npz');split=np.load(root/'data/point_split.npz')
    norm=json.loads((root/'normalization.json').read_text())
    x=torch.from_numpy(train['initial'].copy())
    t=torch.from_numpy(train['query_times'].copy())
    y=torch.from_numpy(train['values'].reshape(-1,19).copy())/torch.tensor(norm['output_range'])
    nt=len(t);train_indices=split['train'];valid_indices=split['validation']
    torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    model=FlexDeepONet(norm['input_range'],norm['output_range'])
    optimizer=KerasAdam(model.parameters())
    (root/'models').mkdir(exist_ok=True)
    best=float('inf');early_best=float('inf');best_epoch=0;wait=0;updates=0;active=0.;last_lr=.001
    reason='requested_epochs_completed';reserve=0. if fixture else config['report_reserve_seconds']
    def validation():
        total=0.;count=0
        with torch.no_grad():
            for start in range(0,len(valid_indices),2048):
                ids=torch.from_numpy(valid_indices[start:start+2048].copy())
                error=(model(x[ids//nt],t[ids%nt])-y[ids]).square()
                total+=error.sum().item();count+=error.numel()
            return total/count+model.regularization().item()
    initial_val=validation()
    save_checkpoint(root/'models/initial.pt',model,optimizer,rng,0,0,initial_val,0.)
    histories=[]
    file=root/'training.csv'
    fields=['epoch','update','active_seconds','wall_seconds','train_MSE','validation_total_loss','lr','best_epoch']
    with file.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for epoch in range(1,config['max_epochs']+1):
            ids_epoch=rng.permutation(train_indices);train_loss=0.;batches=0
            for start in range(0,len(ids_epoch),256):
                if time.monotonic()>=deadline-reserve:
                    reason='overall_time_cap';break
                begin=time.perf_counter()
                ids=torch.from_numpy(ids_epoch[start:start+256].copy())
                last_lr=.001*.99**(updates/10000)
                for group in optimizer.param_groups:group['lr']=last_lr
                optimizer.zero_grad(set_to_none=True)
                prediction=model(x[ids//nt],t[ids%nt])
                mse=(prediction-y[ids]).square().mean();loss=mse+model.regularization()
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Non-finite training gradient')
                optimizer.step();updates+=1
                active+=time.perf_counter()-begin
                train_loss+=mse.item();batches+=1
                if updates%250==0:
                    atomic(root/'status.json',dict(phase='training',epoch=epoch,update=updates,
                        active_seconds=active,remaining_seconds=max(0,deadline-time.monotonic()),lr=last_lr,at=stamp()))
            val=validation()
            if val<best:
                best=val;best_epoch=epoch
                save_checkpoint(root/'models/selected.pt',model,optimizer,rng,epoch,updates,val,active)
            if val<early_best-1e-8:
                early_best=val;wait=0
            else:wait+=1
            row=dict(epoch=epoch,update=updates,active_seconds=active,wall_seconds=time.monotonic()-started,
                train_MSE=train_loss/max(1,batches),validation_total_loss=val,lr=last_lr,best_epoch=best_epoch)
            writer.writerow(row);f.flush();histories.append(row)
            save_checkpoint(root/'models/last.pt',model,optimizer,rng,epoch,updates,val,active)
            if reason=='overall_time_cap':break
            if wait>=3000:
                reason='author_validation_early_stopping';break
    selected=torch.load(root/'models/selected.pt',weights_only=False,map_location='cpu')
    selection=dict(status='frozen_before_external_test',at=stamp(),criterion='minimum pointwise-validation total loss',
        selected_epoch=selected['epoch'],selected_update=selected['update'],selected_loss=selected['validation_loss'],
        selected_sha256=sha(root/'models/selected.pt'),final_epoch=epoch,final_update=updates,
        final_sha256=sha(root/'models/last.pt'),stop_reason=reason,active_seconds=active,final_lr=last_lr,
        validation_partition='random time points of the500training trajectories; distinct from independent testICs')
    atomic(root/'selection_frozen.json',selection)
    atomic(root/'final_training.json',selection)
    atomic(root/'test_access_started.json',dict(at=stamp(),selection_sha256=sha(root/'selection_frozen.json')))
    atomic(root/'status.json',dict(phase='evaluation_and_reporting',at=stamp()))
    from kinetics.flex_report import build_report
    build_report(root,deadline)
    verify_inputs(root)
    artifacts={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()
        and '__pycache__' not in p.parts and p.suffix not in {'.lock','.tmp'}
        and p.name not in ['complete.json','exit.json','failure.json','verification.json','status.json']}
    atomic(root/'verification.json',dict(status='passed',at=stamp(),inputs_and_sources_unchanged=True,
        selection_precedes_test_access=True,output_sha256=artifacts))
    atomic(root/'complete.json',dict(status='passed',at=stamp(),stop_reason=reason,
        wall_seconds=time.monotonic()-started,verification_sha256=sha(root/'verification.json'),
        report_complete_sha256=sha(root/'report/complete.json')))
    atomic(root/'status.json',dict(status='passed',phase='complete',at=stamp()))
    atomic(root/'exit.json',dict(returncode=0,at=stamp()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','freeze','run'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--fixture',action='store_true')
    parser.add_argument('--budget-seconds',type=float,default=21600)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.root,args.fixture)
    elif args.action=='freeze':freeze(args.root,args.budget_seconds)
    else:
        try:run(args.root,args.fixture)
        except Exception:
            atomic(args.root/'failure.json',dict(at=stamp(),traceback=traceback.format_exc()))
            atomic(args.root/'exit.json',dict(returncode=1,at=stamp()))
            raise


if __name__=='__main__':main()
