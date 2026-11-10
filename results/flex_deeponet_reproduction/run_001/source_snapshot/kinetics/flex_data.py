"""Frozen Cantera references and provenance for the published flexDeepONet case."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import time
import numpy as np

SPECIES = ['H2','H','O','O2','OH','H2O','HO2','H2O2','N','NH','NH2','NH3',
           'NNH','NO','NO2','N2O','HNO','N2']
VARIABLES = ['T'] + SPECIES
REPO = Path('/workspace/personal')
FOLLOW = Path('/workspace/flex_reproduction_followup/run_001')
AUTHOR = Path('/workspace/literature_sources/romnet_author')
AUTHOR_SHA = '07278fc3ffe9eda7be7eed2cf96068f4be3efa51'
SOURCE_FILES = [
    'input/0DReactor/DeepONet/0DReactor_H2_TestCase3/ROMNet_Input.py',
    'scripts/generating_data/0DReactor/Generate_Data_1_Isobaric.py',
    'scripts/generating_data/0DReactor/Generate_Data_3.ipynb',
    'romnet/architecture/deeponet.py',
    'romnet/architecture/building_blocks/normalization.py',
    'romnet/architecture/building_blocks/system_of_components/deeponet_system.py',
    'romnet/architecture/building_blocks/component.py',
    'romnet/architecture/building_blocks/layer.py',
    'romnet/data/data.py', 'romnet/__init__.py',
]
PAPER_RMSE = [1.3629,7.4989e-5,2.1747e-5,2.7380e-5,3.6907e-4,2.8910e-5,
              3.3034e-4,3.0370e-7,2.0288e-8,2.5091e-9,7.1590e-10,3.0020e-10,
              1.4949e-9,4.6500e-10,8.5739e-6,3.3697e-9,1.5176e-9,5.7490e-10,2.4410e-5]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def lhs_center(n, seed):
    rng=np.random.default_rng(seed)
    z=np.column_stack([(rng.permutation(n)+.5)/n for _ in range(2)])
    return np.column_stack([.5+3.5*z[:,0],1000+1000*z[:,1]])


def trajectory(ic, times, mechanism, strict=False):
    import cantera as ct
    gas=ct.Solution(str(mechanism))
    gas.TP=float(ic[1]),ct.one_atm
    gas.set_equivalence_ratio(float(ic[0]),'H2:1','O2:1,N2:4')
    initial=np.r_[gas.T,gas.Y[[gas.species_index(s) for s in SPECIES]]].copy()
    reactor=ct.IdealGasConstPressureReactor(gas,clone=True)
    solver=ct.ReactorNet([reactor]);solver.max_steps=100000
    if strict:solver.rtol=1e-11;solver.atol=1e-18
    indices=[reactor.phase.species_index(s) for s in SPECIES]
    values=np.empty((len(times),19))
    for i,t in enumerate(times):
        if t>0:solver.advance(float(t))
        values[i,0]=reactor.T
        values[i,1:]=reactor.phase.Y[indices].copy()
    assert np.isfinite(values).all()
    assert np.max(np.abs(values[:,1:].sum(1)-1))<1e-7
    assert np.max(values[:,0])-initial[0]>100, 'Reference must contain actual ignition'
    assert np.ptp(values[:,1])>1e-3, 'Reference species cannot be an identity map'
    return initial,values


def audit_sources(root):
    import subprocess
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=AUTHOR,text=True).strip()
    assert actual==AUTHOR_SHA
    sources={p:sha(AUTHOR/p) for p in SOURCE_FILES}
    audit=dict(
        title='Venturi--Casey 0D isobaric H2/air flexDeepONet protocol reproduction',
        paper='https://arxiv.org/abs/2204.12670v3',
        doi='https://doi.org/10.1016/j.cma.2022.115718',
        repository='https://github.com/simoneventuri/ROMNet_',commit=AUTHOR_SHA,
        author_source_sha256=sources,parameters=34038,
        branch=dict(count=19,layers=[19,16,16,16,9],hidden_activation='tanh'),
        trunk=dict(count=19,layers=[1,16,16,16,8],hidden_activation='tanh'),
        stretch=dict(count=1,layers=[19,16,16,16,19],hidden_activation='tanh',output='softplus'),
        query_transform='ln(t * stretch_j(x0)); multiply BEFORE log as in released component.py',
        output='dot(branch_j[:8], trunk_j) + branch_j[8]; multiply by output_range',
        preprocessing='Released range scaling: divide by range without subtracting minimum. Input CustomNormalization extrema initialize at ones; preserved explicitly. Output extrema use the 500-scenario training pool.',
        training=dict(dtype='float64',optimizer='Keras Adam',lr=.001,beta1=.9,beta2=.999,epsilon=1e-7,
                      lr_decay='.99 ** (global_update/10000)',batch=256,epochs=100000,
                      regularization='L1=L2=1e-12 on all kernels and hidden biases; no final bias regularization',
                      early_stopping=dict(metric='validation total loss',min_delta=1e-8,patience_epochs=3000)),
        paper_table_S3_RMSE=dict(zip(VARIABLES,PAPER_RMSE)),
        comparison_limits=[
            'No released trained Case3 checkpoint or exact original LHS random seed was located.',
            'Independent batched PyTorch implementation, checked against original source operations and TensorFlow gradients/Adam; not an original-weight replay.',
            'Cantera 3.2 uses the distributed GRI30 YAML; original library version may differ.',
            'Primary external test is 10 independent LHS conditions as in the paper; another 32 conditions are a separately labeled robustness check.',
            'Pointwise train/validation split follows released notebook. All external test ICs remain isolated until selection is frozen.',
            'Direct full-trajectory operator is not a chemistry stepper. It cannot establish success for 20us feedback or CFD coupling.',
            'Time cap may stop before published training convergence; do not equate time-cap with plateau or impossibility.',
        ],
        prior_observations=[
            'Previous React-DeepONet K5 gradient L2 was about164 times K1 on the controlled batch; lowering LR mitigated the first-update collapse.',
            'Previous strict raw eligibility also rejected tiny negative interior0.1us queries, despite nonnegative saved representative20us endpoints.',
            'Private fusion and exact Burke20-reaction file remained unresolved in the previous reconstruction.',
        ],
    )
    atomic(root/'protocol_audit.json',audit)
    # Audit another author's released subset rather than mistaking an identity map for ignition.
    import scipy.io as sio
    folder=Path('/workspace/literature_sources/stiff_deeponet/AE-DeepONet/Data')
    subset=[]
    for name,label in [('training_data_ds.mat','train'),('testing_data_ds.mat','test')]:
        f=folder/name;d=sio.loadmat(f);x=d['f_'+label][:,:12]
        for tau in [250,500,750,1000]:
            y=d[f'u{tau}_{label}']
            subset.append(dict(file=name,sha256=sha(f),rows=len(x),tau_index=tau,
                all_11_species_bitwise_unchanged=bool(np.array_equal(x[:,:11],y[:,:11])),
                identity_relative_L2_in_stored_log_space=float(np.linalg.norm(y-x)/np.linalg.norm(y))))
    atomic(root/'alternative_released_subset_audit.json',dict(
        source='https://github.com/somdattagoswami/Approximating-StiffKinetics',
        observations=subset,decision='Do not use this limited released subset as proof of evolving ignition species. This does not establish that the full paper dataset has the same issue.'))


def prepare(root, fixture=False):
    import cantera as ct
    root=Path(root)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    start=time.monotonic()
    atomic(root/'preparation_started.json',dict(at=stamp(),monotonic=start,fixture=fixture))
    audit_sources(root)
    config=dict(seed=11,threads=2,train_ICs=4 if fixture else 500,
                primary_test_ICs=2 if fixture else 10,robustness_ICs=2 if fixture else 32,
                training_times=32 if fixture else 500,fixture=fixture,
                default_budget_seconds=21600,report_reserve_seconds=600,
                max_epochs=2 if fixture else 100000,
                validation_fraction=.2,validation_seed=42)
    atomic(root/'config.json',config)
    gas=ct.Solution('gri30.yaml');gas.write_yaml(str(root/'gri30.yaml'))
    atomic(root/'mechanism_manifest.json',dict(cantera_version=ct.__version__,mechanism='gri30.yaml',
        sha256=sha(root/'gri30.yaml'),full_species=gas.n_species,full_reactions=gas.n_reactions,
        outputs=VARIABLES,pressure_Pa=ct.one_atm,fuel='H2:1',oxidizer='O2:1,N2:4'))
    times=np.r_[0.,np.logspace(-7,0,config['training_times']-1)]
    query=np.maximum(times,1e-32)
    cases={label:lhs_center(n,seed) for label,n,seed in [
        ('train',config['train_ICs'],1101),('paper_like_test',config['primary_test_ICs'],1102),
        ('robustness_test',config['robustness_ICs'],1103)]}
    if fixture:
        cases['paper_like_test']=np.array([[.8,1100.],[1.5,1300.]])
        cases['robustness_test']=np.array([[.9,1200.],[2.,1600.]])
    keys=[tuple(x) for part in cases.values() for x in part]
    assert len(keys)==len(set(keys))
    atomic(root/'cases.json',{k:v.tolist() for k,v in cases.items()})
    (root/'data').mkdir()
    for label,ics in cases.items():
        initial=[];values=[]
        for index,ic in enumerate(ics):
            x,y=trajectory(ic,times,root/'gri30.yaml')
            initial.append(x);values.append(y)
            if index%10==0 or index==len(ics)-1:
                atomic(root/'status.json',dict(phase='reference_preparation',split=label,
                    completed=index+1,total=len(ics),at=stamp()))
        np.savez_compressed(root/'data'/f'{label}.npz',ics=ics,initial=np.array(initial),
                            values=np.array(values),times=times,query_times=query)
    train=np.load(root/'data/train.npz')
    # Exact ShuffleSplit index rule used by train_test_split(random_state=42).
    count=config['train_ICs']*len(times)
    permutation=np.random.RandomState(42).permutation(count)
    n_validation=int(np.ceil(.2*count))
    b=permutation[:n_validation];a=permutation[n_validation:]
    np.savez(root/'data/point_split.npz',train=a,validation=b)
    from kinetics.flex_model import author_input_range
    input_range=author_input_range(train['initial'])
    output_range=np.ptp(train['values'].reshape(-1,19),axis=0)
    assert np.all(output_range>0)
    atomic(root/'normalization.json',dict(input_range=input_range.tolist(),output_range=output_range.tolist(),
        fitted_on='500 training trajectories including their pointwise validation partition; external tests excluded',
        no_box_cox=True,output_centering=False))
    # Independent tighter Cantera integration on two predetermined TRAIN ICs.
    checks=[]
    for index in range(2):
        _,tight=trajectory(cases['train'][index],times,root/'gri30.yaml',strict=True)
        err=np.sqrt(np.mean((tight-train['values'][index])**2,axis=0))/output_range
        checks.append(dict(train_index=index,max_RMSE_over_training_range=float(err.max())))
    assert max(v['max_RMSE_over_training_range'] for v in checks)<1e-4
    atomic(root/'reference_verification.json',dict(status='passed',tighter_solver_checks=checks,
        genuine_H2_consumption=True,genuine_temperature_rise=True,external_ICs_disjoint=True))
    atomic(root/'data_manifest.json',{str(p.relative_to(root)):sha(p) for p in sorted((root/'data').glob('*.npz'))})
    atomic(root/'reference_prepared.json',dict(status='passed',seconds=time.monotonic()-start,at=stamp()))
    atomic(root/'status.json',dict(phase='reference_prepared',at=stamp()))
    return root
