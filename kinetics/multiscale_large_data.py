"""Checkpointed, process-parallel generation of the full 5.3-million-pair dataset.

Each shard has an independent deterministic RNG and a new ReactorNet. Within a
shard the reactor is reset using Cantera's documented syncState/reinitialize API.
No trajectory perturbation, HRR rejection rule, or approximate chemical solver.
"""
import argparse,concurrent.futures,fcntl,hashlib,json,math,os,time
from pathlib import Path
import cantera as ct
import numpy as np
from .data import mechanism,SPECIES
from .multiscale_data import sample_states


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def tasks_for(cfg):
    tasks=[];group_offset=0
    for hot,k,total in [(False,k,cfg['per_scale']) for k in cfg['scales']]+[(True,4,cfg['hot_initial_states'])]:
        for j,start in enumerate(range(0,total,cfg['shard_initial_states'])):
            count=min(cfg['shard_initial_states'],total-start)
            name=f'{"hot" if hot else "scale"}_{k:02d}_{j:05d}'
            tasks.append(dict(name=name,hot=hot,k=k,ordinal=j,initial_states=count,group_offset=group_offset+start))
        group_offset+=total
    return tasks


def generate_shard(task,cfg,out,config_hash):
    out=Path(out);name=task['name'];start=time.perf_counter()
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],int(task['hot']),task['k'],task['ordinal']]))
    gas=mechanism();reactor=ct.IdealGasConstPressureReactor(gas,clone=True);phase=reactor.phase
    net=ct.ReactorNet([reactor]);net.rtol=1e-10;net.atol=1e-20
    steps=2 if task['hot'] else 1;rows=task['initial_states']*steps
    x=np.empty((rows,11));y=np.empty((rows,10));q=np.empty(rows);factor=np.empty(rows)
    groups=np.empty(rows,np.int64);positions=np.empty(rows,np.int8)
    matrix=np.array([[gas.n_atoms(s,e)*gas.atomic_weight(e)/gas.molecular_weights[j]
        for j,s in enumerate(SPECIES)] for e in ('H','O','N')])
    accepted=0;attempts=0;rejections={};max_elements=0.
    while accepted<task['initial_states']:
        if attempts>20*task['initial_states']+100:raise RuntimeError('Too many rejected candidates')
        draws,factors=sample_states(min(task['initial_states']-accepted,128),task['k'],rng,cfg,task['hot'])
        for state,rescale in zip(draws,factors):
            attempts+=1;reason=None;xx=[];yy=[];qq=[]
            try:
                phase.TPY=state[0],state[-1],state[1:10]
                reactor.syncState();net.initial_time=0.;net.reinitialize()
                for step in range(steps):
                    if phase.Y.min() < -1e-14:raise ValueError('negative_input')
                    xx.append(np.r_[reactor.T,np.maximum(phase.Y,0),state[-1]])
                    qq.append(phase.heat_release_rate)
                    net.advance((step+1)*cfg['dt'])
                    if phase.Y.min() < -1e-14:raise ValueError('negative_target')
                    yy.append(np.r_[reactor.T,np.maximum(phase.Y,0)])
                xx=np.asarray(xx);yy=np.asarray(yy);qq=np.asarray(qq)
                if not np.isfinite(xx).all() or not np.isfinite(yy).all() or not np.isfinite(qq).all():reason='nonfinite'
                elif np.any(yy[:,0]<cfg['temperature_admissibility_K'][0]) or np.any(yy[:,0]>cfg['temperature_admissibility_K'][1]):reason='target_outside_existing_200_6000K_domain'
            except (ValueError,ct.CanteraError) as e:reason=type(e).__name__
            if reason:
                rejections[reason]=rejections.get(reason,0)+1
                # Reset a failed integrator completely; failure history is not reused.
                gas=mechanism();reactor=ct.IdealGasConstPressureReactor(gas,clone=True);phase=reactor.phase
                net=ct.ReactorNet([reactor]);net.rtol=1e-10;net.atol=1e-20
                continue
            begin=accepted*steps;end=begin+steps
            x[begin:end]=xx;y[begin:end]=yy;q[begin:end]=qq;factor[begin:end]=rescale
            groups[begin:end]=task['group_offset']+accepted;positions[begin:end]=np.arange(steps)
            max_elements=max(max_elements,float(np.max(np.abs((yy[:,1:]-xx[:,1:10])@matrix.T))))
            accepted+=1
    assert max_elements<1e-9
    np.testing.assert_allclose(x[:,1:10].sum(axis=1),1,atol=2e-14)
    np.testing.assert_allclose(y[:,1:].sum(axis=1),1,atol=2e-10)
    target=out/'parts'/f'{name}.npz';tmp=target.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,x=x,y=y,group_id=groups,group_step=positions,scale_k=np.full(rows,task['k'],np.int8),
            initial_hrr_W_m3=q,normalization_factor=factor,dt=cfg['dt'],hot=task['hot'])
    os.replace(tmp,target)
    record=dict(**task,rows=rows,attempts=attempts,rejections=rejections,
        negative_initial_hrr_pairs=int((q<0).sum()),max_element_residual=max_elements,
        seconds=time.perf_counter()-start,file=str(target.relative_to(out)),sha256=sha(target),config_sha256=config_hash,source_sha256=sha(__file__))
    atomic_json(out/'parts'/f'{name}.json',record)
    return record


def assemble(out,records,total):
    """Contiguous memory-mapped arrays avoid loading 5.3 million Python rows."""
    out=Path(out);cache=out/'arrays';cache.mkdir(exist_ok=True)
    for key,width in [('x',11),('y',10)]:
        path=cache/f'{key}.npy';tmp=cache/f'{key}.npy.tmp'
        result=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float64,shape=(total,width))
        offset=0
        for record in records:
            with np.load(out/record['file']) as d:a=d[key]
            result[offset:offset+len(a)]=a;offset+=len(a)
        assert offset==total;result.flush();del result;os.replace(tmp,path)
    return {key:dict(path=f'arrays/{key}.npy',sha256=sha(cache/f'{key}.npy'),shape=[total,width]) for key,width in [('x',11),('y',10)]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/zhang_5p3m_data.json'));parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=Path(cfg['output'])
    if out.exists() and not args.resume:raise FileExistsError('Choose a new output or --resume')
    out.mkdir(parents=True,exist_ok=True);(out/'parts').mkdir(exist_ok=True)
    lock=(out/'generation.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A generator is already running; do not duplicate it')
    encoded=json.dumps(cfg,sort_keys=True).encode();config_hash=hashlib.sha256(encoded).hexdigest()
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==cfg
    atomic_json(out/'config.json',cfg)
    source_hash=sha(__file__);tasks=tasks_for(cfg);records={}
    if (out/'manifest.json').exists():
        m=json.loads((out/'manifest.json').read_text())
        if m.get('status')=='completed':
            print('Already completed',m['pairs'],'pairs',flush=True);return
    for task in tasks:
        p=out/'parts'/f'{task["name"]}.json'
        if p.exists():
            r=json.loads(p.read_text());assert r['config_sha256']==config_hash
            assert r['source_sha256']==source_hash,'Cannot mix shards from changed generator code'
            assert r['sha256']==sha(out/r['file']),r['file'];records[task['name']]=r
    remaining=[t for t in tasks if t['name'] not in records]
    started=time.perf_counter();previous=0.
    if (out/'progress.json').exists():previous=float(json.loads((out/'progress.json').read_text()).get('elapsed_seconds',0.))
    def persist(status):
        rows=sum(r['rows'] for r in records.values());elapsed=previous+time.perf_counter()-started
        atomic_json(out/'progress.json',dict(status=status,pairs_completed=rows,target_pairs=cfg['target_pairs'],
            shards_completed=len(records),target_shards=len(tasks),elapsed_seconds=elapsed,
            resumed_from_shards=len(tasks)-len(remaining),workers=cfg['workers']))
    persist('running');print('GENERATION_START',len(remaining),'remaining shards; target',cfg['target_pairs'],flush=True)
    for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[variable]='1'
    import multiprocessing
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            queued=iter(remaining);futures={}
            def submit_next():
                task=next(queued,None)
                if task is not None:futures[pool.submit(generate_shard,task,cfg,str(out),config_hash)]=task
            for _ in range(cfg['workers']*2):submit_next()
            try:
                while futures:
                    done,_=concurrent.futures.wait(futures,return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        r=future.result();futures.pop(future);records[r['name']]=r;persist('running')
                        if len(records)%5==0 or len(records)==len(tasks):
                            print('SHARDS',len(records),'/',len(tasks),'PAIRS',sum(v['rows'] for v in records.values()),'SECONDS',round(previous+time.perf_counter()-started,1),flush=True)
                        submit_next()
                    if previous+time.perf_counter()-started>cfg['max_seconds']:
                        raise TimeoutError('Configured generation budget reached; completed shards are reusable')
            except BaseException:
                for f in futures:f.cancel()
                pool.shutdown(wait=False,cancel_futures=True)
                raise
        ordered=[records[t['name']] for t in tasks]
        assert sum(r['rows'] for r in ordered)==cfg['target_pairs']
        persist('assembling_arrays');arrays=assemble(out,ordered,cfg['target_pairs'])
        assert source_hash==sha(__file__),'Generator changed while running'
        manifest=dict(status='completed',pairs=cfg['target_pairs'],independent_initial_states=sum(t['initial_states'] for t in tasks),
            basic_pairs=25*cfg['per_scale'],hot_pairs=2*cfg['hot_initial_states'],parts=ordered,arrays=arrays,
            config_sha256=config_hash,source_sha256=source_hash,seconds=previous+time.perf_counter()-started,
            physics_inference=False,perturbation=False,manifold_data_used=False,hrr_filter_applied=False,
            negative_initial_hrr_pairs=sum(r['negative_initial_hrr_pairs'] for r in ordered),
            max_element_residual=max(r['max_element_residual'] for r in ordered),
            rejected_initial_states=sum(sum(r['rejections'].values()) for r in ordered),
            normalization=cfg['normalization'],pressure_atm=cfg['pressure_atm'],dt=cfg['dt'],species=SPECIES,
            reference_method='Cantera CVODE, rtol=1e-10, atol=1e-20; documented reactor reset inside each independently seeded shard')
        atomic_json(out/'manifest.json',manifest);persist('completed');print('COMPLETED',manifest['pairs'],'pairs',flush=True)
    except BaseException:persist('interrupted');raise


if __name__=='__main__':main()
