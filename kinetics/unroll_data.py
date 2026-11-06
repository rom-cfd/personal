"""11-step Cantera labels for Liu et al. (2026) unrolling, from the existing 5.3M-pair dataset.

Environment: UNROLL_SRC (preprocessing dir), UNROLL_OUT (output), UNROLL_DATA (raw dataset root;
when set, initial states are the exact float64 x rows instead of decoded float32 features).

The raw float64 dataset is not in this workspace; only the float32 standardized
training features are (GPU handoff). Each of the 4.9M independent initial states
(rows 0..4,499,999 and every even row of the 400k two-step hot block) is decoded
from its float32 feature row -- exactly the state the one-step networks saw -- and
integrated for 11 consecutive dt = 2 us steps with the original label settings:
Cantera CVODE rtol 1e-10 / atol 1e-20, reactor reset via syncState/reinitialize, a
fresh rtol 1e-12 / atol 1e-24 retry when an element residual reaches 1e-9 or a mass
fraction is negative beyond 1e-14. No new sampling, perturbation or projection.
A trajectory that still fails is kept up to its last admissible step (valid_len).

  --prepare            decode initial states -> initial_states.npy
  --chunk-range a b    integrate chunks [a, b) with a process pool (SLURM array task)
  --assemble           standardized float32 training arrays + checks
"""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from .multiscale_large_data import atomic_json,sha

# Paths come from the environment so spawned pool workers see the same values (CLI flags set them).
SRC=Path(os.environ.get('UNROLL_SRC','results/zhang_5p3m_gelu/g0.1/preprocessing'));OUT=Path(os.environ.get('UNROLL_OUT','results/zhang_5p3m_unroll11'))
DATA=os.environ.get('UNROLL_DATA')  # raw float64 dataset root; when set, initial states are read exactly from its x array
STEPS=11;DT=2e-6;CHUNK=5000;GAMMA=0.1


def initial_rows():
    if DATA:
        m=json.loads((Path(DATA)/'manifest.json').read_text());b,h=m['basic_pairs'],m['hot_pairs']
        return np.r_[np.arange(b),np.arange(b,b+h,2)]
    return np.r_[np.arange(4_500_000),np.arange(4_500_000,5_300_000,2)]


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'initial_states.npy').exists():raise FileExistsError(OUT/'initial_states.npy')
    info=json.loads((SRC/'preprocessing.json').read_text());im,isc=np.array(info['input_mean']),np.array(info['input_scale'])
    f=np.load(SRC/'features.npy',mmap_mode='r');rows=initial_rows();out=np.empty((len(rows),11))
    if DATA:
        m=json.loads((Path(DATA)/'manifest.json').read_text());assert sha(Path(DATA)/'manifest.json')==info['data_manifest_sha256']
        x=np.load(Path(DATA)/m['arrays']['x']['path'],mmap_mode='r')
        for i in range(0,len(rows),500_000):out[i:i+500_000]=x[rows[i:i+500_000]]
    for i in (range(0) if DATA else range(0,len(rows),500_000)):
        z=f[rows[i:i+500_000]].astype(np.float64)*isc+im
        y=np.clip(1+GAMMA*z[:,2:],0,None)**(1/GAMMA)
        out[i:i+500_000]=np.c_[z[:,0],y,z[:,1]*101325.]
    np.save(OUT/'initial_states.npy',out);np.save(OUT/'initial_rows.npy',rows)
    atomic_json(OUT/'prepare.json',dict(initial_states=len(rows),features_sha256=info['features_sha256'],data=DATA,
        decode='exact float64 x rows of the raw dataset' if DATA else 'float32 standardized feature -> [T, P_atm, BCT(Y)] -> Y=(1+gamma*b)^(1/gamma); Cantera normalizes Y on TPY',
        steps=STEPS,dt=DT,chunk=CHUNK,source_sha256=sha(__file__)))
    print('PREPARED',len(rows),flush=True)


def _integrate(states,strict=False):
    import cantera as ct
    from .data import mechanism,SPECIES
    gas=mechanism();reactor=ct.IdealGasConstPressureReactor(gas,clone=True);phase=reactor.phase
    net=ct.ReactorNet([reactor]);net.rtol,net.atol=(1e-12,1e-24) if strict else (1e-10,1e-20)
    matrix=np.array([[gas.n_atoms(s,e)*gas.atomic_weight(e)/gas.molecular_weights[j] for j,s in enumerate(SPECIES)] for e in ('H','O','N')])
    return gas,reactor,phase,net,matrix


def run_one(state,objs,strict_objs_factory):
    """Returns (traj (12,10), valid_len, retried)."""
    gas,reactor,phase,net,matrix=objs
    def attempt(reactor,phase,net,reset):
        traj=np.full((STEPS+1,10),np.nan);n=0
        phase.TPY=state[0],state[-1],state[1:10]
        if reset:reactor.syncState();net.initial_time=0.;net.reinitialize()
        traj[0]=np.r_[reactor.T,np.maximum(phase.Y,0)];e0=matrix@traj[0,1:]
        for k in range(STEPS):
            net.advance((k+1)*DT)
            if phase.Y.min()<-1e-14:return traj,n,'negative'
            s=np.r_[reactor.T,np.maximum(phase.Y,0)]
            if not np.isfinite(s).all() or s[0]<200 or s[0]>6000:return traj,n,'domain'
            if np.max(np.abs(matrix@s[1:]-e0))>=1e-9:return traj,n,'elements'
            traj[k+1]=s;n=k+1
        return traj,n,None
    try:traj,n,why=attempt(reactor,phase,net,True)
    except Exception:traj,n,why=np.full((STEPS+1,10),np.nan),0,'error'
    if why is None:return traj,n,False,None
    # Fresh, stricter integrator on the identical state (as in the original generator).
    # As in multiscale_large_data.fresh_conservation_retry: set the state BEFORE building the reactor.
    import cantera as ct
    g2=strict_objs_factory();g2.TPY=state[0],state[-1],state[1:10]
    r2=ct.IdealGasConstPressureReactor(g2,clone=True);n2=ct.ReactorNet([r2]);n2.rtol=1e-12;n2.atol=1e-24
    try:t2,m2,why2=attempt(r2,r2.phase,n2,False)
    except Exception:t2,m2,why2=traj,n,'error'
    if m2>=n:traj,n=t2,m2
    return traj,n,True,why2


def do_chunk(c):
    path=OUT/'parts'/f'chunk_{c:05d}.npz'
    if path.exists():return c,'exists',0.
    from .data import mechanism
    began=time.perf_counter();init=np.load(OUT/'initial_states.npy',mmap_mode='r');a=c*CHUNK;b=min(a+CHUNK,len(init))
    objs=_integrate(None);st=np.empty((b-a,STEPS+1,10));vl=np.empty(b-a,np.int8);rt=np.zeros(b-a,bool);reasons={}
    for i in range(a,b):
        traj,n,retried,why=run_one(np.asarray(init[i]),objs,mechanism)
        st[i-a]=traj;vl[i-a]=n;rt[i-a]=retried
        if retried:
            reasons[str(why)]=reasons.get(str(why),0)+1;objs=_integrate(None)  # discard history after any failure
    tmp=path.with_suffix('.tmp.npz');np.savez(tmp,states=st,valid_len=vl,retried=rt,start=a);os.replace(tmp,path)
    atomic_json(path.with_suffix('.json'),dict(chunk=c,start=a,stop=b,retried=int(rt.sum()),incomplete=int((vl<STEPS).sum()),
        reasons=reasons,seconds=time.perf_counter()-began,sha256=sha(path)))
    return c,'done',time.perf_counter()-began


def chunk_range(a,b,workers):
    (OUT/'parts').mkdir(exist_ok=True)
    for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[v]='1'
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(workers) as pool:
        for c,status,sec in pool.imap_unordered(do_chunk,range(a,b)):print('CHUNK',c,status,round(sec,1),flush=True)


def repair_chunk(c):
    """Redo incomplete trajectories of one chunk with the corrected fresh retry (one fresh integrator each)."""
    from .data import mechanism
    path=OUT/'parts'/f'chunk_{c:05d}.npz';meta=json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as d:st,vl,rt,a=d['states'].copy(),d['valid_len'].copy(),d['retried'].copy(),int(d['start'])
    bad=np.flatnonzero(vl<STEPS)
    if not len(bad):return c,0,0
    init=np.load(OUT/'initial_states.npy',mmap_mode='r');fixed=0;reasons={}
    for j in bad:
        traj,n,retried,why=run_one(np.asarray(init[a+j]),_integrate(None),mechanism)
        if n>vl[j]:st[j],vl[j]=traj,n
        rt[j]=True;fixed+=int(vl[j]==STEPS)
        if vl[j]<STEPS:reasons[str(why)]=reasons.get(str(why),0)+1
    tmp=path.with_suffix('.tmp.npz');np.savez(tmp,states=st,valid_len=vl,retried=rt,start=a);os.replace(tmp,path)
    atomic_json(path.with_suffix('.json'),{**meta,'incomplete':int((vl<STEPS).sum()),'reasons':reasons,
        'repaired':dict(attempted=int(len(bad)),now_complete=fixed,method='fresh integrator per state; retry builds reactor after setting state'),'sha256':sha(path)})
    return c,len(bad),fixed


def assemble():
    """Standardized float32 arrays for unrolled training (same normalizers as the parents)."""
    info=json.loads((SRC/'preprocessing.json').read_text())
    im,isc,om,osc=[np.array(info[k]) for k in ('input_mean','input_scale','output_mean','output_scale')]
    init=np.load(OUT/'initial_states.npy',mmap_mode='r');n=len(init);chunks=(n+CHUNK-1)//CHUNK
    A=OUT/'arrays';A.mkdir(exist_ok=True)
    f0=np.lib.format.open_memmap(A/'features0.npy','w+',np.float32,(n,11))
    lab=np.lib.format.open_memmap(A/'labels.npy','w+',np.float32,(n,STEPS,11))
    vl=np.empty(n,np.int8);rt=np.empty(n,bool);reasons={};orig_t=np.load(SRC/'targets.npy',mmap_mode='r');rows=np.load(OUT/'initial_rows.npy')
    worst=np.zeros(11)
    for c in range(chunks):
        with np.load(OUT/'parts'/f'chunk_{c:05d}.npz') as d:st,v,r,a=d['states'],d['valid_len'],d['retried'],int(d['start'])
        b=a+len(st);P=init[a:b,10]/101325.
        enc=np.concatenate([st[...,:1],np.broadcast_to(P[:,None,None],(len(st),STEPS+1,1)),(np.maximum(st[...,1:],0)**GAMMA-1)/GAMMA],-1)
        f0[a:b]=((enc[:,0]-im)/isc).astype(np.float32)
        delta=enc[:,1:]-enc[:,:-1];l=(delta-om)/osc;l[~np.isfinite(l)]=0.
        lab[a:b]=l.astype(np.float32);vl[a:b]=v;rt[a:b]=r
        # step-1 label vs the original one-step target of the same row (float32 decode round-off expected)
        ok=v>=1;worst=np.maximum(worst,np.abs(l[ok,0]-orig_t[rows[a:b][ok]]).max(0))
        m=json.loads((OUT/'parts'/f'chunk_{c:05d}.json').read_text())
        for k,q in m['reasons'].items():reasons[k]=reasons.get(k,0)+q
    f0.flush();lab.flush();np.save(A/'valid_len.npy',vl)
    atomic_json(OUT/'manifest.json',dict(status='completed',initial_states=n,steps=STEPS,dt=DT,
        complete_trajectories=int((vl==STEPS).sum()),incomplete=int((vl<STEPS).sum()),valid_len_histogram=np.bincount(vl,minlength=STEPS+1).tolist(),
        fresh_retries=int(rt.sum()),retry_final_reasons=reasons,
        step1_label_vs_original_target_max_abs_standardized=worst.tolist(),
        arrays=dict(features0=dict(path='arrays/features0.npy',sha256=sha(A/'features0.npy')),labels=dict(path='arrays/labels.npy',sha256=sha(A/'labels.npy')),
            valid_len=dict(path='arrays/valid_len.npy',sha256=sha(A/'valid_len.npy'))),
        normalizers='parent preprocessing.json (train-fitted, unchanged)',source_sha256=sha(__file__)))
    print('ASSEMBLED',n,'complete',int((vl==STEPS).sum()),'step1 max diff',worst.round(5).tolist(),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare',action='store_true');ap.add_argument('--assemble',action='store_true')
    ap.add_argument('--all-chunks',action='store_true',help='integrate every chunk in one process pool')
    ap.add_argument('--chunk-range',type=int,nargs=2);ap.add_argument('--repair',action='store_true');ap.add_argument('--workers',type=int,default=int(os.environ.get('SLURM_CPUS_PER_TASK',8)))
    a=ap.parse_args()
    if a.prepare:prepare()
    if a.all_chunks:chunk_range(0,(len(np.load(OUT/'initial_states.npy',mmap_mode='r'))+CHUNK-1)//CHUNK,a.workers)
    if a.chunk_range:chunk_range(*a.chunk_range,a.workers)
    if a.repair:
        import multiprocessing as mp
        total=(len(np.load(OUT/'initial_states.npy',mmap_mode='r'))+CHUNK-1)//CHUNK
        chunks=[c for c in range(total) if json.loads((OUT/'parts'/f'chunk_{c:05d}.json').read_text())['incomplete']>0]
        with mp.get_context('spawn').Pool(a.workers) as pool:
            for c,n,f in pool.imap_unordered(repair_chunk,chunks):print('REPAIR',c,n,f,flush=True)
    if a.assemble:assemble()


if __name__=='__main__':main()
