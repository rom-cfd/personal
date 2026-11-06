"""UR=10 training windows for consolidating the 0-D gym data, with replay of the original training windows.

Gym windows: every saved gym pair input x (results/t1000_gym/rounds, train+val episodes only) is an exact Cantera
state; it is integrated for 11 consecutive dt = 2 us steps (reference mechanism, rtol 1e-10, atol 1e-20, as in
kinetics.multiscale_data.integrate_labels) -> 12 states -> standardized features0 and 11 increment labels, with the
parents' normalizers (same convention as kinetics.unroll_data.assemble). A window that Cantera rejects is dropped.
Replay: --ratio R times as many windows drawn at random (seed 2026) from results/t1000_unroll11 (the original UR10
training windows). Writes results/t1000_gymur_r{R}/ in the layout kinetics.unroll_training reads.
Round 2 (DAgger-style aggregation): --gyms takes several ROUNDS_DIR:CACHE pairs; all their gym windows are pooled and
R times as many original windows are drawn (--out-prefix names the output, --parent the normalizers, identical files).
"""
import argparse,glob,json,os,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .multiscale_large_data import atomic_json,sha

PRE=Path('results/t1000_ur10_phys_best/g0.1/preprocessing/preprocessing.json');REPLAY=Path('results/t1000_unroll11')
GYM=Path('results/t1000_gym/rounds');CACHE=Path('results/t1000_gymur_windows.npz');STEPS=11;DT=2e-6;GAMMA=.1


def integrate(xs):
    from .data import mechanism
    from .multiscale_data import integrate_labels
    gas=mechanism();out=np.full((len(xs),STEPS+1,10),np.nan);ok=np.zeros(len(xs),bool)
    for i,x in enumerate(xs):
        try:a,b,_=integrate_labels(x,STEPS,DT,gas);out[i,:STEPS]=a[:,:10];out[i,STEPS]=b[-1];ok[i]=True
        except Exception:pass
    return out,ok


def gym_windows(workers,GYM=GYM,CACHE=CACHE):
    if CACHE.exists():
        with np.load(CACHE) as d:return d['x'],d['states'],d['ok']
    files=sorted(glob.glob(str(GYM/'*.npz')));x=np.concatenate([np.load(f)['x'] for f in files])
    chunks=np.array_split(np.arange(len(x)),workers*8);t0=time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:res=list(ex.map(integrate,[x[c] for c in chunks]))
    states=np.concatenate([r[0] for r in res]);ok=np.concatenate([r[1] for r in res])
    # step 0 of each window must reproduce the saved gym pair's label exactly (same Cantera settings)
    y=np.concatenate([np.load(f)['y'] for f in files]);err=(float(np.nanmax(np.abs(states[ok,1,0]-y[ok,0]))),float(np.nanmax(np.abs(states[ok,1,1:]-y[ok,1:]))))
    np.savez(CACHE,x=x,states=states,ok=ok,files=np.array([Path(f).name for f in files]))
    print('GYM_WINDOWS',len(x),'ok',int(ok.sum()),'step1 vs saved label max abs (T K, Y)',err,f'{time.perf_counter()-t0:.0f}s',flush=True)
    assert err[0]<1e-3 and err[1]<1e-6,err   # restarted CVODE vs the original continuous run: round-off only
    return x,states,ok


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--ratios',type=int,nargs='+',default=[1,3])
    ap.add_argument('--workers',type=int,default=os.cpu_count() or 8)
    ap.add_argument('--gyms',nargs='+',default=[f'{GYM}:{CACHE}'],help='ROUNDS_DIR:CACHE.npz pairs, pooled')
    ap.add_argument('--out-prefix',default='results/t1000_gymur_r');ap.add_argument('--parent',default=str(PRE.parent.parent))
    a=ap.parse_args();pre=Path(a.parent)/'preprocessing/preprocessing.json'
    assert pre.read_text()==PRE.read_text(),'normalizers differ from strategy 5'
    info=json.loads(pre.read_text());im,isc,om,osc=[np.array(info[k]) for k in ('input_mean','input_scale','output_mean','output_scale')]
    W=[gym_windows(a.workers,Path(g.split(':')[0]),Path(g.split(':')[1])) for g in a.gyms]
    x=np.concatenate([w[0][w[2]] for w in W]);st=np.concatenate([w[1][w[2]] for w in W]);n=len(x)
    print('POOLED gym windows',[int(w[2].sum()) for w in W],'total',n,flush=True);P=x[:,-1]/101325.
    enc=np.concatenate([st[...,:1],np.broadcast_to(P[:,None,None],(n,STEPS+1,1)),(np.maximum(st[...,1:],0)**GAMMA-1)/GAMMA],-1)
    gf0=((enc[:,0]-im)/isc).astype(np.float32);gl=((enc[:,1:]-enc[:,:-1]-om)/osc).astype(np.float32);gv=np.full(n,STEPS,np.int8)
    rf0=np.load(REPLAY/'arrays/features0.npy',mmap_mode='r');rl=np.load(REPLAY/'arrays/labels.npy',mmap_mode='r');rv=np.load(REPLAY/'arrays/valid_len.npy')
    for r in a.ratios:
        out=Path(f'{a.out_prefix}{r}');A=out/'arrays'
        if (out/'manifest.json').exists():print('EXISTS',out);continue
        A.mkdir(parents=True,exist_ok=True)
        idx=np.sort(np.random.default_rng(2026).choice(len(rf0),r*n,replace=False))
        np.save(A/'features0.npy',np.concatenate([gf0,np.asarray(rf0[idx])]));np.save(A/'labels.npy',np.concatenate([gl,np.asarray(rl[idx])]))
        np.save(A/'valid_len.npy',np.concatenate([gv,rv[idx]]))
        atomic_json(out/'manifest.json',dict(status='completed',initial_states=int(n*(1+r)),steps=STEPS,dt=DT,gym_windows=int(n),replay_windows=int(r*n),
            replay_ratio=r,replay_source=str(REPLAY),replay_seed=2026,gym_rounds=a.gyms,normalizers=str(pre),
            note='gym windows first, then replay rows; unroll_training shuffles every epoch',source_sha256=sha(__file__)))
        print('WROTE',out,'gym',n,'replay',r*n,flush=True)


if __name__=='__main__':main()
