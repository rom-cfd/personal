"""0-D ignition gym: uncertainty-triggered online fine-tuning carried across episodes, then an offline final model.

Episodes are the TRAIN + VAL initial conditions of results/t1000_long10ms_data (test cases are never visited, so the
offline test evaluation stays held out). Two passes, each in a fixed shuffled order. In every episode the five
strategy-5 members roll out in lockstep exactly as in kinetics.t1000_online_finetune (same trigger, perturbation,
Cantera labels, L2-SP fine-tune recipe, Adam 1e-5), except that the weights are NOT reset between episodes; the
L2-SP anchor stays the starting (--root) weights. An episode ends when every seed has risen 50 K (no further
trigger is possible) or at 20 ms. Every round's 1000 pairs are saved; weights are checkpointed after every episode
(--resume continues); after each pass the five members are written as an evaluation root
results/t1000_gym/pass{P}/g0.1 (same layout as t1000_ur10_phys_best) for offline kinetics.member_rollouts.
"""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
from .multiscale_large_data import atomic_json,sha
from .multiscale_data import integrate_labels
from .zhang_model import ZhangCommittee
from .zhang_wang import WangHardCorrection
from .data import mechanism
from . import t1000_online_finetune as O

OUT=Path('results/t1000_gym');SRC=O.SRC;ROOT=O.ROOT;SEEDS=O.SEEDS


def episode(cid,data,w,tuner,max_rounds,log_round,force=False):
    """force: if no triggered round has happened when the first seed reaches T0 + 50 K, run one round there
    (perturbing the states of all active seeds), so every episode contributes data."""
    base=w.predictor.base;gamma=base.gamma;scale=np.r_[base.output_norm.scale[0],base.output_norm.scale[2:]]
    gas=mechanism();dt=data['dt'];n=len(SEEDS)
    S=np.tile(np.r_[data['states'][cid,0],data['pressures'][cid]],(n,1));T0=S[0,0]
    Rbuf=np.full((n,O.WINDOW),np.nan);active=np.ones(n,bool);rounds=0;cool=0;Rmax=0.;step=0;forced=False;Rs=np.full(n,np.nan)
    def do_round(states,j,Rs):
        nonlocal rounds
        rng=np.random.default_rng([cid,log_round.salt,rounds]);starts=O.perturb(states,O.CFG['starts'],rng);xs,ys=[],[];t0=time.perf_counter()
        for s in starts:
            try:xi,yi,_=integrate_labels(s,O.CFG['ode_steps'],dt,gas);xs.append(xi);ys.append(yi)
            except Exception:pass
        xs=np.concatenate(xs);ys=np.concatenate(ys);ode_s=time.perf_counter()-t0
        pick=rng.choice(len(xs),O.CFG['pairs'],replace=False);f0=time.perf_counter()
        rep=tuner.fit(xs[pick],ys[pick],rng);ft_s=time.perf_counter()-f0
        rounds+=1;Rbuf[:]=np.nan
        log_round(cid,rounds,(j-1)*dt,Rs,xs[pick],ys[pick],rep,ode_s,ft_s,len(xs))
    for j in range(1,10001):
        step=j
        if force and rounds==0 and not forced and active.any() and (S[active,0]>=T0+O.RISE).any():
            forced=True;do_round(S[active],j,Rs);cool=O.WINDOW
        if not (active&(S[:,0]<T0+O.RISE)).any():break
        c=w.candidates(S);ok=np.isfinite(c).all(-1)
        D=(O.encode(c[...,:10],gamma)-O.encode(S[:,:10],gamma)[None])/scale;D[~ok]=np.nan
        with np.errstate(all='ignore'):
            R=np.sqrt(np.nansum(np.nanvar(D,axis=0),-1))/(np.linalg.norm(np.nanmean(D,axis=0),axis=-1)+O.EPS)
            Rbuf[:,(j-1)%O.WINDOW]=R;Rs=np.nanmedian(Rbuf,axis=1)
        pre=active&(S[:,0]<T0+O.RISE);cool=max(0,cool-1);Rmax=max(Rmax,float(np.nanmax(np.where(pre,Rs,np.nan))) if pre.any() else 0.)
        if rounds<max_rounds and cool==0 and np.any(pre&(Rs>O.CFG['threshold'])):
            do_round(S[pre],j,Rs);c=w.candidates(S);cool=O.WINDOW
        for k in np.flatnonzero(active):
            y=c[k,k]
            if not np.isfinite(y).all() or abs(y[0])>1e6 or np.max(np.abs(y[1:10]))>1e3:active[k]=False;continue
            S[k]=y
    return dict(case=int(cid),rounds=rounds,forced=forced,steps=step,max_R_smoothed_pre=Rmax,completed_seeds=int(active.sum()))


def snapshot(com,dst):
    (dst/'preprocessing').mkdir(parents=True,exist_ok=True)
    for f in ('preprocessing.json','metric.json'):shutil.copyfile(ROOT/'preprocessing'/f,dst/'preprocessing'/f)
    shutil.copyfile(ROOT/'config.json',dst/'config.json')
    for s,m in zip(SEEDS,com.members):
        d=dst/f'seed{s}';d.mkdir(exist_ok=True);m.model.save(d/'model.keras');shutil.copyfile(ROOT/f'seed{s}/model.json',d/'model.json')
        atomic_json(d/'completed.json',dict(status='completed',seed=s,model_sha256=sha(d/'model.keras'),parent=str(ROOT/f'seed{s}'),
            parent_model_sha256=sha(ROOT/f'seed{s}/model.keras'),method='0-D gym online fine-tune (kinetics.t1000_gym)',selected_model='end of pass'))


def main():
    global ROOT,OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default=str(ROOT),help='starting members (best-validation root)');ap.add_argument('--out',default=str(OUT))
    ap.add_argument('--passes',type=int,default=2);ap.add_argument('--max-rounds',type=int,default=30);ap.add_argument('--resume',action='store_true');ap.add_argument('--force-round',action='store_true',help='one round per episode at the first 50 K rise if none was triggered');ap.add_argument('--limit',type=int,help='debug: episodes per pass')
    a=ap.parse_args();ROOT,OUT=Path(a.root),Path(a.out);OUT.mkdir(parents=True,exist_ok=True);(OUT/'rounds').mkdir(exist_ok=True);(OUT/'ckpt').mkdir(exist_ok=True)
    metric=json.loads((ROOT/'preprocessing/metric.json').read_text())['metric']
    com=ZhangCommittee([str(ROOT/f'seed{s}/model') for s in SEEDS],list(SEEDS),metric,'mean');w=WangHardCorrection(com);tuner=O.Tuner(com)
    with np.load(SRC) as d:data=dict(states=d['states'],pressures=d['pressures'],split=d['split'],dt=float(d['dt']))
    gym=np.flatnonzero(np.isin(data['split'],['train','val']));assert not np.isin(data['split'][gym],'test').any()
    orders=[np.random.default_rng(2026+p).permutation(gym).tolist()[:a.limit] for p in range(a.passes)]
    cfg=dict(O.CFG,window=O.WINDOW,eps=O.EPS,rise=O.RISE,passes=a.passes,max_rounds=a.max_rounds,orders=orders,force_round=a.force_round,
             episodes='train+val initial conditions; test never visited',models=str(ROOT),source_sha256=sha(__file__),online_sha256=sha(O.__file__))
    state_f=OUT/'state.json'
    if state_f.exists():
        assert a.resume,'gym exists; use --resume';old=json.loads((OUT/'config.json').read_text())
        assert old['orders']==orders and old['online_sha256']==cfg['online_sha256'] and old.get('force_round',False)==a.force_round,'configuration changed'
        st=json.loads(state_f.read_text())
        for s,m in zip(SEEDS,com.members):m.model.load_weights(OUT/'ckpt'/f'seed{s}.weights.h5')
        print('RESUME',st,flush=True)
    else:
        atomic_json(OUT/'config.json',cfg);st=dict(pass_=0,episode=0,log=[])
    class Log:
        salt=0
        def __call__(self,cid,r,t,Rs,x,y,rep,ode_s,ft_s,npairs):
            np.savez_compressed(OUT/'rounds'/f'p{st["pass_"]+1}_e{st["episode"]:02d}_c{cid}_r{r:02d}.npz',x=x,y=y)
            print('ROUND',f'p{st["pass_"]+1}',f'e{st["episode"]}',cid,r,f't={t*1e6:.0f}us','R',np.round(Rs,3).tolist(),
                  'val',[f"{m['val_mae_before']:.1e}->{m['val_mae_best']:.1e}" for m in rep],f'ode {ode_s:.1f}s ft {ft_s:.1f}s',flush=True)
    log=Log();t_start=time.perf_counter()
    while st['pass_']<a.passes:
        order=orders[st['pass_']]
        while st['episode']<len(order):
            cid=order[st['episode']];log.salt=st['pass_'];t0=time.perf_counter()
            res=episode(cid,data,w,tuner,a.max_rounds,log,a.force_round);res.update(pass_=st['pass_']+1,episode=st['episode'],split=str(data['split'][cid]),
                T0=float(data['states'][cid,0,0]),seconds=time.perf_counter()-t0)
            print('EPISODE',json.dumps(res),flush=True)
            for s,m in zip(SEEDS,com.members):m.model.save_weights(OUT/'ckpt'/f'seed{s}.weights.h5')
            st['log'].append(res);st['episode']+=1;atomic_json(state_f,st)
        snapshot(com,OUT/f'pass{st["pass_"]+1}/g0.1');print('PASS_DONE',st['pass_']+1,flush=True)
        st['pass_']+=1;st['episode']=0;atomic_json(state_f,st)
    atomic_json(OUT/'manifest.json',dict(status='completed',passes=a.passes,seconds=time.perf_counter()-t_start))
    print('GYM_DONE',flush=True)


if __name__=='__main__':main()
