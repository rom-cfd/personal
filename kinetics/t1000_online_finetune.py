"""Uncertainty-triggered online fine-tuning (pure data-driven rollout; ODE only labels new training data).

Per case, the five strategy-5 seeds (best-validation UR10 physics-in-loop members, Wang + enthalpy T at inference)
roll out in lockstep. At every step all five members are queried on each seed's state; seed k advances with its
own member, and R_k = ||std_m D_m|| / (||mean_m D_m|| + 1e-3) is formed on its state (as in t1000_disagreement).

Trigger: some seed is pre-ignition (T < T0 + 50 K) with its 10-step rolling-median R above 0.3 (threshold from
validation), at most --max-rounds rounds per case, none during the 10-step refill after a round. A round:
  1. take the current states of the pre-ignition seeds (5 when all are pre-ignition);
  2. perturb each into 100/n starts: T + N(0, 5 K); Y_i -> max(Y_i,1e-12)*exp(N(0, 0.1)) for i != N2; N2 = 1 - sum;
     P unchanged;
  3. Cantera (reference mechanism, rtol 1e-10, atol 1e-20) for 100 steps of dt from each start -> 10 000 pairs;
  4. random 1 000 pairs, 900/100 train/validation;
  5. fine-tune ALL five members from their current weights: MAE on standardized increments (pretraining loss,
     pressure column pinned) + lambda * sum (theta - theta_pretrained)^2 (L2-SP, lambda 1e-3), Adam 1e-5,
     batch 100, clip 5, up to 200 epochs, patience 30, best-validation weights kept.
Adam 1e-5, not the first-proposed 1e-4: at 1e-4 the round-internal validation MAE barely moved (updates coarser than
the ~1e-3 local error); 1e-5 cut it ~10x on validation case 52 (R over the next 200 us 0.92 -> 0.04).
Weights are reset to the pretrained ones at the start of every case (per-trajectory adaptation).
The reference trajectory is used only for scoring.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .multiscale_large_data import atomic_json,sha
from .multiscale_data import integrate_labels
from .zhang_model import ZhangCommittee
from .zhang_wang import WangHardCorrection
from .zhang_raw_study import assess
from .data import mechanism,SPECIES
from .study import tf

ROOT=Path('results/t1000_ur10_phys_best/g0.1');SEEDS=(11,22,33,44,55)
SRC=Path('results/t1000_long10ms_data/trajectories.npz');REF=Path('results/t1000_reference/reference20ms.npz')
OUT=Path('results/t1000_online_ft');EPS=1e-3;WINDOW=10;RISE=50.
CFG=dict(threshold=.3,starts=100,ode_steps=100,pairs=1000,val_pairs=100,sigma_T=5.,sigma_logY=.1,floor_Y=1e-12,
         l2sp_lambda=1e-3,learning_rate=1e-5,batch=100,epochs=200,patience=30,gradient_clip=5.)


def encode(s,gamma):
    with np.errstate(invalid='ignore',divide='ignore'):
        return np.concatenate([s[...,:1],(np.maximum(s[...,1:10],0)**gamma-1)/gamma],axis=-1)


def perturb(states,count,rng):
    n2=SPECIES.index('N2');out=[]
    while len(out)<count:
        s=states[len(out)%len(states)].copy()
        s[0]+=rng.normal(0,CFG['sigma_T'])
        y=np.maximum(s[1:10],CFG['floor_Y'])*np.exp(rng.normal(0,CFG['sigma_logY'],9));y[n2]=0;y[n2]=1-y.sum()
        if y[n2]<=0:continue
        s[1:10]=y;out.append(s)
    return np.asarray(out)


class Tuner:
    def __init__(self,committee):
        self.c=committee;self.base=committee.base;self.models=[m.model for m in committee.members]
        self.theta0=[[tf.constant(v) for v in m.get_weights()] for m in self.models]
        assert committee.base.output_norm.mean[1]==0. and committee.base.output_norm.scale[1]==1.
        for m in self.models:
            W,b=m.layers[-1].get_weights();assert np.abs(W[:,1]).max()==0. and b[1]==0.,'pressure column not pinned'
        self.keep=tf.constant([1.,0.]+[1.]*9,tf.float32)

    def reset(self):
        for m,t in zip(self.models,self.theta0):m.set_weights([v.numpy() for v in t])

    def fit(self,x,ynext,rng):
        f=self.base.features(x).astype('float32')
        t=self.base.output_norm.forward(self.base.encode(np.c_[ynext,x[:,-1]])-self.base.encode(x)).astype('float32')
        order=rng.permutation(len(x));va,tr=order[:CFG['val_pairs']],order[CFG['val_pairs']:]
        fv,tv=tf.constant(f[va]),tf.constant(t[va]);report=[]
        for m,anchor in zip(self.models,self.theta0):
            keep=self.keep;opt=tf.keras.optimizers.Adam(CFG['learning_rate']);lam=CFG['l2sp_lambda']
            @tf.function
            def step(xb,tb):
                with tf.GradientTape() as tape:
                    data=tf.reduce_mean(tf.abs(m(xb,training=True)*keep-tb))
                    sp=tf.add_n([tf.reduce_sum(tf.square(v-a)) for v,a in zip(m.trainable_variables,anchor)])
                    loss=data+lam*sp
                g=tape.gradient(loss,m.trainable_variables);g,_=tf.clip_by_global_norm(g,CFG['gradient_clip'])
                opt.apply_gradients(zip(g,m.trainable_variables));return data,sp
            val=lambda:float(tf.reduce_mean(tf.abs(m(fv,training=False)*keep-tv)).numpy())
            v0=val();best=v0;bw=m.get_weights();bad=0;ep=0;sp=0.
            for ep in range(1,CFG['epochs']+1):
                p=rng.permutation(tr)
                for i in range(0,len(p),CFG['batch']):
                    _,sp=step(tf.constant(f[p[i:i+CFG['batch']]]),tf.constant(t[p[i:i+CFG['batch']]]))
                v=val()
                if v<best:best,bw,bad=v,m.get_weights(),0
                else:
                    bad+=1
                    if bad>=CFG['patience']:break
            m.set_weights(bw);report.append(dict(val_mae_before=v0,val_mae_best=best,epochs=ep,l2sp_last=float(sp)))
        return report


def run_case(cid,w,tuner,data,max_rounds):
    base=w.predictor.base;gamma=base.gamma;scale=np.r_[base.output_norm.scale[0],base.output_norm.scale[2:]]
    refs,taus,pressures,matrix,dt=data['refs'],data['taus'],data['pressures'],data['matrix'],data['dt']
    tuner.reset();gas=mechanism();steps=10000;n=len(SEEDS)
    S=np.tile(np.r_[refs[cid][0],pressures[cid]],(n,1));T0=S[0,0]
    hist=np.full((n,steps+1,11),np.nan);hist[:,0]=S;Rh=np.full((n,steps+1),np.nan);Rbuf=np.full((n,WINDOW),np.nan)
    active=np.ones(n,bool);status=['completed']*n;rounds=[];cool=0;net_s=0.
    def query(S):
        c=w.candidates(S);ok=np.isfinite(c).all(-1)
        D=(encode(c[...,:10],gamma)-encode(S[:,:10],gamma)[None])/scale;D[~ok]=np.nan
        with np.errstate(all='ignore'):
            R=np.sqrt(np.nansum(np.nanvar(D,axis=0),-1))/(np.linalg.norm(np.nanmean(D,axis=0),axis=-1)+EPS)
        return c,R
    for j in range(1,steps+1):
        if not active.any():break
        t0=time.perf_counter();c,R=query(S);net_s+=time.perf_counter()-t0
        Rbuf[:,(j-1)%WINDOW]=R
        with np.errstate(all='ignore'):Rs=np.nanmedian(Rbuf,axis=1)
        pre=active&(S[:,0]<T0+RISE);cool=max(0,cool-1)
        if len(rounds)<max_rounds and cool==0 and np.any(pre&(Rs>CFG['threshold'])):
            r0=time.perf_counter();rng=np.random.default_rng([cid,len(rounds)])
            starts=perturb(S[pre],CFG['starts'],rng);xs,ys=[],[];fails=0
            for s in starts:
                try:xi,yi,_=integrate_labels(s,CFG['ode_steps'],dt,gas);xs.append(xi);ys.append(yi)
                except Exception:fails+=1
            xs=np.concatenate(xs);ys=np.concatenate(ys);ode_s=time.perf_counter()-r0
            pick=rng.choice(len(xs),CFG['pairs'],replace=False)
            f0=time.perf_counter();rep=tuner.fit(xs[pick],ys[pick],rng);ft_s=time.perf_counter()-f0
            c,R=query(S);Rbuf[:]=np.nan;Rbuf[:,(j-1)%WINDOW]=R;cool=WINDOW
            rounds.append(dict(step=j-1,time_s=(j-1)*dt,seeds_used=[SEEDS[k] for k in np.flatnonzero(pre)],
                R_smoothed_before=Rs.tolist(),R_after=R.tolist(),ode_pairs=int(len(xs)),ode_failed_starts=fails,
                ode_seconds=ode_s,finetune_seconds=ft_s,members=rep,T_range=[float(starts[:,0].min()),float(starts[:,0].max())]))
            print('ROUND',cid,len(rounds),f't={(j-1)*dt*1e6:.0f}us','R',np.round(Rs,3).tolist(),'->',np.round(R,3).tolist(),
                  'val',[f"{m['val_mae_before']:.2e}->{m['val_mae_best']:.2e}" for m in rep],f'ode {ode_s:.1f}s ft {ft_s:.1f}s',flush=True)
        Rh[:,j-1]=R
        for k in np.flatnonzero(active):
            y=c[k,k];finite=np.isfinite(y).all()
            huge=bool(finite and (abs(y[0])>1e6 or np.max(np.abs(y[1:10]))>1e3 or abs(y[-1])>1e15))
            if not finite or huge:active[k]=False;status[k]='numeric_magnitude_guard' if huge else 'undefined_BCT_inverse_or_nonfinite';continue
            S[k]=y;hist[k,j]=y
    rows=[]
    for k,s in enumerate(SEEDS):
        tr=hist[k][np.isfinite(hist[k]).all(-1)]
        for h in (.01,.02):
            q,_=assess(tr,refs[cid],pressures[cid],dt,matrix,h,cid,status[k],taus[cid])
            rows.append(dict(seed=s,horizon=h,rounds=len(rounds),**q))
    return hist,Rh,rounds,rows,net_s


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases',type=int,nargs='+');ap.add_argument('--max-rounds',type=int,default=30)
    ap.add_argument('--name',default='online_ft_l2sp1e-3_lr1e-5');a=ap.parse_args()
    out=OUT/a.name;out.mkdir(parents=True,exist_ok=True)
    metric=json.loads((ROOT/'preprocessing/metric.json').read_text())['metric']
    com=ZhangCommittee([str(ROOT/f'seed{s}/model') for s in SEEDS],list(SEEDS),metric,'mean');w=WangHardCorrection(com);tuner=Tuner(com)
    with np.load(SRC) as d:split=d['split'];pressures=d['pressures'];matrix=d['element_matrix'];dt=float(d['dt'])
    with np.load(REF) as d:refs={int(i):x for i,x in zip(d['ids'],d['states'])};taus={int(i):t for i,t in zip(d['ids'],d['refined_tau400'])}
    data=dict(refs=refs,taus=taus,pressures=pressures,matrix=matrix,dt=dt)
    atomic_json(out/'config.json',dict(CFG,window=WINDOW,eps=EPS,rise=RISE,max_rounds=a.max_rounds,models=str(ROOT),seeds=list(SEEDS),source_sha256=sha(__file__)))
    for cid in (a.cases or sorted(refs)):
        d=out/f'case{cid}'
        if (d/'done.json').exists():continue
        d.mkdir(exist_ok=True);t0=time.perf_counter()
        hist,Rh,rounds,rows,net_s=run_case(cid,w,tuner,data,a.max_rounds)
        pd.DataFrame([dict(case=cid,split=str(split[cid]),**r) for r in rows]).to_csv(d/'trajectories.csv',index=False)
        np.savez_compressed(d/'traces.npz',T=hist[...,0].astype('float32'),R=Rh.astype('float32'),states=hist.astype('float32'))
        atomic_json(d/'rounds.json',rounds)
        atomic_json(d/'done.json',dict(case=cid,rounds=len(rounds),wall_seconds=time.perf_counter()-t0,network_seconds=net_s))
        r2=[r for r in rows if r['horizon']==.02]
        print('CASE',cid,str(split[cid]),'rounds',len(rounds),'tau_err',[None if np.isnan(r['tau400_relative_error_refined']) else round(r['tau400_relative_error_refined'],4) for r in r2],
              'T_rmse',[round(r['full_temperature_rmse'],2) for r in r2],f'{time.perf_counter()-t0:.0f}s',flush=True)


if __name__=='__main__':main()
