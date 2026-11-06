"""EWC consolidation of the 0-D gym data into the strategy-5 members (offline model).

New task: all Cantera-labelled pairs gathered by kinetics.t1000_gym (results/t1000_gym/rounds, train+val episodes
only); 10% of the ROUND FILES held out as new-task validation. Old task: the original training distribution.
For each seed the diagonal empirical Fisher F of the pretrained model (mean squared per-sample gradient of the
pretraining loss, standardized-increment MAE with pinned pressure) is estimated on 20 000 random pairs of
results/t1000_gelu/prep/g0.1/preprocessing (the 5.3M training set). Fine-tune from the pretrained weights on the
gym pairs with  MAE + lam * sum_i F_i (theta_i - theta0_i)^2,  lam = c / mean(F), c in --cs (c=0: no penalty).
Adam 1e-5, batch 256, clip 5, <= 40 epochs, patience 8. Epoch selection (validation only): mean of new-task val MAE
and old-task val MAE (one-step pairs of the val trajectories in results/t1000_long10ms_data, the pretraining
validation set), each relative to its pretrained value. Writes evaluation roots results/t1000_ewc/c{c}/g0.1.
"""
import argparse,glob,json,shutil,time
from pathlib import Path
import numpy as np
from .multiscale_large_data import atomic_json,sha
from .zhang_model import ZhangPredictor
from .data import samples
from .study import tf

ROOT=Path('results/t1000_ur10_phys_best/g0.1');PREP=Path('results/t1000_gelu/prep/g0.1/preprocessing')
GYM=Path('results/t1000_gym/rounds');SRC=Path('results/t1000_long10ms_data/trajectories.npz');OUT=Path('results/t1000_ewc')
CFG=dict(fisher_pairs=20000,learning_rate=1e-5,batch=256,epochs=40,patience=8,gradient_clip=5.,val_round_fraction=.1)
KEEP=tf.constant([1.,0.]+[1.]*9,tf.float32)


def gym_pairs(p):
    files=sorted(glob.glob(str(GYM/'*.npz')));rng=np.random.default_rng(7);val=set(rng.choice(len(files),max(1,round(len(files)*CFG['val_round_fraction'])),replace=False).tolist())
    def load(ids):
        xs,ys=zip(*[(np.load(files[i])['x'],np.load(files[i])['y']) for i in ids]);x=np.concatenate(xs);y=np.concatenate(ys)
        return p.features(x).astype('float32'),p.output_norm.forward(p.encode(np.c_[y,x[:,-1]])-p.encode(x)).astype('float32')
    return load([i for i in range(len(files)) if i not in val]),load(sorted(val)),[Path(files[i]).name for i in sorted(val)],len(files)


def fisher(model,f,t,chunk=32):
    var=model.trainable_variables;acc=[tf.zeros_like(v) for v in var]
    @tf.function
    def sq(xb,tb):
        def one(a):
            x,y=a
            with tf.GradientTape() as tape:l=tf.reduce_mean(tf.abs(model(x[None],training=False)*KEEP-y[None]))
            return tape.gradient(l,var)
        g=tf.vectorized_map(one,(xb,tb));return [tf.reduce_sum(tf.square(gi),0) for gi in g]
    for i in range(0,len(f),chunk):
        s=sq(tf.constant(f[i:i+chunk]),tf.constant(t[i:i+chunk]));acc=[a+b for a,b in zip(acc,s)]
    return [a/len(f) for a in acc]


def train(p,theta0,F,lam,tr,va,old,log):
    m=p.model;m.set_weights([v.numpy() for v in theta0]);opt=tf.keras.optimizers.Adam(CFG['learning_rate'])
    fx,ft=tr;var=m.trainable_variables
    @tf.function
    def step(xb,tb):
        with tf.GradientTape() as tape:
            data=tf.reduce_mean(tf.abs(m(xb,training=True)*KEEP-tb))
            pen=tf.add_n([tf.reduce_sum(Fi*tf.square(v-a)) for v,a,Fi in zip(var,theta0,F)])
            loss=data+lam*pen
        g=tape.gradient(loss,var);g,_=tf.clip_by_global_norm(g,CFG['gradient_clip']);opt.apply_gradients(zip(g,var));return data,pen
    @tf.function
    def mae_b(x,y):return tf.reduce_sum(tf.abs(m(x,training=False)*KEEP-y))
    def mae(d):
        x,y=d;s=0.
        for i in range(0,len(x),8192):s+=float(mae_b(tf.constant(x[i:i+8192]),tf.constant(y[i:i+8192])).numpy())
        return s/(len(x)*11)
    n0,o0=mae(va),mae(old);best=(1.,0);bw=m.get_weights();bad=0;rng=np.random.default_rng(0);hist=[]
    for ep in range(1,CFG['epochs']+1):
        order=rng.permutation(len(fx));t0=time.perf_counter()
        for i in range(0,len(order),CFG['batch']):
            idx=np.sort(order[i:i+CFG['batch']]);d,pen=step(tf.constant(fx[idx]),tf.constant(ft[idx]))
        n,o=mae(va),mae(old);score=(n/n0+o/o0)/2
        hist.append(dict(epoch=ep,new_val=n,old_val=o,score=score,penalty=float(pen),seconds=time.perf_counter()-t0));log(hist[-1])
        if score<best[0]:best,bw,bad=(score,ep),m.get_weights(),0
        else:
            bad+=1
            if bad>=CFG['patience']:break
    m.set_weights(bw);return dict(new_val_pretrained=n0,old_val_pretrained=o0,best_score=best[0],best_epoch=best[1],
        best=next(h for h in hist if h['epoch']==best[1]) if best[1] else None,history=hist)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55])
    ap.add_argument('--cs',type=float,nargs='+',default=[0.,.01,.1,1.,10.]);a=ap.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/'fisher').mkdir(exist_ok=True)
    atomic_json(OUT/f'config.{"-".join(map(str,a.seeds))}.json',dict(CFG,cs=a.cs,seeds=a.seeds,models=str(ROOT),gym_rounds=str(GYM),
        fisher_data=str(PREP),source_sha256=sha(__file__)))
    for s in a.seeds:
        p=ZhangPredictor.load(ROOT/f'seed{s}/model');theta0=[tf.constant(v) for v in p.model.get_weights()]
        assert np.array_equal(p.input_norm.mean,np.array(json.loads((PREP/'preprocessing.json').read_text())['input_mean']))
        tr,va,val_files,nfiles=gym_pairs(p)
        with np.load(SRC) as d:vx,vy,_=samples(d,'val')
        old=(p.features(vx).astype('float32'),p.output_norm.forward(p.encode(np.c_[vy,vx[:,-1]])-p.encode(vx)).astype('float32'))
        ff=OUT/'fisher'/f'seed{s}.npz'
        if ff.exists():
            with np.load(ff) as d:F=[tf.constant(d[f'f{i}']) for i in range(len(theta0))]
        else:
            fe=np.load(PREP/'features.npy',mmap_mode='r');te=np.load(PREP/'targets.npy',mmap_mode='r')
            idx=np.sort(np.random.default_rng(s).choice(len(fe),CFG['fisher_pairs'],replace=False))
            t0=time.perf_counter();F=fisher(p.model,np.asarray(fe[idx]),np.asarray(te[idx]))
            np.savez(ff,**{f'f{i}':v.numpy() for i,v in enumerate(F)})
            print('FISHER',s,f'{time.perf_counter()-t0:.0f}s',flush=True)
        meanF=float(sum(float(tf.reduce_sum(f)) for f in F)/sum(int(tf.size(f)) for f in F))
        print('DATA',s,'gym train',len(tr[0]),'gym val',len(va[0]),'old val',len(old[0]),'mean F',meanF,flush=True)
        for c in a.cs:
            tag=f'c{c:g}';dst=OUT/tag/'g0.1';d=dst/f'seed{s}'
            if (d/'completed.json').exists():continue
            lam=c/meanF if c>0 else 0.;t0=time.perf_counter()
            rec=train(p,theta0,F,lam,tr,va,old,lambda h:print('EPOCH',s,tag,json.dumps({k:round(v,6) if isinstance(v,float) else v for k,v in h.items()}),flush=True))
            (dst/'preprocessing').mkdir(parents=True,exist_ok=True)
            for f in ('preprocessing.json','metric.json'):shutil.copyfile(ROOT/'preprocessing'/f,dst/'preprocessing'/f)
            shutil.copyfile(ROOT/'config.json',dst/'config.json');d.mkdir(exist_ok=True)
            p.model.save(d/'model.keras');shutil.copyfile(ROOT/f'seed{s}/model.json',d/'model.json')
            atomic_json(d/'training.json',dict(rec,c=c,lam=lam,mean_fisher=meanF,gym_val_files=val_files,gym_files=nfiles,seconds=time.perf_counter()-t0))
            atomic_json(d/'completed.json',dict(status='completed',seed=s,model_sha256=sha(d/'model.keras'),parent=str(ROOT/f'seed{s}'),
                parent_model_sha256=sha(ROOT/f'seed{s}/model.keras'),method=f'EWC consolidation of gym data, c={c:g}',selected_model='best validation (new+old)'))
            b=rec['best'] or {};print('DONE',s,tag,'best_epoch',rec['best_epoch'],'new_val',f"{rec['new_val_pretrained']:.2e}->{b.get('new_val',float('nan')):.2e}",
                  'old_val',f"{rec['old_val_pretrained']:.2e}->{b.get('old_val',float('nan')):.2e}",f'{time.perf_counter()-t0:.0f}s',flush=True)


if __name__=='__main__':main()
