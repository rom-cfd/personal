"""Liu et al. (2026) unrolling fine-tune of a finished one-step Zhang GELU model.

Starting point: the 1000-epoch one-step parent (same architecture, same normalizers).
For each initial state x0 the network is applied N_ur+1 times; its standardized
increment is de-standardized and added in the encoded [T, P_atm, BCT(Y)] space and
fed back (Liu-style: denormalization + addition only, fully differentiable; no
decoding, projection or enthalpy closure during training). The loss is Liu Eq. (5):
the sum over steps i = 0..N_ur of the MAE between the prediction at the fed-back
state and the Cantera increment at the TRUE state of step i (11-step labels from
kinetics.unroll_data). Steps beyond an inadmissible label are masked.
Validation (selection only): the same unrolled loss on 11-step windows of the
validation trajectories, plus the parents' one-step validation MAE for reference.
Physics (Wang correction + enthalpy temperature) is applied only at evaluation.
pin_pressure (config, default false): the pressure-increment output is multiplied by 0 in
every forward pass (training and validation), so the fed-back pressure is exactly constant
and the pressure weights get no gradient. The zero-initialized parents' pressure column is
exactly 0, so the saved model's raw pressure output stays exactly 0 (asserted before saving).
physics_in_loop (config, default false; approach 5): the fed-back state is advanced with the
rollout physics (kinetics.tf_physics: decoder guard, Wang element projection, enthalpy
temperature, pinned pressure), carried in float64 like the rollout state, differentiable.
The loss is unchanged (Liu Eq. 5 against Cantera increments at the true states). A sample
whose corrected state is rejected (negative Y, element residual, infeasible enthalpy root),
as the rollout would reject it, contributes no loss from the next step on.
"""
import argparse,fcntl,hashlib,json,math,time
from pathlib import Path
import numpy as np
from .multiscale_large_data import atomic_json,sha


def windows(cfg,info,ur):
    """Standardized validation windows: f0 (m,11), labels (m,ur+1,11)."""
    from .data import samples
    im,isc,om,osc=[np.array(info[k]) for k in ('input_mean','input_scale','output_mean','output_scale')]
    with np.load(cfg['validation_source']) as d:
        ids=np.flatnonzero(d['split']=='val');S=d['states'][ids];P=d['pressures'][ids]
    g=cfg['gamma'];n,T=S.shape[:2]
    enc=np.concatenate([S[...,:1],np.broadcast_to(P[:,None,None]/101325.,(n,T,1)),(np.maximum(S[...,1:],0)**g-1)/g],-1)
    starts=np.arange(0,T-(ur+1),cfg['validation_window_stride'])
    f0=((enc[:,starts]-im)/isc).reshape(-1,11)
    lab=np.stack([(enc[:,starts+k+1]-enc[:,starts+k]-om)/osc for k in range(ur+1)],2).reshape(-1,ur+1,11)
    vx,vy,_=samples(np.load(cfg['validation_source']),'val')
    e0=np.c_[vx[:,:1],vx[:,-1:]/101325.,(vx[:,1:10]**g-1)/g];e1=np.c_[vy[:,:1],vx[:,-1:]/101325.,(np.maximum(vy[:,1:],0)**g-1)/g]
    return f0.astype(np.float32),lab.astype(np.float32),((e0-im)/isc).astype(np.float32),((e1-e0-om)/osc).astype(np.float32)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=Path('configs/zhang_unroll10.json'))
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--smoke',type=int,help='CPU smoke test: N updates on the first 20k states, separate output')
    a=ap.parse_args();cfg=json.loads(a.config.read_text());seed=a.seed;ur=cfg['unroll']
    from .study import tf
    from .zhang_model import ZhangPredictor
    tf.config.experimental.enable_tensor_float_32_execution(False)
    gpus=tf.config.list_physical_devices('GPU')
    if not a.smoke and cfg.get('require_gpu') and not gpus:raise RuntimeError('No GPU visible; refusing to train on CPU')
    parent_root=Path(cfg['parent_root']);parent=parent_root/f'seed{seed}'
    pr=json.loads((parent/'completed.json').read_text());assert pr['status']=='completed' and pr['model_sha256']==sha(parent/'model.keras')
    info=json.loads((parent_root/'preprocessing/preprocessing.json').read_text())
    data=Path(cfg['data']);dm=json.loads((data/'manifest.json').read_text());assert dm['status']=='completed' and dm['steps']>=ur+1
    root=Path(cfg['output'] if not a.smoke else cfg['output']+'_smoke')/f'g{cfg["gamma"]:g}';out=root/f'seed{seed}'
    out.mkdir(parents=True,exist_ok=True);(root/'preprocessing').mkdir(exist_ok=True)
    for name in ('preprocessing.json','metric.json'):(root/'preprocessing'/name).write_bytes((parent_root/'preprocessing'/name).read_bytes())
    atomic_json(root/'config.json',{**cfg,'physics':False,'physics_in_training':bool(cfg.get('physics_in_loop',False)),'epochs':cfg['epochs']})
    lock=(out/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    binding=dict(config=cfg,seed=seed,parent_model_sha256=pr['model_sha256'],data_manifest_sha256=sha(data/'manifest.json'),source_sha256=sha(__file__))
    bhash=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    if (out/'binding.json').exists():assert json.loads((out/'binding.json').read_text())==binding,'config/data/source changed'
    else:atomic_json(out/'binding.json',binding)
    if (out/'completed.json').exists():print('ALREADY_COMPLETE',flush=True);return
    tf.keras.utils.set_random_seed(seed)
    p=ZhangPredictor.load(parent/'model')
    pin=bool(cfg.get('pin_pressure',False))
    def pressure_column_max():
        W,b=p.model.layers[-1].get_weights();return float(max(np.abs(W[:,1]).max(),abs(b[1])))
    if pin:
        assert info['output_mean'][1]==0. and info['output_scale'][1]==1.,'pressure output must be exactly delta P_atm'
        assert pressure_column_max()==0.,'parent pressure output column is not exactly zero'
    KEEP=tf.constant([1.,0.]+[1.]*9 if pin else [1.]*11,tf.float32)
    net=lambda x,training:p.model(x,training=training)*KEEP   # pinned: delta P == 0 exactly, no gradient to the P column
    def save(path):
        if pin:assert pressure_column_max()==0.,'pressure column moved'
        p.save(path)
    im,isc,om,osc=[tf.constant(np.array(info[k]),tf.float32) for k in ('input_mean','input_scale','output_mean','output_scale')]
    step_add=lambda x,pred:x+(pred*osc+om)/isc      # encoded z_{i+1}=z_i+delta, re-standardized
    physics=bool(cfg.get('physics_in_loop',False))
    if physics:
        assert pin,'physics_in_loop requires pin_pressure (rollout physics keeps pressure constant)'
        from .tf_physics import TFWangPhysics
        phys=TFWangPhysics(info,cfg['gamma'])
    def unroll(x,L,m,training):
        """Per-step masked |error| sums and weights; the state is fed back raw (Liu) or through the physics."""
        sums=[];weights=[]
        if physics:
            z=phys.z_from_features(x);_,_,y0=phys.decode(z);h0=phys.enthalpy(z[:,0],y0);alive=tf.ones_like(m[:,0])
        for k in range(ur+1):
            pred=net(phys.features(z) if physics else x,training)
            w=m[:,k]*alive if physics else m[:,k]
            sums.append(tf.reduce_sum(tf.reduce_mean(tf.abs(pred-L[:,k]),axis=1)*w));weights.append(tf.reduce_sum(w))
            if k<ur:
                if physics:
                    zn,ok,_=phys.step(z,pred,h0);alive=alive*tf.cast(ok,alive.dtype)
                    z=tf.where(ok[:,None],zn,tf.stop_gradient(z))
                else:x=step_add(x,pred)
        return tf.stack(sums),tf.stack(weights)
    f0=np.load(data/'arrays/features0.npy',mmap_mode='r');lab=np.load(data/'arrays/labels.npy',mmap_mode='r');vl=np.load(data/'arrays/valid_len.npy')
    if a.smoke:f0,lab,vl=f0[:20000],lab[:20000],vl[:20000]
    n=len(f0);bs=cfg['batch_size'];batches=math.ceil(n/bs)
    F0=tf.constant(np.asarray(f0));LAB=tf.constant(np.asarray(lab[:,:ur+1]));MASK=tf.constant((np.arange(ur+1)[None]<vl[:,None]).astype(np.float32))
    vf0,vlab,vf1,vt1=windows(cfg,info,ur);vf0,vlab,vf1,vt1=[tf.constant(v) for v in (vf0,vlab,vf1,vt1)]
    lr_at=lambda epoch:cfg['learning_rate']*cfg['lr_step']['factor']**(epoch//cfg['lr_step']['every_epochs'])
    opt=tf.keras.optimizers.Adam(lr_at(0));opt.build(p.model.trainable_variables)
    step=tf.Variable(0,dtype=tf.int64,trainable=False)
    best=tf.Variable(np.inf,dtype=tf.float64,trainable=False)
    state=tf.train.Checkpoint(model=p.model,optimizer=opt,step=step,best=best)
    manager=tf.train.CheckpointManager(state,str(out/'checkpoints'),max_to_keep=2);elapsed=0.
    if (out/'checkpoint.json').exists():
        if not a.resume:raise FileExistsError('existing run; use --resume')
        c=json.loads((out/'checkpoint.json').read_text());assert c['binding_sha256']==bhash
        state.restore(c['checkpoint']).assert_consumed();elapsed=c['elapsed_seconds']
    total=cfg['epochs']*batches;first=int(step.numpy())

    @tf.function(input_signature=[tf.TensorSpec([None],tf.int32)])
    def update(ids):
        x=tf.gather(F0,ids);L=tf.gather(LAB,ids);m=tf.gather(MASK,ids)
        with tf.GradientTape() as tape:
            s,wt=unroll(x,L,m,True);losses=s/tf.maximum(wt,1.)
            loss=tf.reduce_sum(losses)
            tf.debugging.check_numerics(loss,'nonfinite unrolled loss')
        g=tape.gradient(loss,p.model.trainable_variables);g,norm=tf.clip_by_global_norm(g,cfg['gradient_clip'])
        opt.apply_gradients(zip(g,p.model.trainable_variables));step.assign_add(1)
        return loss,losses,norm,wt[-1]/tf.maximum(wt[0],1.)

    @tf.function(input_signature=[tf.TensorSpec([None,11],tf.float32),tf.TensorSpec([None,ur+1,11],tf.float32)])
    def val_unrolled(x,L):
        return unroll(x,L,tf.ones_like(L[:,:,0]),False)
    @tf.function(input_signature=[tf.TensorSpec([None,11],tf.float32),tf.TensorSpec([None,11],tf.float32)])
    def val_one(x,y):return tf.reduce_sum(tf.abs(net(x,False)-y))
    def validation():
        s=np.zeros(ur+1);w=np.zeros(ur+1)
        for i in range(0,len(vf0),4096):
            a_,b_=val_unrolled(vf0[i:i+4096],vlab[i:i+4096]);s+=a_.numpy();w+=b_.numpy()
        s/=np.maximum(w,1.);o=0.;validation.alive=float(w[-1]/len(vf0))
        for i in range(0,len(vf1),4096):o+=float(val_one(vf1[i:i+4096],vt1[i:i+4096]).numpy())
        return float(s.sum()),s.tolist(),o/(len(vf1)*11)
    began=time.perf_counter()
    def checkpoint(status='running'):
        path=manager.save(checkpoint_number=int(step.numpy()))
        r=dict(status=status,seed=seed,completed_updates=int(step.numpy()),total_updates=total,completed_epochs=int(step.numpy())/batches,
            target_epochs=cfg['epochs'],checkpoint=path,elapsed_seconds=elapsed+time.perf_counter()-began,binding_sha256=bhash,
            learning_rate=float(opt.learning_rate.numpy()),best_unrolled_validation=float(best.numpy()))
        atomic_json(out/'checkpoint.json',r);atomic_json(out/'progress.json',r)
    if first==0:
        v,vs,v1=validation()
        atomic_json(out/'parent_validation.json',dict(unrolled_validation=v,unrolled_by_step=vs,one_step_validation_mae=v1,last_step_alive=validation.alive,physics_in_training=physics,parent=str(parent)))
        print('PARENT_VALIDATION',v,v1,flush=True);checkpoint()
    print('UNROLL_START seed',seed,'UR',ur,'pin_pressure',pin,'states',n,'batches/epoch',batches,'from update',first,'gpus',len(gpus),flush=True)
    sums=np.zeros(ur+1);cnt=0;last=-1;order=None;alive_sum=0.
    for it in range(first,total):
        epoch=it//batches;within=it%batches
        if epoch!=last:
            order=np.random.default_rng([seed,epoch]).permutation(n).astype(np.int32);last=epoch
            opt.learning_rate.assign(lr_at(epoch))
        loss,per,norm,alive=update(tf.constant(order[within*bs:(within+1)*bs]));sums+=per.numpy();cnt+=1;alive_sum+=float(alive.numpy())
        done=it+1
        if done%200==0:
            r=dict(epoch=epoch+1,update=done,loss=float(loss.numpy()),first_step=float(per[0].numpy()),last_step=float(per[-1].numpy()),last_step_alive=float(alive.numpy()),
                gradient_norm=float(norm.numpy()),learning_rate=float(opt.learning_rate.numpy()),elapsed_seconds=elapsed+time.perf_counter()-began)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            if done%2000==0:print('TRAIN',json.dumps(r),flush=True)
        if within==batches-1:
            v,vs,v1=validation()
            r=dict(epoch=epoch+1,training_unrolled_loss=float(sums.sum()/cnt),training_by_step=(sums/cnt).tolist(),
                validation_unrolled=v,validation_by_step=vs,validation_one_step_mae=v1,learning_rate=float(opt.learning_rate.numpy()),
                training_last_step_alive=alive_sum/cnt,validation_last_step_alive=validation.alive,
                updates=done,elapsed_seconds=elapsed+time.perf_counter()-began);sums[:]=0;cnt=0;alive_sum=0.
            with (out/'validation_history.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            print('EPOCH',json.dumps({k:r[k] for k in ('epoch','training_unrolled_loss','validation_unrolled','validation_one_step_mae','learning_rate','elapsed_seconds')}),flush=True)
            if v<float(best.numpy()):
                best.assign(v);(out/'best').mkdir(exist_ok=True);save(out/'best/model')
                atomic_json(out/'best/selection.json',{**r,'selection':'unrolled validation loss only','test_used':False,
                    'model_sha256':sha(out/'best/model.keras'),'metadata_sha256':sha(out/'best/model.json')})
            if epoch+1 in cfg['snapshot_epochs']:(out/'snapshots').mkdir(exist_ok=True);save(out/'snapshots'/f'epoch{epoch+1}')
            checkpoint()
        elif done%cfg['checkpoint_every_updates']==0:checkpoint()
        if a.smoke and done-first>=a.smoke:checkpoint('smoke_paused');print('SMOKE_DONE pressure_column_max',pressure_column_max(),flush=True);return
    checkpoint('finished');save(out/'model')
    atomic_json(out/'completed.json',dict(status='completed',seed=seed,unroll=ur,epochs=cfg['epochs'],updates=total,examples_per_epoch=n,
        model_sha256=sha(out/'model.keras'),metadata_sha256=sha(out/'model.json'),best_unrolled_validation=float(best.numpy()),
        parent=str(parent),parent_model_sha256=pr['model_sha256'],binding_sha256=bhash,physics=False,pressure_pinned=pin,physics_in_training=physics,
        temperature_mode_at_evaluation='wang (Wang projection + enthalpy T)',elapsed_seconds=elapsed+time.perf_counter()-began,
        final_model='last epoch; best unrolled validation in best/'))


if __name__=='__main__':main()
