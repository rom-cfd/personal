"""Memory-mapped full-data Z-scores and durable GELU training, ten full epochs.

No physics closure/projection/clipping. Every epoch shuffles all N examples once.
Model, Adam state and progress are checkpointed together for exact step resume.
"""
import argparse,fcntl,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
from .multiscale_large_data import atomic_json,sha
from .transforms import Standardizer


def stream_moments(iterator):
    count=0;mean=None;m2=None
    for values in iterator:
        values=np.asarray(values,np.float64);n=len(values);bmean=values.mean(axis=0)
        bm2=np.sum((values-bmean)**2,axis=0)
        if count==0:count=n;mean=bmean;m2=bm2;continue
        delta=bmean-mean;total=count+n
        m2+=bm2+delta*delta*(count*n/total);mean+=delta*(n/total);count=total
    scale=np.sqrt(m2/count);scale=np.where(scale>1e-14,scale,1.)
    return mean,scale,count


def encode(x,gamma):
    return np.c_[x[:,0],x[:,-1]/101325.,(x[:,1:10]**gamma-1)/gamma]


def prepare_cache(data_root,output,gamma,chunk=131072):
    data_root=Path(data_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data_root/'manifest.json').read_text());assert manifest['status']=='completed'
    manifest_hash=sha(data_root/'manifest.json')
    metadata_path=output/'preprocessing.json'
    if metadata_path.exists():
        info=json.loads(metadata_path.read_text());assert info['data_manifest_sha256']==manifest_hash and info['gamma']==gamma
        assert info['preprocessing_source_sha256']==sha(__file__),'Preprocessing implementation changed'
        for key in ('features','targets'):assert sha(output/f'{key}.npy')==info[key+'_sha256']
        return info
    x=np.load(data_root/manifest['arrays']['x']['path'],mmap_mode='r')
    y=np.load(data_root/manifest['arrays']['y']['path'],mmap_mode='r');n=len(x)
    for key,array in [('x',x),('y',y)]:assert len(array)==manifest['pairs'] and sha(data_root/manifest['arrays'][key]['path'])==manifest['arrays'][key]['sha256']
    def blocks(target=False):
        for start in range(0,n,chunk):
            xx=x[start:start+chunk];a=encode(xx,gamma)
            if target:a=encode(np.c_[y[start:start+chunk],xx[:,-1]],gamma)-a
            yield a
    im,iscale,ni=stream_moments(blocks());om,oscale,no=stream_moments(blocks(True));assert ni==no==n
    info=dict(gamma=gamma,examples=n,data_manifest_sha256=manifest_hash,input_mean=im.tolist(),input_scale=iscale.tolist(),
        output_mean=om.tolist(),output_scale=oscale.tolist(),target='delta[T_K,P_atm,BCT_Y9]',physics=False,
        preprocessing_source_sha256=sha(__file__))
    for name,mean,scale,target in [('features',im,iscale,False),('targets',om,oscale,True)]:
        path=output/f'{name}.npy';tmp=output/f'{name}.npy.tmp'
        dest=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(n,11));offset=0
        for values in blocks(target):dest[offset:offset+len(values)]=(values-mean)/scale;offset+=len(values)
        dest.flush();del dest;os.replace(tmp,path);info[name+'_sha256']=sha(path)
    atomic_json(metadata_path,info);return info


def permutation_for(n,seed,epoch):
    return np.random.default_rng(np.random.SeedSequence([seed,epoch,20261030])).permutation(n).astype(np.int32)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/zhang_5p3m_gelu.json'))
    parser.add_argument('--seeds',type=int,nargs='+');parser.add_argument('--gamma',type=float);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--run-updates',type=int,help='Checkpoint and pause after this many updates in this invocation')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    gamma=cfg['gamma'] if args.gamma is None else args.gamma;seeds=args.seeds or cfg['seeds']
    trainer_hash=sha(__file__)
    root=Path(cfg['output'])/f'g{gamma:g}';root.mkdir(parents=True,exist_ok=True)
    lock=(root/'training.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Training for this gamma is already running')
    configuration={**cfg,'gamma':gamma}
    configuration.pop('seeds',None);configuration.pop('scope',None)
    saved=root/'config.json'
    if saved.exists():assert json.loads(saved.read_text())==configuration,'Configuration changed'
    else:atomic_json(saved,configuration)
    atomic_json(root/'requested_seeds.json',dict(seeds=seeds))
    info=prepare_cache(cfg['data'],root/'preprocessing',gamma)
    print('PREPROCESSING_READY',info['examples'],'examples gamma',gamma,flush=True)
    if args.prepare_only:return
    from .zhang_model import ZhangPredictor
    from .study import tf
    tf.config.threading.set_intra_op_parallelism_threads(cfg['threads'])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    gpus=tf.config.list_physical_devices('GPU')
    if cfg.get('require_gpu',False) and not gpus:raise RuntimeError('GPU required by config but unavailable')
    for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    features=tf.constant(np.load(root/'preprocessing/features.npy',mmap_mode='r'))
    targets=tf.constant(np.load(root/'preprocessing/targets.npy',mmap_mode='r'))
    n=info['examples'];batches=math.ceil(n/cfg['batch_size']);total=batches*cfg['epochs']
    for seed in seeds:
        config_hash=hashlib.sha256(json.dumps({**configuration,'seed':seed},sort_keys=True).encode()).hexdigest()
        out=root/f'seed{seed}';out.mkdir(exist_ok=True)
        if (out/'completed.json').exists():
            record=json.loads((out/'completed.json').read_text())
            assert record['config_sha256']==config_hash and record['model_sha256']==sha(out/'model.keras')
            assert record['source_sha256']==trainer_hash,'Trainer source changed'
            print('ALREADY_COMPLETE',seed,flush=True);continue
        if (out/'checkpoint.json').exists() and not args.resume:raise FileExistsError('Existing checkpoint; use --resume')
        p=ZhangPredictor.__new__(ZhangPredictor);p.gamma=gamma;p.spec=dict(gamma=gamma)
        p.input_norm=Standardizer(np.array(info['input_mean']),np.array(info['input_scale']))
        p.output_norm=Standardizer(np.array(info['output_mean']),np.array(info['output_scale']))
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
        inputs=tf.keras.Input((11,));h=inputs
        for width in cfg['hidden_layers']:h=tf.keras.layers.Dense(width,activation=cfg['activation'])(h)
        output=tf.keras.layers.Dense(11,kernel_initializer='zeros',bias_initializer='zeros')(h)
        p.model=tf.keras.Model(inputs,output)
        optimizer=tf.keras.optimizers.Adam(cfg['learning_rate']);optimizer.build(p.model.trainable_variables)
        state=tf.train.Checkpoint(model=p.model,optimizer=optimizer)
        manager=tf.train.CheckpointManager(state,str(out/'checkpoints'),max_to_keep=2)
        step0=0;elapsed=0.
        if (out/'checkpoint.json').exists():
            old=json.loads((out/'checkpoint.json').read_text());assert old['config_sha256']==config_hash
            assert old['source_sha256']==trainer_hash,'Trainer source changed'
            state.restore(old['checkpoint']).assert_consumed()
            step0=int(optimizer.iterations.numpy());assert step0==old['completed_updates'];elapsed=old['elapsed_seconds']
        began=time.perf_counter();history=[]
        @tf.function(input_signature=[tf.TensorSpec([None],tf.int32),tf.TensorSpec([],tf.float32)])
        def update(ids,lr):
            optimizer.learning_rate.assign(lr)
            with tf.GradientTape() as tape:
                pred=p.model(tf.gather(features,ids),training=True)
                loss=tf.reduce_mean(tf.abs(pred-tf.gather(targets,ids)))
                tf.debugging.check_numerics(loss,'nonfinite training loss')
            gradients=tape.gradient(loss,p.model.trainable_variables)
            gradients,norm=tf.clip_by_global_norm(gradients,cfg['gradient_clip'])
            optimizer.apply_gradients(zip(gradients,p.model.trainable_variables))
            return loss,norm
        def checkpoint(completed):
            path=manager.save(checkpoint_number=completed)
            value=dict(seed=seed,gamma=gamma,completed_updates=completed,total_updates=total,
                completed_epochs=completed/batches,examples_per_epoch=n,checkpoint=path,
                elapsed_seconds=elapsed+time.perf_counter()-began,config_sha256=config_hash,source_sha256=trainer_hash)
            atomic_json(out/'checkpoint.json',value);atomic_json(out/'progress.json',{**value,'status':'running'})
        last_epoch=-1;order=None
        for step in range(step0,total):
            epoch=step//batches;within=step%batches
            if epoch!=last_epoch:order=permutation_for(n,seed,epoch);last_epoch=epoch
            ids=order[within*cfg['batch_size']:min((within+1)*cfg['batch_size'],n)]
            lr=cfg['learning_rate']*(1 if step<total*.6 else .3 if step<total*.9 else .1)
            loss,norm=update(tf.constant(ids),np.float32(lr));completed=step+1
            if completed%100==0 or completed==total:
                row=dict(seed=seed,update=completed,epoch=epoch+1,loss=float(loss.numpy()),learning_rate=lr,gradient_norm=float(norm.numpy()),elapsed_seconds=elapsed+time.perf_counter()-began)
                with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                atomic_json(out/'progress.json',{**row,'status':'running','total_updates':total,'examples_per_epoch':n})
                print('TRAIN',seed,'update',completed,'/',total,'epoch',epoch+1,'loss',row['loss'],flush=True)
            if completed%cfg['checkpoint_every_updates']==0 or within==batches-1:checkpoint(completed)
            if within==batches-1 and epoch+1 in cfg['model_snapshot_epochs']:
                folder=out/'snapshots';folder.mkdir(exist_ok=True);p.save(folder/f'epoch{epoch+1}')
            if elapsed+time.perf_counter()-began>cfg['max_seconds_per_model']:
                checkpoint(completed);raise TimeoutError('Per-model wall-time budget reached; resume checkpoint is available')
            if args.run_updates and completed-step0>=args.run_updates and completed<total:
                checkpoint(completed)
                atomic_json(out/'progress.json',{**json.loads((out/'checkpoint.json').read_text()),'status':'paused'})
                print('SESSION_PAUSED',seed,completed,flush=True);return
        checkpoint(total);p.save(out/'model')
        record=dict(seed=seed,gamma=gamma,status='completed',epochs=cfg['epochs'],updates=total,
            examples_per_epoch=n,examples_seen=n*cfg['epochs'],elapsed_seconds=elapsed+time.perf_counter()-began,
            model_sha256=sha(out/'model.keras'),metadata_sha256=sha(out/'model.json'),
            config_sha256=config_hash,preprocessing_sha256=sha(root/'preprocessing/preprocessing.json'),physics=False,
            source_sha256=trainer_hash,tensorflow_version=tf.__version__,gpus=[g.name for g in gpus],
            intra_op_threads=cfg['threads'],dtype='float32',tf32_enabled=False)
        atomic_json(out/'completed.json',record);atomic_json(out/'progress.json',record)
        print('MODEL_COMPLETE',record,flush=True)


if __name__=='__main__':main()
