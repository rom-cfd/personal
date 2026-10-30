"""Continue immutable GELU/Adam checkpoints with validation ReduceLROnPlateau.

Epoch numbering and update numbering are separate so changing batch size does
not repeat the first ten epochs or confuse Adam's retained iteration counter.
"""
import argparse,fcntl,hashlib,json,math,time
from pathlib import Path
import numpy as np
from .multiscale_large_data import atomic_json,sha
from .large_zhang_training import prepare_cache,encode,permutation_for
from .transforms import Standardizer


def plateau_step(state,value,lr,config):
    state=dict(state)
    if not np.isfinite(value):raise ValueError('Nonfinite validation loss')
    if value < state['best']-config['min_delta']:
        state.update(best=float(value),wait=0)
    elif state['cooldown']>0:
        state.update(cooldown=state['cooldown']-1,wait=0)
    else:
        state['wait']+=1
        if state['wait']>=config['patience']:
            lr=max(lr*config['factor'],config['min_lr'])
            state.update(wait=0,cooldown=config['cooldown'])
    return state,float(lr)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--run-updates',type=int)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    from .study import tf
    from .zhang_model import ZhangPredictor
    tf.config.threading.set_intra_op_parallelism_threads(cfg['threads'])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    seed=cfg['seed'];base=Path(cfg['base_root']);parent=base/f'seed{seed}'
    parent_record=json.loads((parent/'completed.json').read_text())
    parent_config=json.loads((base/'config.json').read_text())
    assert parent_record['status']=='completed' and cfg['epochs']>parent_record['epochs']
    for key in ['data','gamma','hidden_layers','activation','gradient_clip','loss']:
        assert cfg[key]==parent_config[key],f'Continuation changes {key}'
    assert sha(parent/'model.keras')==parent_record['model_sha256']
    assert sha(parent/'model.json')==parent_record['metadata_sha256']
    assert parent_record['source_sha256']==sha(Path(__file__).with_name('large_zhang_training.py'))
    info=prepare_cache(cfg['data'],base/'preprocessing',cfg['gamma'])
    assert sha(base/'preprocessing/preprocessing.json')==parent_record['preprocessing_sha256']
    n=info['examples'];batches=math.ceil(n/cfg['batch_size']);base_epochs=parent_record['epochs']
    additional=(cfg['epochs']-base_epochs)*batches
    root=Path(cfg['output'])/f'g{cfg["gamma"]:g}';root.mkdir(parents=True,exist_ok=True)
    lock=(root/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=root/f'seed{seed}';out.mkdir(exist_ok=True)
    source_hash=sha(__file__)
    binding=dict(config=cfg,parent_record_sha256=sha(parent/'completed.json'),
        preprocessing_sha256=parent_record['preprocessing_sha256'],validation_sha256=sha(cfg['validation_source']))
    binding_hash=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    if (root/'binding.json').exists():assert json.loads((root/'binding.json').read_text())==binding
    else:atomic_json(root/'binding.json',binding)
    atomic_json(root/'config.json',cfg)
    (root/'preprocessing').mkdir(exist_ok=True)
    (root/'preprocessing/preprocessing.json').write_bytes((base/'preprocessing/preprocessing.json').read_bytes())
    if (out/'completed.json').exists():
        r=json.loads((out/'completed.json').read_text())
        assert r['config_sha256']==binding_hash and r['model_sha256']==sha(out/'model.keras')
        print('ALREADY_COMPLETE',flush=True);return
    p=ZhangPredictor.load(parent/'model');optimizer=tf.keras.optimizers.Adam(cfg['learning_rate'])
    optimizer.build(p.model.trainable_variables)
    # Old checkpoint has model/Adam only. Restore it before adding scheduler state.
    old=json.loads((parent/'checkpoint.json').read_text());assert old['completed_updates']==parent_record['updates']
    assert old['source_sha256']==parent_record['source_sha256'] and old['config_sha256']==parent_record['config_sha256']
    base_state=tf.train.Checkpoint(model=p.model,optimizer=optimizer)
    expected=[v.numpy().copy() for v in p.model.weights]
    base_state.restore(old['checkpoint']).assert_consumed()
    for a,b in zip(expected,p.model.weights):np.testing.assert_array_equal(a,b.numpy())
    base_updates=int(optimizer.iterations.numpy());assert base_updates==parent_record['updates']
    optimizer.learning_rate.assign(cfg['learning_rate'])
    scheduler=tf.Module()
    scheduler.best=tf.Variable(np.inf,dtype=tf.float64,trainable=False)
    scheduler.best_model=tf.Variable(np.inf,dtype=tf.float64,trainable=False)
    scheduler.wait=tf.Variable(0,dtype=tf.int64,trainable=False)
    scheduler.cooldown=tf.Variable(0,dtype=tf.int64,trainable=False)
    local_step=tf.Variable(0,dtype=tf.int64,trainable=False)
    state=tf.train.Checkpoint(model=p.model,optimizer=optimizer,scheduler=scheduler,continuation_updates=local_step)
    manager=tf.train.CheckpointManager(state,str(out/'checkpoints'),max_to_keep=2)
    elapsed=parent_record['elapsed_seconds']
    if (out/'checkpoint.json').exists():
        if not args.resume:raise FileExistsError('Use --resume for existing continuation')
        previous=json.loads((out/'checkpoint.json').read_text())
        assert previous['config_sha256']==binding_hash and previous['source_sha256']==source_hash
        state.restore(previous['checkpoint']).assert_consumed();elapsed=previous['elapsed_seconds']
    first_step=int(local_step.numpy());assert int(optimizer.iterations.numpy())==base_updates+first_step
    features=tf.constant(np.load(base/'preprocessing/features.npy',mmap_mode='r'))
    targets=tf.constant(np.load(base/'preprocessing/targets.npy',mmap_mode='r'))
    from .data import samples
    with np.load(cfg['validation_source']) as d:vx,vy,_=samples(d,'val')
    vf=(encode(vx,cfg['gamma'])-np.array(info['input_mean']))/np.array(info['input_scale'])
    vt=(encode(np.c_[vy,vx[:,-1]],cfg['gamma'])-encode(vx,cfg['gamma'])-np.array(info['output_mean']))/np.array(info['output_scale'])
    vf=tf.constant(vf,dtype=tf.float32);vt=tf.constant(vt,dtype=tf.float32)
    @tf.function(input_signature=[tf.TensorSpec([None],tf.int32)])
    def update(ids):
        with tf.GradientTape() as tape:
            pred=p.model(tf.gather(features,ids),training=True)
            loss=tf.reduce_mean(tf.abs(pred-tf.gather(targets,ids)))
            tf.debugging.check_numerics(loss,'nonfinite training loss')
        gradients=tape.gradient(loss,p.model.trainable_variables)
        gradients,norm=tf.clip_by_global_norm(gradients,cfg['gradient_clip'])
        optimizer.apply_gradients(zip(gradients,p.model.trainable_variables));local_step.assign_add(1)
        return loss,norm
    @tf.function(input_signature=[tf.TensorSpec([None,11],tf.float32),tf.TensorSpec([None,11],tf.float32)])
    def validation_batch(x,y):return tf.reduce_sum(tf.abs(p.model(x,training=False)-y))
    def validation():
        total=0.
        for i in range(0,len(vx),cfg['validation_batch_size']):
            total+=float(validation_batch(vf[i:i+cfg['validation_batch_size']],vt[i:i+cfg['validation_batch_size']]).numpy())
        return total/(len(vx)*11)
    def set_scheduler(s):
        scheduler.best.assign(s['best']);scheduler.wait.assign(s['wait']);scheduler.cooldown.assign(s['cooldown'])
    def get_scheduler():return dict(best=float(scheduler.best.numpy()),wait=int(scheduler.wait.numpy()),cooldown=int(scheduler.cooldown.numpy()))
    began=time.perf_counter()
    def checkpoint():
        completed=int(local_step.numpy());path=manager.save(checkpoint_number=base_updates+completed)
        r=dict(status='running',seed=seed,gamma=cfg['gamma'],completed_updates=base_updates+completed,
            continuation_updates=completed,total_updates=base_updates+additional,
            completed_epochs=base_epochs+completed/batches,target_epochs=cfg['epochs'],examples_per_epoch=n,
            examples_seen=n*(base_epochs+completed//batches)+min((completed%batches)*cfg['batch_size'],n),
            checkpoint=path,elapsed_seconds=elapsed+time.perf_counter()-began,
            source_sha256=source_hash,config_sha256=binding_hash,learning_rate=float(optimizer.learning_rate.numpy()),scheduler=get_scheduler())
        atomic_json(out/'checkpoint.json',r);atomic_json(out/'progress.json',r)
    if first_step==0:
        v=validation();set_scheduler(dict(best=v,wait=0,cooldown=0));scheduler.best_model.assign(v)
        (out/'best').mkdir(exist_ok=True);p.save(out/'best/model')
        atomic_json(out/'best/selection.json',dict(epoch=base_epochs,validation_mae=v,selection='validation only',test_used=False,
            model_sha256=sha(out/'best/model.keras'),metadata_sha256=sha(out/'best/model.json')))
        atomic_json(out/'parent_restored.json',dict(parent=parent_record,model_weights_exactly_match=True,
            adam_restored=True,optimizer_iterations=base_updates,validation_mae=v,continuation_batch_size=cfg['batch_size']))
        checkpoint()
    print('CONTINUATION_START',base_epochs+first_step/batches,'->',cfg['epochs'],'batch',cfg['batch_size'],flush=True)
    last_epoch=-1;order=None
    for step in range(first_step,additional):
        epoch=base_epochs+step//batches;within=step%batches
        if epoch!=last_epoch:order=permutation_for(n,seed,epoch);last_epoch=epoch
        ids=order[within*cfg['batch_size']:min((within+1)*cfg['batch_size'],n)]
        loss,norm=update(tf.constant(ids));completed=step+1
        if within==batches-1:
            v=validation();s,new_lr=plateau_step(get_scheduler(),v,float(optimizer.learning_rate.numpy()),cfg['scheduler'])
            set_scheduler(s);optimizer.learning_rate.assign(new_lr)
            r=dict(epoch=epoch+1,validation_mae=v,learning_rate_next_epoch=new_lr,
                optimizer_updates=base_updates+completed,elapsed_seconds=elapsed+time.perf_counter()-began)
            with (out/'validation_history.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            if v<float(scheduler.best_model.numpy()):
                p.save(out/'best/model');scheduler.best_model.assign(v)
                atomic_json(out/'best/selection.json',{**r,'selection':'validation MAE only','test_used':False,
                    'model_sha256':sha(out/'best/model.keras'),'metadata_sha256':sha(out/'best/model.json')})
            if epoch+1 in cfg['model_snapshot_epochs']:
                (out/'snapshots').mkdir(exist_ok=True);p.save(out/'snapshots'/f'epoch{epoch+1}')
            print('VALIDATION',json.dumps(r),flush=True)
        if completed%100==0:
            r=dict(epoch=epoch+1,completed_epochs=base_epochs+completed/batches,continuation_updates=completed,
                completed_updates=base_updates+completed,total_updates=base_updates+additional,target_epochs=cfg['epochs'],
                loss=float(loss.numpy()),gradient_norm=float(norm.numpy()),learning_rate=float(optimizer.learning_rate.numpy()),
                elapsed_seconds=elapsed+time.perf_counter()-began,status='running')
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            atomic_json(out/'progress.json',r);print('TRAIN',json.dumps(r),flush=True)
        if completed%cfg['checkpoint_every_updates']==0 or within==batches-1:checkpoint()
        if args.run_updates and completed-first_step>=args.run_updates and completed<additional:
            checkpoint();atomic_json(out/'progress.json',{**json.loads((out/'checkpoint.json').read_text()),'status':'paused'});return
        if time.perf_counter()-began>cfg['max_seconds_per_invocation']:
            checkpoint();raise TimeoutError('Invocation time limit reached; checkpoint can be resumed')
    checkpoint();p.save(out/'model')
    record=dict(status='completed',seed=seed,gamma=cfg['gamma'],epochs=cfg['epochs'],updates=base_updates+additional,
        examples_per_epoch=n,examples_seen=n*cfg['epochs'],elapsed_seconds=elapsed+time.perf_counter()-began,
        model_sha256=sha(out/'model.keras'),metadata_sha256=sha(out/'model.json'),config_sha256=binding_hash,
        preprocessing_sha256=sha(root/'preprocessing/preprocessing.json'),source_sha256=source_hash,
        physics=False,temperature_mode_at_evaluation='enthalpy',best_validation=float(scheduler.best_model.numpy()),
        parent_epochs=base_epochs,parent_updates=base_updates,batch_size=cfg['batch_size'])
    atomic_json(out/'completed.json',record);atomic_json(out/'progress.json',record)


if __name__=='__main__':main()
