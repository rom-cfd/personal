"""Independent fixed-budget Adam training in groups of same-shape DNNs.

Each member keeps its own initialization, minibatch RNG, Adam moments and global
norm clipping. Losses SUM over members, so group size does not scale gradients.
This is computational batching, not parameter sharing or a changed architecture.
"""
import time
import numpy as np
from .study import Predictor,tf


def clip_member_gradients(grads,clip=5.):
    norms=tf.sqrt(tf.add_n([tf.reduce_sum(g*g,axis=tuple(range(1,len(g.shape)))) for g in grads]))
    factor=tf.minimum(tf.ones_like(norms),clip/tf.maximum(norms,1e-30))
    return [g*tf.reshape(factor,[-1]+[1]*(len(g.shape)-1)) for g in grads]


def train_group(spec,seeds,updates,x,y,cfg,deadline=float('inf'),progress=False):
    if cfg['activation']!='tanh':raise ValueError('Grouped trainer currently supports tanh only')
    started=time.perf_counter();members=[Predictor(spec,x,y,cfg,seed) for seed in seeds]
    first=members[0]
    for member in members[1:]:
        np.testing.assert_array_equal(first.train_features,member.train_features)
        np.testing.assert_array_equal(first.train_target,member.train_target)
    features=tf.constant(first.train_features);target=tf.constant(first.train_target)
    params=[tf.Variable(np.stack([p.model.get_weights()[i] for p in members]),name=f'group_parameter_{i}')
            for i in range(len(first.model.get_weights()))]
    optimizer=tf.keras.optimizers.Adam(cfg['learning_rate'])
    @tf.function(input_signature=[tf.TensorSpec([len(seeds),None],tf.int32),tf.TensorSpec([],tf.float32)])
    def step(ids,lr):
        optimizer.learning_rate.assign(lr)
        with tf.GradientTape() as tape:
            h=tf.gather(features,ids)
            for layer in range(len(params)//2):
                h=tf.matmul(h,params[2*layer])+params[2*layer+1][:,None,:]
                if layer<len(params)//2-1:h=tf.math.tanh(h)
            losses=tf.reduce_mean((h-tf.gather(target,ids))**2,axis=(1,2))
            objective=tf.reduce_sum(losses)
        grads=clip_member_gradients(tape.gradient(objective,params))
        optimizer.apply_gradients(zip(grads,params))
        return losses
    rngs=[np.random.default_rng(seed) for seed in seeds];history=[]
    for k in range(updates):
        if k%100==0 and time.perf_counter()>deadline:raise TimeoutError('Grouped training budget exhausted')
        lr=cfg['learning_rate']*(1 if k<updates*.6 else .3 if k<updates*.9 else .1)
        ids=np.stack([r.integers(0,len(x),cfg['batch_size'],dtype=np.int32) for r in rngs])
        loss=step(ids,np.float32(lr))
        if (k+1)%500==0 or k==updates-1:
            values=loss.numpy()
            history.extend(dict(seed=int(seed),update=k+1,loss=float(value),learning_rate=lr) for seed,value in zip(seeds,values))
            if progress:print('GROUP_PROGRESS',seeds,'update',k+1,'seconds',round(time.perf_counter()-started,2),flush=True)
    arrays=[v.numpy() for v in params]
    for i,member in enumerate(members):member.model.set_weights([a[i] for a in arrays])
    return members,time.perf_counter()-started,history
