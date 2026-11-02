"""Sequential, state-preserving DeepONet continuation and train-only plateau checks.

The existing P-only trainer and checkpoints remain immutable. New helpers group
optimizer updates inside one serial TensorFlow loop; members are never concurrent.
"""
import json
from pathlib import Path
import time

import numpy as np
from .deeponet import ThermoDeepONet
from .p_only_training import PressureTrainer, signature, sha256, learning_rate
from .study import tf


def restore_trainer(directory, x, y, config, seed, family):
    if family == 'p_only':
        return PressureTrainer.restore(directory,x,y,config,seed)
    if family != 'tp':
        raise ValueError('Unknown model family')
    directory = Path(directory)
    marker = json.loads((directory/'complete.json').read_text())
    required = {'model.keras','model.json','adam.npz','state.json','history.csv','full_train_history.csv'}
    if set(marker['output_sha256']) != required:
        raise ValueError('Incomplete checkpoint')
    for name,expected in marker['output_sha256'].items():
        if Path(name).name != name or sha256(directory/name) != expected:
            raise ValueError('Checkpoint hash mismatch')
    state = json.loads((directory/'state.json').read_text())
    if state['format_version'] != 1 or state['seed'] != seed or state['signature'] != signature(x,y,config):
        raise ValueError('Checkpoint provenance mismatch')
    predictor = ThermoDeepONet.load(directory/'model')
    trainer = PressureTrainer(x,y,config,seed,predictor=predictor)
    variables = trainer.optimizer.variables
    layout = [dict(name=v.name,shape=list(v.shape)) for v in variables]
    if layout != state['optimizer_variables']:
        raise ValueError('Adam variable layout mismatch')
    with np.load(directory/'adam.npz',allow_pickle=False) as data:
        if len(data.files) != len(variables):
            raise ValueError('Incomplete Adam state')
        for i,v in enumerate(variables):
            v.assign(data[f'v{i}'])
    if int(trainer.optimizer.iterations.numpy()) != state['update'] or state['optimizer_iteration'] != state['update']:
        raise ValueError('Adam iteration/update mismatch')
    trainer.rng.bit_generator.state = state['rng_state']
    trainer.tf_rng = tf.random.Generator.from_state(state['tf_rng_state'],state['tf_rng_algorithm'])
    trainer.update,trainer.carried_seconds = state['update'],state['elapsed_seconds']
    trainer.history,trainer.full_train_history = state['history'],state['full_train_history']
    trainer.plateau_monitor = state['plateau_monitor']
    return trainer


def fresh_tp(x,y,config,seed):
    predictor = ThermoDeepONet(dict(kind='boxcox',gamma=.1),x,y,config,seed)
    return PressureTrainer(x,y,config,seed,predictor=predictor)


def new_monitor(policy, origin_update=0, origin_seconds=0.):
    return dict(enabled=True,policy=policy,checks=[],passing_since=None,plateau=False,
                origin_update=origin_update,origin_seconds=origin_seconds,
                timing=dict(rng_seconds=0.,optimizer_seconds=0.,full_train_seconds=0.,
                            trace_seconds=0.,checkpoint_seconds=0.))


def check_plateau(history,update,monitor):
    """Two disjoint observed windows, quantiles, and consecutive sustained checks."""
    policy = monitor['policy']
    if update < policy['minimum_updates'] or update % policy['check_every']:
        return False
    if monitor['checks'] and monitor['checks'][-1]['update'] == update:
        return monitor['plateau']
    width = policy['window_updates']
    windows = [[r['full_train_loss'] for r in history
                if lo < r['update'] <= hi] for lo,hi in
               [(update-2*width,update-width),(update-width,update)]]
    expected = width//policy['observe_every']
    complete = all(len(w)==expected and np.isfinite(w).all() and np.all(np.asarray(w)>=0)
                   for w in windows)
    row = dict(update=update,window_counts=[len(w) for w in windows],passed=False,
               old_quantiles=None,new_quantiles=None,relative_changes=None)
    if complete:
        old,new = [np.quantile(w,[.25,.5,.75]) for w in windows]
        changes = [abs(float(b/a)-1) if a>0 else (0. if b==0 else None) for a,b in zip(old,new)]
        row.update(old_quantiles=old.tolist(),new_quantiles=new.tolist(),relative_changes=changes,
            passed=all(d is not None and d<=limit for d,limit in zip(changes,policy['relative_thresholds'])))
    previous = monitor['checks'][-1]['update'] if monitor['checks'] else None
    if row['passed']:
        if monitor['passing_since'] is None or previous != update-policy['check_every']:
            monitor['passing_since'] = update
        monitor['plateau'] = update-monitor['passing_since'] >= policy['sustain_updates']
    else:
        monitor['passing_since'],monitor['plateau'] = None,False
    monitor['checks'].append(row)
    return monitor['plateau']


class SerialSteps:
    """One member, serial Adam updates; preserve the exact NumPy minibatch stream."""
    def __init__(self,trainer):
        self.trainer = trainer
        p,opt = trainer.predictor,trainer.optimizer
        features,targets = trainer.features,trainer.targets

        @tf.function(input_signature=[tf.TensorSpec([None,None],tf.int32),
                                      tf.TensorSpec([None],tf.float32)])
        def block(ids,rates):
            losses = tf.TensorArray(tf.float32,size=tf.shape(ids)[0])
            def body(i,losses):
                opt.learning_rate.assign(rates[i])
                with tf.GradientTape() as tape:
                    error = p.model(tf.gather(features,ids[i]),training=True)-tf.gather(targets,ids[i])
                    loss = tf.debugging.check_numerics(tf.reduce_mean(error**2),'nonfinite training loss')
                variables = p.model.trainable_variables
                grads,_ = tf.clip_by_global_norm(tape.gradient(loss,variables),5.)
                opt.apply_gradients(zip(grads,variables))
                return i+1,losses.write(i,loss)
            _,losses = tf.while_loop(lambda i,_: i<tf.shape(ids)[0],body,(0,losses),parallel_iterations=1)
            return losses.stack()

        @tf.function(input_signature=[tf.TensorSpec([None,11],tf.float32)])
        def forward(features):
            return p.model(features,training=False)
        self.block,self.forward = block,forward
        started = time.perf_counter()
        block.get_concrete_function()
        forward.get_concrete_function()
        self.timing['trace_seconds'] += time.perf_counter()-started

    @property
    def timing(self):
        return self.trainer.plateau_monitor['timing']

    def advance(self,target,deadline=float('inf')):
        t = self.trainer
        if target<t.update:
            raise ValueError('Update cannot move backwards')
        if time.perf_counter() >= deadline:
            return 'time_limit'
        if target == t.update:
            return 'target_update'
        started = time.perf_counter()
        count = target-t.update
        # Fixed even batch512 draws produce the same stream as scalar draws.
        ids = t.rng.integers(0,len(t.features),(count,t.config['batch_size']),dtype=np.int32)
        updates = np.arange(t.update+1,target+1)
        rates = np.array([learning_rate(int(u)) for u in updates],dtype=np.float32)
        self.timing['rng_seconds'] += time.perf_counter()-started
        started = time.perf_counter()
        losses = self.block(ids,rates).numpy()
        self.timing['optimizer_seconds'] += time.perf_counter()-started
        if int(t.optimizer.iterations.numpy()) != target:
            raise ValueError('Serial block optimizer iteration mismatch')
        for u,loss in zip(updates,losses):
            if u==1 or u%50==0 or u in (6001,9001):
                t.history.append(dict(seed=t.seed,update=int(u),loss=float(loss),learning_rate=learning_rate(int(u))))
        t.update = target
        return 'target_update'

    def full_train_loss(self,deadline=float('inf')):
        t = self.trainer
        started = time.perf_counter()
        squared,count = 0.,0
        for start in range(0,len(t.features),512):
            if time.perf_counter() >= deadline:
                self.timing['full_train_seconds'] += time.perf_counter()-started
                return None
            output = self.forward(t.features[start:start+512]).numpy().astype('float64')
            error = output-t.full_targets[start:start+512]
            squared += float(np.sum(error**2));count += error.size
        loss = squared/count
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite full train loss')
        t.full_train_history.append(dict(seed=t.seed,update=t.update,full_train_loss=loss))
        self.timing['full_train_seconds'] += time.perf_counter()-started
        return loss
