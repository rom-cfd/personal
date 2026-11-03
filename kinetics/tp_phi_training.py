"""Independent T/P/phi Adam/RNG checkpoints and serial optimizer updates."""
import json
from pathlib import Path
import numpy as np
import time
from .p_only_training import PressureTrainer, signature, sha256, learning_rate
from .deeponet_long_training import new_monitor, check_plateau
from .tp_phi_deeponet import PhiDeepONet
from .study import tf


def phi_signature(x,y,config):
    result=signature(x,y,config)
    result['phi_source_sha256']={str(p):sha256(p) for p in sorted(Path('kinetics').glob('tp_phi_*.py'))}
    return result


class PhiTrainer(PressureTrainer):
    def __init__(self,x,y,config,seed,predictor=None):
        started=time.perf_counter()
        p=predictor if predictor is not None else PhiDeepONet(dict(kind='boxcox',gamma=.1),x,y,config,seed)
        super().__init__(x,y,config,seed,predictor=p)
        self.started=started
        self.signature=phi_signature(x,y,config)

    @classmethod
    def restore(cls,directory,x,y,config,seed):
        directory=Path(directory)
        marker=json.loads((directory/'complete.json').read_text())
        required={'model.keras','model.json','adam.npz','state.json','history.csv','full_train_history.csv'}
        if set(marker['output_sha256'])!=required:
            raise ValueError('Incomplete T/P/phi checkpoint')
        for name,expected in marker['output_sha256'].items():
            if Path(name).name!=name or sha256(directory/name)!=expected:
                raise ValueError('Checkpoint hash mismatch')
        state=json.loads((directory/'state.json').read_text())
        if (state['format_version']!=1 or state['seed']!=seed
                or state['signature']!=phi_signature(x,y,config)):
            raise ValueError('T/P/phi checkpoint provenance mismatch')
        trainer=cls(x,y,config,seed,predictor=PhiDeepONet.load(directory/'model'))
        layout=[dict(name=v.name,shape=list(v.shape)) for v in trainer.optimizer.variables]
        if layout!=state['optimizer_variables']:
            raise ValueError('Adam layout changed')
        with np.load(directory/'adam.npz',allow_pickle=False) as saved:
            if set(saved.files)!={f'v{i}' for i in range(len(layout))}:
                raise ValueError('Incomplete Adam variables')
            for i,v in enumerate(trainer.optimizer.variables):v.assign(saved[f'v{i}'])
        if int(trainer.optimizer.iterations.numpy())!=state['update'] or state['optimizer_iteration']!=state['update']:
            raise ValueError('Optimizer iteration/update mismatch')
        trainer.rng.bit_generator.state=state['rng_state']
        trainer.tf_rng=tf.random.Generator.from_state(state['tf_rng_state'],state['tf_rng_algorithm'])
        trainer.update,trainer.carried_seconds=state['update'],state['elapsed_seconds']
        trainer.history,trainer.full_train_history=state['history'],state['full_train_history']
        trainer.plateau_monitor=state['plateau_monitor']
        return trainer


class PhiSerialSteps:
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

        @tf.function(input_signature=[tf.TensorSpec([None,12],tf.float32)])
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
