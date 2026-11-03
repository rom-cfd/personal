"""Fixed-update serial training with a physical element-L1 soft penalty.

The loss follows the uploaded methane reconstruction example, adapted to this
fixed2us residual-state problem. Preprocessing and inference remain unchanged.
"""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from .study import Predictor,tf
from .deeponet import ThermoDeepONet
from .p_only_deeponet import PressureDeepONet
from .tp_phi_deeponet import PhiDeepONet
from .p_only_training import sha256,write_json,signature,learning_rate

FAMILIES=('ann','p_only','tp','tp_phi')
LOADERS=dict(ann=Predictor,p_only=PressureDeepONet,tp=ThermoDeepONet,tp_phi=PhiDeepONet)


def make_predictor(family,x,y,config,seed):
    if family not in FAMILIES:raise ValueError('Unknown family')
    return LOADERS[family](dict(kind='boxcox',gamma=.1),x,y,config,seed)


def soft_signature(x,y,config,family,matrix,mu):
    result=signature(x,y,config)
    result.update(family=family,mu=float(mu),element_matrix=np.asarray(matrix).tolist(),
        soft_source_sha256={str(p):sha256(p) for p in sorted(Path('kinetics').glob('soft_constraint_*.py'))},
        phi_source_sha256={str(p):sha256(p) for p in sorted(Path('kinetics').glob('tp_phi_*.py'))})
    return result


def signed_boxcox_inverse(z,gamma=.1):
    """Exact original inverse for z>=0; finite signed extension for loss only.

The methane rate example also uses a signed power inverse. Clipping would hide
element violations and zero gradients. Strict inference still rejects z<0.
"""
    z=tf.cast(z,tf.float64)
    return tf.sign(z)*tf.pow(tf.constant(gamma,tf.float64)*tf.abs(z),1./gamma)


class SoftObjective:
    def __init__(self,predictor,matrix,mu):
        if not np.isfinite(mu) or mu<0:raise ValueError('Nonnegative finite mu required')
        self.mu=float(mu)
        self.mean=tf.constant(predictor.norm.mean[1:],tf.float64)
        self.scale=tf.constant(predictor.norm.scale[1:],tf.float64)
        self.matrix=tf.constant(np.asarray(matrix),tf.float64)
        if self.matrix.shape!=(3,9):raise ValueError('Expected H/O/N x9 species matrix')

    def components(self,prediction,target,current_standardized,true_species):
        data=tf.reduce_mean(tf.square(prediction-target))
        z=((tf.cast(current_standardized[:,1:],tf.float64)+tf.cast(prediction[:,1:],tf.float64))
           *self.scale+self.mean)
        species=signed_boxcox_inverse(z)
        difference=tf.linalg.matmul(species-tf.cast(true_species,tf.float64),self.matrix,transpose_b=True)
        element=tf.reduce_mean(tf.reduce_sum(tf.abs(difference),axis=1))
        total=tf.cast(data,tf.float64)+self.mu*element
        # mu0 regression exactly follows the original MSE update graph.
        objective=data if self.mu==0 else total
        return objective,tf.stack([total,tf.cast(data,tf.float64),element])


class SoftTrainer:
    def __init__(self,x,y,config,seed,family,matrix,mu,predictor=None):
        self.started=time.perf_counter();self.carried_seconds=0.
        self.config=dict(config);self.seed=int(seed);self.family=family
        self.mu=float(mu);self.matrix=np.asarray(matrix)
        self.predictor=predictor if predictor is not None else make_predictor(family,x,y,config,seed)
        p=self.predictor
        self.signature=soft_signature(x,y,config,family,matrix,mu)
        a,b=p.encode_state(x[:,:10]),p.encode_state(y)
        self.current_standardized=tf.constant(p.norm.forward(a),tf.float64)
        self.features=tf.constant(p.features(x).astype('float32'))
        self.targets=tf.constant((p.norm.forward(b)-p.norm.forward(a)).astype('float32'))
        self.true_species=tf.constant(y[:,1:],tf.float64)
        self.optimizer=tf.keras.optimizers.Adam(.001);self.optimizer.build(p.model.trainable_variables)
        self.rng=np.random.default_rng(seed);self.tf_rng=tf.random.Generator.from_seed(seed)
        self.objective=SoftObjective(p,matrix,mu)
        self.update=0;self.history=[];self.full_train_history=[]
        self.timing=dict(trace_seconds=0.,rng_seconds=0.,optimizer_seconds=0.,
                         full_train_seconds=0.,checkpoint_seconds=0.)
        self.plateau_monitor=dict(enabled=False,reason='User specified fixed10000 updates')
        self._build_steps()

    def _build_steps(self):
        @tf.function(input_signature=[tf.TensorSpec([None,None],tf.int32),tf.TensorSpec([None],tf.float32)])
        def block(ids,rates):
            losses=tf.TensorArray(tf.float64,size=tf.shape(ids)[0])
            def body(i,losses):
                self.optimizer.learning_rate.assign(rates[i])
                with tf.GradientTape() as tape:
                    prediction=self.predictor.model(tf.gather(self.features,ids[i]),training=True)
                    objective,components=self.objective.components(prediction,tf.gather(self.targets,ids[i]),
                        tf.gather(self.current_standardized,ids[i]),tf.gather(self.true_species,ids[i]))
                    objective=tf.debugging.check_numerics(objective,'Nonfinite soft objective')
                variables=self.predictor.model.trainable_variables
                gradients=tape.gradient(objective,variables)
                for g in gradients:tf.debugging.assert_all_finite(g,'Nonfinite soft gradient')
                gradients,_=tf.clip_by_global_norm(gradients,5.)
                self.optimizer.apply_gradients(zip(gradients,variables))
                return i+1,losses.write(i,components)
            _,losses=tf.while_loop(lambda i,_:i<tf.shape(ids)[0],body,(0,losses),parallel_iterations=1)
            return losses.stack()
        width=int(self.features.shape[1])
        @tf.function(input_signature=[tf.TensorSpec([None,width],tf.float32)])
        def forward(features):return self.predictor.model(features,training=False)
        started=time.perf_counter();block.get_concrete_function();forward.get_concrete_function()
        self.timing['trace_seconds']+=time.perf_counter()-started
        self.block,self.forward=block,forward

    @property
    def elapsed_seconds(self):return self.carried_seconds+time.perf_counter()-self.started

    def advance(self,target,deadline=float('inf')):
        if target<self.update:raise ValueError('Updates cannot go backwards')
        if time.perf_counter()>=deadline:return 'time_limit'
        if target==self.update:return 'target_update'
        started=time.perf_counter();updates=np.arange(self.update+1,target+1)
        ids=self.rng.integers(0,len(self.features),(len(updates),self.config['batch_size']),dtype=np.int32)
        rates=np.array([learning_rate(int(u)) for u in updates],dtype=np.float32)
        self.timing['rng_seconds']+=time.perf_counter()-started
        started=time.perf_counter();values=self.block(ids,rates).numpy()
        self.timing['optimizer_seconds']+=time.perf_counter()-started
        self.update=target
        if int(self.optimizer.iterations.numpy())!=self.update:raise ValueError('Iteration mismatch')
        for u,(total,data,element) in zip(updates,values):
            if u==1 or u%50==0 or u in (6001,9001):
                self.history.append(dict(seed=self.seed,update=int(u),loss=float(total),
                    data_loss=float(data),element_loss=float(element),learning_rate=learning_rate(int(u))))
        return 'target_update'

    def full_train_loss(self,deadline=float('inf')):
        started=time.perf_counter();data_sum=element_sum=0.;invalid=negative=0;n=len(self.features)
        for start in range(0,n,512):
            if time.perf_counter()>=deadline:return None
            end=min(start+512,n);prediction=self.forward(self.features[start:end])
            _,parts=self.objective.components(prediction,self.targets[start:end],
                self.current_standardized[start:end],self.true_species[start:end])
            data_sum+=float(parts[1].numpy())*(end-start)
            element_sum+=float(parts[2].numpy())*(end-start)
            p=self.predictor
            z=p.norm.inverse(self.current_standardized[start:end].numpy().astype('float64')+
                             prediction.numpy().astype('float64'))[:,1:]
            invalid+=int(np.any(z<0,axis=1).sum());negative+=int((z<0).sum())
        data,element=data_sum/n,element_sum/n
        row=dict(seed=self.seed,update=self.update,full_train_loss=data+self.mu*element,
            data_loss=data,element_loss=element,invalid_inverse_row_fraction=invalid/n,
            negative_encoded_fraction=negative/(n*9))
        self.full_train_history.append(row)
        self.timing['full_train_seconds']+=time.perf_counter()-started
        return row

    def save_checkpoint(self,directory,reason):
        started=time.perf_counter();directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        self.predictor.save(directory/'model')
        variables=self.optimizer.variables
        np.savez_compressed(directory/'adam.npz',**{f'v{i}':v.numpy() for i,v in enumerate(variables)})
        state=dict(format_version=2,family=self.family,seed=self.seed,update=self.update,
            signature=self.signature,optimizer_iteration=int(self.optimizer.iterations.numpy()),
            optimizer_variables=[dict(name=v.name,shape=list(v.shape)) for v in variables],
            learning_rate=float(self.optimizer.learning_rate.numpy()),rng_state=self.rng.bit_generator.state,
            tf_rng_state=self.tf_rng.state.numpy().tolist(),tf_rng_algorithm=int(self.tf_rng.algorithm),
            history=self.history,full_train_history=self.full_train_history,
            plateau_monitor=self.plateau_monitor,timing=self.timing,
            elapsed_seconds=self.elapsed_seconds,reason=reason)
        write_json(directory/'state.json',state)
        pd.DataFrame(self.history).to_csv(directory/'history.csv',index=False)
        pd.DataFrame(self.full_train_history).to_csv(directory/'full_train_history.csv',index=False)
        write_json(directory/'complete.json',dict(output_sha256={p.name:sha256(p) for p in directory.iterdir() if p.is_file()}))
        self.timing['checkpoint_seconds']+=time.perf_counter()-started
        return directory

    @classmethod
    def restore(cls,directory,x,y,config,seed,family,matrix,mu):
        directory=Path(directory);marker=json.loads((directory/'complete.json').read_text())
        required={'model.keras','model.json','adam.npz','state.json','history.csv','full_train_history.csv'}
        if set(marker['output_sha256'])!=required:raise ValueError('Incomplete soft checkpoint')
        for name,h in marker['output_sha256'].items():
            if Path(name).name!=name or sha256(directory/name)!=h:raise ValueError('Soft checkpoint hash mismatch')
        state=json.loads((directory/'state.json').read_text())
        if state['format_version']!=2 or state['signature']!=soft_signature(x,y,config,family,matrix,mu):
            raise ValueError('Soft checkpoint provenance mismatch')
        self=cls(x,y,config,seed,family,matrix,mu,predictor=LOADERS[family].load(directory/'model'))
        if [dict(name=v.name,shape=list(v.shape)) for v in self.optimizer.variables]!=state['optimizer_variables']:
            raise ValueError('Adam variable layout changed')
        with np.load(directory/'adam.npz',allow_pickle=False) as saved:
            if set(saved.files)!={f'v{i}' for i in range(len(self.optimizer.variables))}:
                raise ValueError('Missing Adam variables')
            for i,v in enumerate(self.optimizer.variables):v.assign(saved[f'v{i}'])
        if int(self.optimizer.iterations.numpy())!=state['update'] or state['optimizer_iteration']!=state['update']:
            raise ValueError('Adam iteration mismatch')
        self.rng.bit_generator.state=state['rng_state']
        self.tf_rng=tf.random.Generator.from_state(state['tf_rng_state'],state['tf_rng_algorithm'])
        self.update,self.carried_seconds=state['update'],state['elapsed_seconds']
        self.history,self.full_train_history=state['history'],state['full_train_history']
        restore_trace=self.timing['trace_seconds']
        self.timing=state['timing'];self.timing['trace_seconds']+=restore_trace
        self.plateau_monitor=state['plateau_monitor']
        return self
