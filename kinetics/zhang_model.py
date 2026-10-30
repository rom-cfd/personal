"""Zhang 2022 Eq.(2) full transformed-state increments, physics disabled.

External state order is T,Y9,P; network order is T,P,BCT(Y9). There is no
enthalpy closure, projection, decoder clipping or post-decoding normalization.
"""
import json
from pathlib import Path
import numpy as np
from .study import tf
from .transforms import Standardizer


class ZhangPredictor:
    def __init__(self,spec,x,y,config,seed):
        self.spec=spec;self.gamma=float(spec['gamma'])
        encoded=self.encode(x)
        next_full=np.c_[y,x[:,-1]]
        delta=self.encode(next_full)-encoded
        self.input_norm=Standardizer().fit(encoded)
        self.output_norm=Standardizer().fit(delta)
        self.train_features=self.input_norm.forward(encoded).astype('float32')
        self.train_target=self.output_norm.forward(delta).astype('float32')
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
        z=tf.keras.Input((11,));h=z
        for width in config['hidden_layers']:h=tf.keras.layers.Dense(width,activation=config['activation'])(h)
        output=tf.keras.layers.Dense(11,kernel_initializer='zeros',bias_initializer='zeros')(h)
        self.model=tf.keras.Model(z,output)

    def encode(self,x):
        x=np.asarray(x)
        if not np.isfinite(x).all() or np.any(x[:,1:10]<0):
            raise ValueError('nonfinite state or negative BCT input')
        # Literal BCT, not BCT(delta Y); pressure is LINEAR, not log(P).
        return np.c_[x[:,0],x[:,-1]/101325.,(x[:,1:10]**self.gamma-1)/self.gamma]

    def features(self,x):return self.input_norm.forward(self.encode(x))

    def decode(self,standardized,x):
        next_encoded=self.encode(x)+self.output_norm.inverse(standardized)
        base=1+self.gamma*next_encoded[...,2:]
        with np.errstate(invalid='ignore',over='ignore',divide='ignore'):
            y=np.where(base>=0,base**(1/self.gamma),np.nan)
        return np.concatenate([next_encoded[...,:1],y,next_encoded[...,1:2]*101325.],axis=-1)

    def predict_full(self,x):
        delta=self.model(self.features(x).astype('float32'),training=False).numpy().astype('float64')
        return self.decode(delta,x)

    def predict(self,x):return self.predict_full(x)[:,:10]

    def save(self,path):
        self.model.save(Path(str(path)+'.keras'))
        Path(str(path)+'.json').write_text(json.dumps(dict(model_type='zhang_full_transformed_increment_raw',spec=self.spec,
            input_mean=self.input_norm.mean.tolist(),input_scale=self.input_norm.scale.tolist(),
            output_mean=self.output_norm.mean.tolist(),output_scale=self.output_norm.scale.tolist(),physics=False,
            input_order='T_K,P_atm,BCT_Y9',output_order='delta_T_K,delta_P_atm,delta_BCT_Y9'),indent=2))

    @classmethod
    def load(cls,path):
        info=json.loads(Path(str(path)+'.json').read_text())
        assert info['model_type']=='zhang_full_transformed_increment_raw' and info['physics'] is False
        self=cls.__new__(cls);self.spec=info['spec'];self.gamma=float(self.spec['gamma'])
        self.input_norm=Standardizer(np.array(info['input_mean']),np.array(info['input_scale']))
        self.output_norm=Standardizer(np.array(info['output_mean']),np.array(info['output_scale']))
        self.model=tf.keras.models.load_model(Path(str(path)+'.keras'),compile=False)
        return self


class ZhangCommittee:
    def __init__(self,paths,seeds,metric,rule='mean'):
        self.paths=list(map(str,paths));self.seeds=np.asarray(seeds);self.metric=metric;self.rule=rule
        self.members=[ZhangPredictor.load(p) for p in paths];self.base=self.members[0]
        for p in self.members[1:]:
            assert p.gamma==self.base.gamma
            for a,b in [(p.input_norm.mean,self.base.input_norm.mean),(p.input_norm.scale,self.base.input_norm.scale),
                        (p.output_norm.mean,self.base.output_norm.mean),(p.output_norm.scale,self.base.output_norm.scale)]:
                np.testing.assert_array_equal(a,b)
        @tf.function(input_signature=[tf.TensorSpec([None,11],tf.float32)])
        def forward(x):return tf.stack([p.model(x,training=False) for p in self.members])
        self.forward=forward;self.last_diagnostics={}

    def candidates(self,x):
        delta=self.forward(self.base.features(x).astype('float32')).numpy().astype('float64')
        return self.base.decode(delta,x)

    def predict_full(self,x):
        from .committee import choose_medoid
        proposals=self.candidates(x);finite=np.isfinite(proposals).all(axis=-1)
        if self.rule=='single':
            out=proposals[0];choice=np.where(finite[0],0,-1)
        elif self.rule=='mean':
            count=finite.sum(axis=0)
            out=np.where(finite[...,None],proposals,0).sum(axis=0)/np.maximum(count,1)[:,None]
            out[count==0]=np.nan;choice=np.full(len(x),-1)
        elif self.rule=='medoid':
            # Select on the existing train-scaled T/Y distance; carry that member's P.
            c=proposals[:,:,:10].copy();c[~finite]=np.nan
            _,choice,_=choose_medoid(c,self.metric,'scaled')
            out=np.full((len(x),11),np.nan);live=np.flatnonzero(choice>=0)
            out[live]=proposals[choice[live],live]
        else:raise ValueError('unknown rule')
        seed=np.full(len(x),-1);live=np.flatnonzero(choice>=0)
        seed[live]=self.seeds[choice[live]]
        self.last_diagnostics=dict(selected_seed=seed,valid_count=finite.sum(axis=0))
        return out

    def predict(self,x):return self.predict_full(x)[:,:10]

    def save(self,path):
        Path(path).write_text(json.dumps(dict(paths=self.paths,seeds=self.seeds.tolist(),metric=self.metric,rule=self.rule),indent=2))

    @classmethod
    def load(cls,path):return cls(**json.loads(Path(path).read_text()))
