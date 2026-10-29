"""TensorFlow implementation of Wang 2025's stated ANN-hard equations.

Paper-defined: species next-G target, gamma=.1, 1600/800/400 GELU, G-space L1,
two-stage Adam and element/energy postprocessing. Explicit assumptions: 9 output
species per Eq.(2) despite the text's contradictory 11-neuron statement; input
train-only standardization; PyTorch-default Linear initialization distribution,
exact GELU and bias-corrected Adam with epsilon outside sqrt(v_hat).

This is not a claim to possess the authors' unpublished 2025 training code.
"""
import json
from pathlib import Path
import numpy as np
from .study import tf
from .transforms import Standardizer
from .data import mechanism
from .physics import element_matrix


def boxcox(values):
    y=np.asarray(values,dtype=np.float64)
    if np.any(~np.isfinite(y)) or np.any(y<0):
        raise ValueError('Nonnegative finite species required')
    return (y**.1-1)/.1


def inverse_boxcox(values):
    g=np.asarray(values,dtype=np.float64);base=1+.1*g
    with np.errstate(invalid='ignore',over='ignore'):
        return np.where(base>=0,base**10,np.nan)


def paper_projection(proposal,current,matrix):
    """Eqs.(5)-(9), direct full-rank solve, no floor/clipping/positivity QP."""
    v=np.asarray(proposal,dtype=float);current=np.asarray(current,dtype=float)
    if not np.all(np.isfinite(v)) or np.any(v<0):
        raise ValueError('Invalid proposed species')
    d=v**.9;b=matrix*d
    correction_g=b.T@np.linalg.solve(b@b.T,matrix@(current-v))
    return v+d*correction_g


class BiasCorrectedAdam:
    """PyTorch-default Adam equations implemented in TensorFlow, no grad clip.

    Keras Adam epsilon-hat placement differs; explicitly divide by
    sqrt(v/(1-beta2**t))+epsilon to match the stated PyTorch default equations.
    Seeded random streams remain backend-specific, not bit-identical.
    """
    def __init__(self,variables,beta1=.9,beta2=.999,epsilon=1e-8):
        self.beta1=beta1;self.beta2=beta2;self.epsilon=epsilon
        self.t=tf.Variable(0,dtype=tf.int64,trainable=False)
        self.m=[tf.Variable(tf.zeros_like(v),trainable=False) for v in variables]
        self.v=[tf.Variable(tf.zeros_like(v),trainable=False) for v in variables]

    def apply(self,gradients,variables,learning_rate):
        self.t.assign_add(1)
        for g,parameter,m,v in zip(gradients,variables,self.m,self.v):
            if g is None:raise ValueError('Missing gradient')
            dtype=parameter.dtype
            b1=tf.cast(self.beta1,dtype);b2=tf.cast(self.beta2,dtype)
            m.assign(b1*m+(1-b1)*g);v.assign(b2*v+(1-b2)*g*g)
            t=tf.cast(self.t,dtype)
            corrected_m=m/(1-tf.pow(b1,t));corrected_v=v/(1-tf.pow(b2,t))
            parameter.assign_sub(tf.cast(learning_rate,dtype)*corrected_m/(tf.sqrt(corrected_v)+tf.cast(self.epsilon,dtype)))


def build_network(seed):
    """RandomUniform +/-1/sqrt(fan_in), matching Linear's default distribution."""
    tf.keras.utils.set_random_seed(seed)
    inputs=tf.keras.Input((11,));h=inputs;fan_in=11
    for i,width in enumerate([1600,800,400,9]):
        limit=1/np.sqrt(fan_in)
        h=tf.keras.layers.Dense(width,
            activation=tf.nn.gelu if i<3 else None,
            kernel_initializer=tf.keras.initializers.RandomUniform(-limit,limit,seed=seed+2*i),
            bias_initializer=tf.keras.initializers.RandomUniform(-limit,limit,seed=seed+2*i+1))(h)
        fan_in=width
    return tf.keras.Model(inputs,h)


class PaperPredictor:
    def __init__(self,training_x,seed=11):
        self.seed=seed
        self.input_norm=Standardizer().fit(self.encode_input(training_x))
        self.model=build_network(seed)
        self.gas=mechanism();self.matrix=element_matrix(self.gas)
        self.counts=dict(proposals=0,inverse_invalid=0,projection_invalid=0,thermal_invalid=0,accepted=0)

    @staticmethod
    def encode_input(x):
        # Raw linear T and P(atm), not log(P); standardization fit on train only.
        return np.c_[x[:,0],x[:,-1]/101325.,boxcox(x[:,1:10])]

    def features(self,x):
        return self.input_norm.forward(self.encode_input(x)).astype('float32')

    @staticmethod
    def targets(next_state):
        # No residual target, output standardization, temperature or pressure head.
        return boxcox(next_state[:,1:10]).astype('float32')

    def proposals(self,x):
        g=self.model(self.features(x),training=False).numpy().astype('float64')
        return inverse_boxcox(g)

    def predict(self,x):
        proposal=self.proposals(x);out=np.full((len(x),10),np.nan)
        for k,(current,v) in enumerate(zip(x,proposal)):
            self.counts['proposals']+=1
            if not np.all(np.isfinite(v)):
                self.counts['inverse_invalid']+=1;continue
            try:
                y=paper_projection(v,current[1:10],self.matrix)
                if np.any(y<0) or not np.all(np.isfinite(y)) or np.max(np.abs(self.matrix@(y-current[1:10])))>1e-10:
                    raise ValueError('Unusable hard-corrected candidate')
            except (ValueError,np.linalg.LinAlgError):
                self.counts['projection_invalid']+=1;continue
            try:
                self.gas.TPY=current[0],current[-1],current[1:10];h=self.gas.enthalpy_mass
                self.gas.HPY=h,current[-1],y
                if not np.isfinite(self.gas.T) or self.gas.T<=0:raise ValueError('Invalid recovered temperature')
                out[k]=np.r_[self.gas.T,y];self.counts['accepted']+=1
            except (ValueError,RuntimeError):self.counts['thermal_invalid']+=1
        return out

    def save(self,path):
        self.model.save(Path(str(path)+'.keras'))
        Path(str(path)+'.json').write_text(json.dumps(dict(
            family='wang2025_species_next_G',seed=self.seed,input_mean=self.input_norm.mean.tolist(),
            input_scale=self.input_norm.scale.tolist(),output='9 raw G_next values; no T/P or residual',
            assumptions='train-only input z-score ddof0; exact GELU; PyTorch-default initialization distribution and Adam equations',
            unmatched='framework/RNG; unpublished 2025 preprocessing details; paper 11-output inconsistency; user-retained data'),indent=2))

    @classmethod
    def load(cls,path):
        info=json.loads(Path(str(path)+'.json').read_text())
        if info['family']!='wang2025_species_next_G':raise ValueError('Wrong model family')
        self=cls.__new__(cls);self.seed=info['seed']
        self.input_norm=Standardizer(np.array(info['input_mean']),np.array(info['input_scale']))
        self.model=tf.keras.models.load_model(Path(str(path)+'.keras'),compile=False)
        self.gas=mechanism();self.matrix=element_matrix(self.gas)
        self.counts=dict(proposals=0,inverse_invalid=0,projection_invalid=0,thermal_invalid=0,accepted=0)
        return self
