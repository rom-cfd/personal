"""Homogeneous regression committee: mean or a physically valid medoid.

All members see the SAME current state. Medoid is an actual repaired candidate,
not a coordinate median. Inference never calls ReactorNet or reference states.
"""
import json,time
from pathlib import Path
import numpy as np
from .study import Predictor,tf
from .physics import PhysicalPredictor,project_species,element_matrix
from .data import mechanism


def fit_metric(training_states,floor=1e-8):
    """Distance scales are trained once; no validation/test fitting."""
    a=np.asarray(training_states)
    temp=max(float(np.std(a[:,0])),100.)
    concentration=np.maximum(np.quantile(a[:,1:],.95,axis=0),floor)
    eps=np.maximum(concentration*1e-3,1e-12)
    log_scale=np.maximum(np.std(np.log1p(a[:,1:]/eps),axis=0),1.)
    return dict(temperature_scale=temp,species_scale=concentration.tolist(),
                log_epsilon=eps.tolist(),log_scale=log_scale.tolist(),fit_split='train',floor=floor)


def coordinates(candidates,metric,kind):
    temp=candidates[...,0:1]/metric['temperature_scale']
    y=candidates[...,1:]
    if kind=='scaled':sp=y/np.asarray(metric['species_scale'])
    elif kind=='softlog':sp=np.log1p(np.maximum(y,0)/np.asarray(metric['log_epsilon']))/np.asarray(metric['log_scale'])
    else:raise ValueError('unknown distance metric')
    return np.concatenate([temp,sp],axis=-1)


def choose_medoid(candidates,metric,kind='scaled',batch_chunk=64):
    """candidates: (members,batch,T+9Y); NaN members excluded per state.

    Sum of Euclidean distances, not squared distances. Ties use member order.
    No viable candidate returns NaN and index -1; never an arbitrary state.
    """
    c=np.asarray(candidates)
    if batch_chunk is not None and c.shape[1]>batch_chunk:
        pieces=[choose_medoid(c[:,i:i+batch_chunk],metric,kind,None) for i in range(0,c.shape[1],batch_chunk)]
        return (np.concatenate([p[0] for p in pieces]),np.concatenate([p[1] for p in pieces]),
                {k:np.concatenate([p[2][k] for p in pieces]) for k in pieces[0][2]})
    valid=np.all(np.isfinite(c),axis=-1)
    z=coordinates(np.where(np.isfinite(c),c,0),metric,kind)
    delta=z[:,None,:,:]-z[None,:,:,:]
    distances=np.sqrt(np.mean(delta*delta,axis=-1))
    pairs=valid[:,None,:]&valid[None,:,:]
    distances=np.where(pairs,distances,0.)
    costs=distances.sum(axis=1)
    costs=np.where(valid,costs,np.inf)
    choice=np.argmin(costs,axis=0)
    count=valid.sum(axis=0);choice=np.where(count>0,choice,-1)
    out=np.full(c.shape[1:],np.nan)
    live=np.flatnonzero(count>0);out[live]=c[choice[live],live]
    spread=distances.sum(axis=(0,1))/np.maximum(count*(count-1),1)
    best_cost=np.min(costs,axis=0)/np.maximum(count-1,1)
    return out,choice,dict(valid_count=count,spread=spread,central_cost=best_cost)


class Committee:
    def __init__(self,paths,seeds,metric,rule='medoid',distance='scaled',collect_mean_diagnostics=True,physics_backend='cantera'):
        if not len(paths) or len(paths)!=len(seeds):raise ValueError('empty or mismatched committee')
        self.paths=[str(p) for p in paths];self.seeds=np.asarray(seeds);self.metric=metric;self.rule=rule;self.distance=distance
        self.collect_mean_diagnostics=collect_mean_diagnostics
        self.physics_backend=physics_backend
        if physics_backend=='vectorized_nasa':
            from .vector_physics import VectorThermo
            self.vector_physics=VectorThermo()
        elif physics_backend=='wang_boxcox':
            pass  # The correction exponent is taken from the loaded model below.
        elif physics_backend!='cantera':raise ValueError('unknown physical correction backend')
        self.members=[Predictor.load(p) for p in paths];self.base=self.members[0]
        if physics_backend=='wang_boxcox':
            from .wang_physics import WangThermo
            gamma = self.base.transform.gamma
            if self.base.spec['kind']!='boxcox' or not 0 < gamma <= 1:
                raise ValueError('Wang backend requires Box-Cox with 0 < gamma <= 1')
            self.vector_physics=WangThermo(gamma=gamma)
        for p in self.members[1:]:
            if p.spec['kind']!=self.base.spec['kind'] or p.transform.gamma!=self.base.transform.gamma:
                raise ValueError('homogeneous preprocessing required')
            for x,y in [(p.norm.mean,self.base.norm.mean),(p.norm.scale,self.base.norm.scale),
                        (p.input_norm.mean,self.base.input_norm.mean),(p.input_norm.scale,self.base.input_norm.scale),
                        (p.transform.epsilon,self.base.transform.epsilon)]:
                np.testing.assert_array_equal(x,y)
        self.gas=mechanism();self.matrix=element_matrix(self.gas)
        self.timing=dict(neural_decode=0.,candidate_repair=0.,selection=0.,mean_repair=0.)
        self.last_diagnostics={}
        @tf.function(input_signature=[tf.TensorSpec([None,11],tf.float32)])
        def forward(features):
            return tf.stack([p.model(features,training=False) for p in self.members],axis=0)
        self.forward=forward

    def proposals(self,x,guard=True):
        base=self.base
        delta=self.forward(base.features(x).astype('float32')).numpy().astype('float64')
        z=base.norm.inverse(base.norm.forward(base.encode_state(x[:,:10]))[None,...]+delta)
        ys=z[...,1:]
        if guard:ys=np.clip(ys,0,base.transform.forward(np.ones((1,9))))
        decoded=base.transform.inverse(ys)
        return np.concatenate([z[...,0:1]*1000,decoded],axis=-1)

    def repair(self,proposals,x):
        """Common element/enthalpy correction; reject thermodynamically infeasible states."""
        if self.physics_backend in ('vectorized_nasa','wang_boxcox'):return self.vector_physics.repair(proposals,x)
        out=np.full_like(proposals,np.nan)
        for b,a in enumerate(x):
            self.gas.TPY=a[0],a[-1],a[1:10];h=self.gas.enthalpy_mass
            for j,p in enumerate(proposals[:,b]):
                try:
                    if not np.all(np.isfinite(p)):continue
                    y=project_species(p[1:],a[1:10],self.matrix)
                    self.gas.TP=a[0],a[-1]
                    self.gas.HPY=h,a[-1],y
                    if not 200<=self.gas.T<=6000:continue
                    out[j,b]=np.r_[self.gas.T,y]
                except (ValueError,RuntimeError):continue
        return out

    def predict(self,x):
        start=time.perf_counter();raw=self.proposals(x,guard=True)
        self.timing['neural_decode']+=time.perf_counter()-start
        if self.rule=='mean_before_repair':
            start=time.perf_counter();out=self.repair(np.mean(raw,axis=0)[None,...],x)[0]
            self.timing['mean_repair']+=time.perf_counter()-start
            self.last_diagnostics=dict(selected_seed=np.full(len(x),-1),valid_count=np.full(len(x),-1),spread=np.full(len(x),np.nan),central_cost=np.full(len(x),np.nan))
            return out
        start=time.perf_counter();candidates=self.repair(raw,x)
        self.timing['candidate_repair']+=time.perf_counter()-start
        if self.rule=='mean_after_repair' and not self.collect_mean_diagnostics:
            center=None;choice=np.full(len(x),-1)
            stats=dict(valid_count=np.all(np.isfinite(candidates),axis=-1).sum(axis=0),spread=np.full(len(x),np.nan),central_cost=np.full(len(x),np.nan))
        else:
            start=time.perf_counter();center,choice,stats=choose_medoid(candidates,self.metric,self.distance)
            self.timing['selection']+=time.perf_counter()-start
        if self.rule=='medoid':out=center
        elif self.rule=='single':out=candidates[0]
        elif self.rule=='mean_after_repair':
            valid=np.all(np.isfinite(candidates),axis=-1)
            count=valid.sum(axis=0)
            mean=np.where(np.isfinite(candidates),candidates,0).sum(axis=0)/np.maximum(count,1)[:,None]
            mean[count==0]=np.nan
            began=time.perf_counter();out=self.repair(mean[None,...],x)[0]
            self.timing['mean_repair']+=time.perf_counter()-began
        else:raise ValueError('unknown aggregation rule')
        selected=np.full(len(x),-1)
        if self.rule in ('medoid','single'):
            valid_choice=choice>=0;selected[valid_choice]=self.seeds[choice[valid_choice]]
        self.last_diagnostics=dict(selected_seed=selected,**stats)
        return out

    def save(self,path):
        Path(path).write_text(json.dumps(dict(paths=self.paths,seeds=self.seeds.tolist(),metric=self.metric,rule=self.rule,distance=self.distance,collect_mean_diagnostics=self.collect_mean_diagnostics,physics_backend=self.physics_backend),indent=2))

    @classmethod
    def load(cls,path):return cls(**json.loads(Path(path).read_text()))
