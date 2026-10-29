"""Run: python -m kinetics.study --output results/study --config configs/study.json.

All candidate selection and checkpointing use validation only. The test partition
is accessed once, after selection is frozen, for the three-seed final comparison.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from .data import generate, samples
from .metrics import evaluate, distributions
from .transforms import Transform, Standardizer, species_epsilon

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)


class Predictor:
    def __init__(self, spec, x, y, config, seed):
        self.spec=spec; self.config=config
        eps=species_epsilon(x[:,1:10],spec.get('multiplier',1e-3)) if spec.get('species',False) else spec.get('epsilon',1e-8)
        self.transform=Transform(spec['kind'],spec.get('gamma',.1),eps)
        a=self.encode_state(x[:,:10]); b=self.encode_state(y)
        self.norm=Standardizer().fit(np.r_[a,b])
        features=np.c_[self.norm.forward(a),np.log(x[:,-1])]
        self.input_norm=Standardizer().fit(features)
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
        inputs=tf.keras.Input((11,))
        h=inputs
        for width in config['hidden_layers']:
            h=tf.keras.layers.Dense(width,activation=config['activation'])(h)
        out=tf.keras.layers.Dense(10,kernel_initializer='zeros',bias_initializer='zeros')(h)
        self.model=tf.keras.Model(inputs,out)
        self.optimizer=tf.keras.optimizers.Adam(config['learning_rate'])
        self.train_features=self.features(x).astype('float32')
        self.train_target=(self.norm.forward(b)-self.norm.forward(a)).astype('float32')

    def encode_state(self,state):
        return np.c_[state[:,0]/1000,self.transform.forward(state[:,1:10])]

    def features(self,x):
        return self.input_norm.forward(np.c_[self.norm.forward(self.encode_state(x[:,:10])),np.log(x[:,-1])])

    def predict(self,x):
        delta=self.model(self.features(x).astype('float32'),training=False).numpy().astype('float64')
        z=self.norm.inverse(self.norm.forward(self.encode_state(x[:,:10]))+delta)
        return np.c_[z[:,0]*1000,self.transform.inverse(z[:,1:])]

    def save(self,path):
        self.model.save(Path(str(path)+'.keras'))
        Path(str(path)+'.json').write_text(json.dumps(dict(spec=self.spec,epsilon=self.transform.epsilon.tolist(),
            output_mean=self.norm.mean.tolist(),output_scale=self.norm.scale.tolist(),
            input_mean=self.input_norm.mean.tolist(),input_scale=self.input_norm.scale.tolist()),indent=2))

    @classmethod
    def load(cls,path):
        path=Path(path)
        info=json.loads(Path(str(path)+'.json').read_text())
        self=cls.__new__(cls)
        self.spec=info['spec']
        self.transform=Transform(self.spec['kind'],self.spec.get('gamma',.1),info['epsilon'])
        self.norm=Standardizer(np.array(info['output_mean']),np.array(info['output_scale']))
        self.input_norm=Standardizer(np.array(info['input_mean']),np.array(info['input_scale']))
        self.model=tf.keras.models.load_model(Path(str(path)+'.keras'),compile=False)
        return self


def train(spec,seed,epochs,train_data,val_data,data,cfg,deadline):
    start=time.perf_counter()
    x,y,_=train_data; vx,vy,_=val_data
    predictor=Predictor(spec,x,y,cfg,seed)
    features=tf.constant(predictor.train_features); target=tf.constant(predictor.train_target)
    @tf.function(input_signature=[tf.TensorSpec([None],tf.int32)])
    def step(ids):
        with tf.GradientTape() as tape:
            error=predictor.model(tf.gather(features,ids),training=True)-tf.gather(target,ids)
            loss=tf.reduce_mean(tf.square(error))
        predictor.optimizer.apply_gradients(zip(tape.gradient(loss,predictor.model.trainable_variables),predictor.model.trainable_variables))
        return loss
    rng=np.random.default_rng(seed)
    history=[]; best=float('inf'); weights=None; best_epoch=0
    for epoch in range(epochs):
        if time.perf_counter()>deadline: raise TimeoutError('Study CPU wall-time budget exhausted')
        order=rng.permutation(len(x)).astype('int32')
        losses=[]
        for offset in range(0,len(x),cfg['batch_size']):
            losses.append(float(step(order[offset:offset+cfg['batch_size']]).numpy()))
        if (epoch+1)%5==0 or epoch==epochs-1:
            p=predictor.predict(vx)
            m=evaluate(vy,p,vx,data['element_matrix'])
            transformed=(predictor.norm.forward(predictor.encode_state(vy))-predictor.norm.forward(predictor.encode_state(vx[:,:10])))
            output=predictor.model(predictor.features(vx).astype('float32'),training=False).numpy()
            m['transformed_mse']=float(np.mean((output-transformed)**2))
            history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),**m))
            if m['selection_score']<best:
                best=m['selection_score']; weights=predictor.model.get_weights(); best_epoch=epoch+1
    if weights is None: raise RuntimeError('No finite validation checkpoint')
    predictor.model.set_weights(weights)
    pred=predictor.predict(vx)
    metrics=evaluate(vy,pred,vx,data['element_matrix'])
    metrics.update(train_seconds=time.perf_counter()-start,best_epoch=best_epoch,
                   transformed_mse=next(h['transformed_mse'] for h in history if h['epoch']==best_epoch))
    return predictor,metrics,history


def rollouts(predictor,data,split,repair=False):
    rows=[]; traces=[]
    ids=np.flatnonzero(data['split']==split)
    states=data['states'][ids,0].copy()
    histories=[[s.copy()] for s in states]
    active=np.ones(len(ids),dtype=bool)
    failures=['completed']*len(ids); repairs=np.zeros(len(ids),int)
    first_bad=[None]*len(ids)
    for step in range(1,data['states'].shape[1]):
        live=np.flatnonzero(active)
        if not len(live): break
        predictions=predictor.predict(np.c_[states[live],data['pressures'][ids[live]]])
        for k,p in zip(live,predictions):
            bad=not np.all(np.isfinite(p)) or p[0]<200 or p[0]>6000 or np.any(p[1:]<0) or p[1:].sum()<=0
            if bad and first_bad[k] is None: first_bad[k]=step
            if not np.all(np.isfinite(p)) or p[0]<200 or p[0]>6000 or p[1:].sum()<=0:
                failures[k]='nonfinite_or_temperature_or_mass'; active[k]=False; continue
            if np.any(p[1:]<0) and not repair:
                failures[k]='negative_species'; active[k]=False; continue
            if repair:
                if np.any(p[1:]<0) or abs(p[1:].sum()-1)>1e-12: repairs[k]+=1
                p[1:]=np.maximum(p[1:],0); p[1:]/=p[1:].sum()
            states[k]=p; histories[k].append(p.copy())
    for k,i in enumerate(ids):
        true=data['states'][i]
        trace=np.array(histories[k]); n=len(trace)
        d=trace[:,1:]-true[:n,1:]
        rows.append(dict(trajectory=int(i),repair=repair,completed_steps=n-1,
                         horizon_steps=len(true)-1,status=failures[k],first_bad_step=first_bad[k],
                         repairs=int(repairs[k]),rmse_completed_prefix=float(np.sqrt(np.mean(d*d))),
                         temperature_rmse_completed_prefix=float(np.sqrt(np.mean((trace[:,0]-true[:n,0])**2))),
                         endpoint_species_rmse=float(np.sqrt(np.mean((trace[-1,1:]-true[-1,1:])**2))) if n==len(true) else float('nan'),
                         mass_sum_max=float(np.abs(trace[:,1:].sum(axis=1)-1).max()),
                         elemental_max=float(np.abs((trace[:,1:]-true[0,1:])@data['element_matrix'].T).max())))
        traces.append((int(i),trace))
    return rows,traces


def initial_candidates():
    c=[dict(name=f'boxcox_g{g}',kind='boxcox',gamma=g) for g in (.1,.3,.5)]
    for kind in ('softlog','asinh','shifted_boxcox'):
        c.append(dict(name=kind+'_global',kind=kind,epsilon=1e-8,gamma=.3))
        for mult in (1e-2,1e-4):
            c.append(dict(name=kind+f'_species_{mult:g}',kind=kind,species=True,multiplier=mult,gamma=.3))
    return c


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/study.json');parser.add_argument('--output',default='results/study');parser.add_argument('--dataset')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    out=Path(args.output)
    if out.exists(): raise FileExistsError('Use a new output directory; results are immutable')
    out.mkdir(parents=True); (out/'models').mkdir()
    started=time.perf_counter(); deadline=started+cfg['max_seconds']
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    if args.dataset:
        source=Path(args.dataset)
        shutil.copyfile(source,out/'dataset.npz')
        shutil.copyfile(source.with_suffix('.json'),out/'dataset.json')
        metadata=json.loads((out/'dataset.json').read_text())
        if hashlib.sha256((out/'dataset.npz').read_bytes()).hexdigest()!=metadata['sha256']:
            raise ValueError('Dataset checksum mismatch')
    else:
        metadata=generate(out/'dataset.npz',cfg['trajectories'],cfg['steps'],cfg['dt'],cfg['data_seed'])
    data=np.load(out/'dataset.npz',allow_pickle=False)
    train_data=samples(data,'train');val_data=samples(data,'val')
    print('DATA',metadata,flush=True)
    candidates=initial_candidates();screen=[];histories=[]
    def screen_one(spec,phase):
        predictor,m,h=train(spec,11,cfg['screen_epochs'],train_data,val_data,data,cfg,deadline)
        screen.append(dict(phase=phase,seed=11,**spec,**m))
        histories.extend(dict(phase=phase,name=spec['name'],seed=11,**v) for v in h)
        pd.DataFrame(screen).to_csv(out/'validation_screen.csv',index=False)
        print('SCREEN',phase,spec['name'],round(m['selection_score'],5),round(m['train_seconds'],2),flush=True)
        return predictor
    for spec in candidates: screen_one(spec,'initial')
    # Two validation-only optimization rounds. Each uses a fixed, equal budget.
    # Global and species-wise scales compete first; refine top epsilon families.
    for round_id in (1,2):
        eligible=[s for s in screen if s['kind']!='boxcox']
        best=[]
        for row in sorted(eligible,key=lambda s:s['selection_score']):
            if row['kind'] not in [s['kind'] for s in best]: best.append(row)
            if len(best)==2:break
        for parent in best:
            for factor in ((.1,10.) if round_id==1 else (10**-.5,10**.5)):
                spec={k:parent[k] for k in ('kind','gamma','species','multiplier','epsilon') if k in parent}
                key='multiplier' if spec.get('species',False) else 'epsilon'
                spec[key]=spec[key]*factor
                spec['name']=f'r{round_id}_{parent["name"]}_{factor:g}'
                candidates.append(spec);screen_one(spec,f'round{round_id}')
    # Freeze method selection before constructing any test samples.
    by_family={}
    for s in sorted(screen,key=lambda s:s['selection_score']):
        by_family.setdefault(s['kind'],s)
    chosen=[c for c in candidates if c['kind']=='boxcox']
    for kind in ('boxcox','softlog','asinh','shifted_boxcox'):
        row=by_family[kind];spec=next(c for c in candidates if c['name']==row['name'])
        if spec not in chosen:chosen.append(spec)
    (out/'selection_frozen.json').write_text(json.dumps(dict(methods=chosen,selection='validation only; best per family plus mandatory gamma=.1',test_accessed=False),indent=2))
    test_data=samples(data,'test');tx,ty,_=test_data
    final=[];species_rows=[];bins=[];roll_rows=[]
    for spec in chosen:
        for seed in cfg['final_seeds']:
            predictor,val,h=train(spec,seed,cfg['final_epochs'],train_data,val_data,data,cfg,deadline)
            tag=f'{spec["name"]}_seed{seed}'
            predictor.save(out/'models'/tag)
            t=time.perf_counter();pred=predictor.predict(tx);inference=time.perf_counter()-t
            m=evaluate(ty,pred,tx,data['element_matrix'])
            encoded=predictor.norm.forward(predictor.encode_state(ty))-predictor.norm.forward(predictor.encode_state(tx[:,:10]))
            network=predictor.model(predictor.features(tx).astype('float32'),training=False).numpy()
            final.append(dict(name=spec['name'],seed=seed,**m,transformed_mse=float(np.mean((encoded-network)**2)),
                              train_seconds=val['train_seconds'],inference_seconds=inference,
                              microseconds_per_sample=1e6*inference/len(tx),best_epoch=val['best_epoch'],validation_score=val['selection_score']))
            histories.extend(dict(phase='final',name=spec['name'],seed=seed,**v) for v in h)
            np.savez_compressed(out/f'predictions_{tag}.npz',pred=pred)
            for j,s in enumerate(data['species']):
                d=pred[:,j+1]-ty[:,j+1]
                species_rows.append(dict(name=spec['name'],seed=seed,species=str(s),rmse=float(np.sqrt(np.nanmean(d*d))),mae=float(np.nanmean(np.abs(d))),negative_fraction=float(np.mean(pred[:,j+1]<0))))
            bins.extend(dict(name=spec['name'],seed=seed,**b) for b in distributions(ty,pred,data['species']))
            for repair in (False,True):
                rr,traces=rollouts(predictor,data,'test',repair)
                roll_rows.extend(dict(name=spec['name'],seed=seed,**r) for r in rr)
                np.savez_compressed(out/f'rollout_{tag}_repair{int(repair)}.npz',**{f'trajectory_{i}':v for i,v in traces})
            pd.DataFrame(final).to_csv(out/'test_comparison.csv',index=False)
            pd.DataFrame(roll_rows).to_csv(out/'rollouts.csv',index=False)
            pd.DataFrame(species_rows).to_csv(out/'species_errors.csv',index=False)
            pd.DataFrame(bins).to_csv(out/'concentration_errors.csv',index=False)
            pd.DataFrame(histories).to_csv(out/'history.csv',index=False)
            print('FINAL',tag,round(m['selection_score'],5),'rmse',m['rmse'],flush=True)
    persistence=evaluate(ty,tx[:,:10],tx,data['element_matrix'])
    (out/'persistence_baseline.json').write_text(json.dumps(persistence,indent=2))
    manifest=dict(python=platform.python_version(),tensorflow=tf.__version__,numpy=np.__version__,
                  gpu=[str(d) for d in tf.config.list_physical_devices('GPU')],
                  wall_seconds=time.perf_counter()-started,dataset_sha256=metadata['sha256'],status='completed')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('COMPLETE',manifest,flush=True)


if __name__=='__main__':main()
