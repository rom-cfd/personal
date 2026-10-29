"""Pilot/full paper-schedule training on retained data. Full execution needs budget approval.

Default is a short validation-only pipeline pilot. No pilot is activated as a
trained production model. Full mode executes both 2500-epoch stages; a short
budget interrupts with saved weights/progress, never claims full completion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .wang_paper import PaperPredictor,BiasCorrectedAdam
from .study import tf
from .metrics import evaluate
from .committee_study import run_rollouts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/wang2025_paper.json')
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--pilot-updates-per-stage',type=int,default=5)
    parser.add_argument('--budget-seconds',type=float,default=120)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    if args.budget_seconds<=0 or args.pilot_updates_per_stage<=0:raise ValueError('Positive limits required')
    output=Path(cfg['output'])/('full' if args.full else 'pilot')
    if output.exists():raise FileExistsError('Preserve prior run; use a config with new output')
    output.mkdir(parents=True);(output/'config.json').write_text(json.dumps(cfg,indent=2))
    with np.load(cfg['training_pairs']) as data:
        x,y,parent,dt=data['x'],data['y'],data['parent'],float(data['dt'])
    with np.load(cfg['source']) as data:
        states,pressures,split,matrix=data['states'],data['pressures'],data['split'],data['element_matrix']
    assert dt==cfg['dt'] and set(parent[:,0])<=set(np.flatnonzero(split=='train'))
    started=time.perf_counter();deadline=started+args.budget_seconds
    model=PaperPredictor(x,cfg['seed']);optimizer=BiasCorrectedAdam(model.model.trainable_variables)
    features=model.features(x);target=model.targets(y)
    @tf.function(input_signature=[tf.TensorSpec([None],tf.int32),tf.TensorSpec([],tf.float32)])
    def step(indices,lr):
        a=tf.gather(feature_tensor,indices);b=tf.gather(target_tensor,indices)
        with tf.GradientTape() as tape:
            loss=tf.reduce_mean(tf.abs(model.model(a,training=True)-b))
        optimizer.apply(tape.gradient(loss,model.model.trainable_variables),model.model.trainable_variables,lr)
        return loss
    feature_tensor=tf.constant(features);target_tensor=tf.constant(target)
    rng=np.random.default_rng(cfg['seed']);history=[];total=0;status='running'
    def save_state(status):
        model.save(output/'model')
        np.savez_compressed(output/'optimizer.npz',t=optimizer.t.numpy(),
            **{f'm_{k}':m.numpy() for k,m in enumerate(optimizer.m)},
            **{f'v_{k}':v.numpy() for k,v in enumerate(optimizer.v)})
        pd.DataFrame(history).to_csv(output/'training.csv',index=False)
        (output/'progress.json').write_text(json.dumps(dict(status=status,full_requested=args.full,
            updates=total,elapsed_seconds=time.perf_counter()-started,budget_seconds=args.budget_seconds,
            rng_state=rng.bit_generator.state),indent=2))
    try:
        for stage_index,stage in enumerate(cfg['stages'],1):
            stage_updates=0
            for epoch in range(1,stage['epochs']+1):
                order=rng.permutation(len(x)).astype('int32');losses=[]
                for begin in range(0,len(x),stage['batch_size']):
                    if time.perf_counter()>deadline:raise TimeoutError('Approved run budget exhausted')
                    indices=order[begin:begin+stage['batch_size']]
                    loss=float(step(indices,np.float32(stage['learning_rate'])).numpy())
                    if not np.isfinite(loss):raise ValueError('Nonfinite training loss')
                    losses.append(loss);stage_updates+=1;total+=1
                    history.append(dict(stage=stage_index,epoch=epoch,update=total,batch=len(indices),loss_G_L1=loss,
                                        learning_rate=stage['learning_rate']))
                    if not args.full and stage_updates>=args.pilot_updates_per_stage:break
                if epoch%100==0 or not args.full:
                    print('PAPER_STAGE',stage_index,'EPOCH',epoch,'UPDATES',total,'LOSS',np.mean(losses),'SECONDS',time.perf_counter()-started,flush=True)
                if epoch%100==0:save_state('training')
                if not args.full and stage_updates>=args.pilot_updates_per_stage:break
        status='full_training_completed' if args.full else 'pipeline_pilot_completed'
        save_state(status)
        restored=PaperPredictor.load(output/'model')
        np.testing.assert_allclose(model.predict(x[:8]),restored.predict(x[:8]),atol=1e-8,rtol=1e-10,equal_nan=True)
        partitions=['val','test'] if args.full else ['val']
        summaries=[];rows_all=[]
        for partition in partitions:
            if time.perf_counter()>deadline:raise TimeoutError('Evaluation budget exhausted')
            ids=np.flatnonzero(split==partition)
            # Pilot is a wiring check, not reported as comparable scientific accuracy.
            if not args.full:ids=ids[:3]
            j=np.unique(np.r_[np.arange(8),np.linspace(0,states.shape[1]-2,24).astype(int)])
            px=np.c_[states[ids][:,j].reshape(-1,10),np.repeat(pressures[ids],len(j))]
            py=states[ids][:,j+1].reshape(-1,10)
            pred=model.predict(px)
            pd.DataFrame([dict(split=partition,**evaluate(py,pred,px,matrix))]).to_csv(output/f'one_step_{partition}.csv',index=False)
            def progress(step,active):
                if time.perf_counter()>deadline:raise TimeoutError('Evaluation budget exhausted')
                print('PAPER_ROLLOUT',partition,step,active,flush=True)
            summary,rows,sp,events,traces=run_rollouts(model,states[ids],pressures[ids],ids,matrix,dt,progress)
            summaries.append(dict(split=partition,**summary));rows_all.extend(dict(split=partition,**r) for r in rows)
            np.savez_compressed(output/f'traces_{partition}.npz',**{f'trajectory_{int(i)}':t for i,t in zip(ids,traces)})
        pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
        pd.DataFrame(rows_all).to_csv(output/'trajectories.csv',index=False)
        (output/'manifest.json').write_text(json.dumps(dict(status=status,full_training_completed=args.full,
            updates=total,parameters=model.model.count_params(),elapsed_seconds=time.perf_counter()-started,
            pilot_is_not_accuracy_comparison=not args.full,physics_counts=model.counts,
            source_sha256=hashlib.sha256(Path(cfg['source']).read_bytes()).hexdigest(),
            pair_sha256=hashlib.sha256(Path(cfg['training_pairs']).read_bytes()).hexdigest(),
            normalization='train-only input z-score; no output normalization',
            authors_unpublished_details_reproduced=False),indent=2))
        print('PAPER_RUN_COMPLETE',status,total,time.perf_counter()-started,flush=True)
    except BaseException:
        save_state('interrupted');raise


if __name__=='__main__':main()
