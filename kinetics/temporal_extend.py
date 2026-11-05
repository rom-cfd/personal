"""One bounded continuation after the pilot, only if loss has not flattened.

Restores exact model/Adam states. Each model gets at most another 150 active
seconds. Multi-step training continues at K=64, rather than restarting K=1.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .temporal_pilot import (METHODS, tf, prepare_training, valid_starts,
    normalized_features, recursive_loss, write_json, sha, evaluate_split, freeze_selection)
from .temporal_report import produce_report, verify


def diagnose(short):
    rows=[]
    for member in json.loads((short/'final_members.json').read_text()):
        folder=short/'models'/f'{member["method"]}_seed{member["seed"]}'
        h=pd.read_csv(folder/'history.csv')
        # Compare only the final K and final LR; K changes are not loss progress.
        tail=h[(h.length==h.length.iloc[-1])&(h.learning_rate==h.learning_rate.iloc[-1])]
        gain=None
        enough=len(tail)>=10 and tail.active_seconds.iloc[-1]-tail.active_seconds.iloc[0]>=10.
        if enough:
            half=len(tail)//2
            before=float(tail.loss.iloc[:half].median())
            after=float(tail.loss.iloc[half:].median())
            gain=1.-after/before if before>0 else 0.
        flat=bool(enough and abs(gain)<.01)
        rows.append(dict(method=member['method'],seed=member['seed'],length=int(h.length.iloc[-1]),
            final_lr=float(h.learning_rate.iloc[-1]),tail_records=len(tail),
            relative_loss_reduction=gain,plateau_supported=flat,
            reason='tail_change_below_1pct' if flat else
                   'insufficient_same_K_same_LR_observations' if not enough else 'tail_loss_still_changing'))
    return rows


def continue_one(member,short,root,prep,seconds):
    method,seed=member['method'],member['seed']
    origin=short/'models'/f'{method}_seed{seed}'
    out=root/'models'/f'{method}_seed{seed}'
    out.mkdir(parents=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model=tf.keras.models.load_model(origin/'model.keras',compile=True)
    optimizer=model.optimizer
    old_iterations=int(optimizer.iterations.numpy())
    if old_iterations!=member['updates'] or sha(origin/'model.keras')!=member['model_sha256']:
        raise ValueError('Predecessor weights/Adam state mismatch')
    optimizer_before=[np.array(v.numpy(),copy=True) for v in optimizer.variables]
    # Keep Adam moments; only the learning rate may subsequently change.
    z=tf.constant(prep['z']);q=tf.constant(prep['q']);dense=tf.constant(prep['dense'])
    parents=tf.constant(prep['local_parent'])
    mean=tf.constant(prep['feature_norm'].mean,tf.float32)
    scale=tf.constant(prep['feature_norm'].scale,tf.float32)
    length=64 if method=='multistep' else 1
    pools={k:valid_starts(prep['parent'],k,5000) for k in (1,2,4,8,64)}

    def objective(ids,horizons,training):
        initial=tf.gather(z,ids);pressure=tf.gather(q,ids);par=tf.gather(parents,ids)
        if method=='multistep':
            times=par[:,1:2]+tf.range(1,65)[None,:]
            ic=tf.broadcast_to(par[:,:1],tf.shape(times))
            targets=tf.gather_nd(dense,tf.stack([ic,times],axis=-1))
            return recursive_loss(model,initial,pressure,targets,mean,scale)
        targets=tf.gather_nd(dense,tf.stack([par[:,0],par[:,1]+horizons],axis=-1))
        f=normalized_features(initial,pressure,mean,scale,horizons if method=='interval' else None)
        return tf.reduce_mean((initial+model(f,training=training)-targets)**2)

    @tf.function(input_signature=[tf.TensorSpec([512],tf.int32),tf.TensorSpec([512],tf.int32)])
    def step(ids,horizons):
        with tf.GradientTape() as tape:
            loss=tf.debugging.check_numerics(objective(ids,horizons,True),'nonfinite continuation loss')
        grads=tape.gradient(loss,model.trainable_variables)
        grads,norm=tf.clip_by_global_norm(grads,5.)
        for g in grads:
            tf.debugging.assert_all_finite(g,'nonfinite continuation gradient')
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        return loss,norm

    @tf.function(input_signature=[tf.TensorSpec([512],tf.int32),tf.TensorSpec([512],tf.int32)])
    def probe(ids,horizons):
        return objective(ids,horizons,False)

    compile_start=time.perf_counter()
    step.get_concrete_function();probe.get_concrete_function()
    compile_seconds=time.perf_counter()-compile_start
    # Verify tracing did not mutate any restored optimizer tensor.
    for a,b in zip(optimizer_before,optimizer.variables):
        np.testing.assert_array_equal(a,b.numpy())
    rng=np.random.default_rng(seed+100000)

    def batch(generator):
        if method=='interval':
            horizons=np.r_[np.ones(256,np.int32),generator.choice([2,4,8],256).astype(np.int32)]
            ids=np.empty(512,np.int32)
            for k in (1,2,4,8):
                mask=horizons==k
                ids[mask]=generator.choice(pools[k],int(mask.sum()))
            return ids,horizons
        return generator.choice(pools[length],512),np.ones(512,np.int32)

    probe_ids,probe_horizons=batch(np.random.default_rng(seed+900000))
    started=time.perf_counter()
    first_probe=float(probe(probe_ids,probe_horizons).numpy())
    active=time.perf_counter()-started
    best=first_probe;bad=0;plateau=False;updates=0;last_probe=active
    history=[dict(update=old_iterations,length=length,loss=first_probe,probe_loss=first_probe,
                  learning_rate=float(optimizer.learning_rate.numpy()),active_seconds=active)]
    next_log=0.
    while active<seconds:
        started=time.perf_counter()
        ids,horizons=batch(rng)
        loss,gradnorm=step(ids,horizons)
        loss=float(loss.numpy());gradnorm=float(gradnorm.numpy())
        active+=time.perf_counter()-started;updates+=1
        probe_loss=None
        if active-last_probe>=5. or active>=seconds:
            started=time.perf_counter()
            probe_loss=float(probe(probe_ids,probe_horizons).numpy())
            active+=time.perf_counter()-started;last_probe=active
            if probe_loss<best*.995:
                best=probe_loss;bad=0
            else:
                bad+=1
            lr=float(optimizer.learning_rate.numpy())
            if bad>=3 and lr>1.00001e-6:
                optimizer.learning_rate.assign(max(lr*.3,1e-6))
                bad=0
            elif bad>=6 and lr<=1.00001e-6:
                plateau=True
        if updates%100==0 or probe_loss is not None or updates==1:
            history.append(dict(update=old_iterations+updates,length=length,loss=loss,
                probe_loss=probe_loss,gradient_norm=gradnorm,
                learning_rate=float(optimizer.learning_rate.numpy()),active_seconds=active))
        if active>=next_log or plateau or active>=seconds:
            status=dict(phase='continuation_training',method=method,seed=seed,length=length,
                        added_updates=updates,loss=loss,learning_rate=float(optimizer.learning_rate.numpy()),
                        active_seconds=active,budget_seconds=seconds)
            write_json(root/'status.json',status)
            print('CONTINUE',json.dumps(status),flush=True)
            next_log=active+10.
        if plateau:
            break
    model.save(out/'model.keras')
    tf.train.Checkpoint(model=model,optimizer=optimizer).write(str(out/'optimizer_checkpoint'))
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    reloaded=tf.keras.models.load_model(out/'model.keras',compile=True)
    features=normalized_features(z[:8],q[:8],mean,scale,tf.ones(8,tf.int32) if method=='interval' else None)
    np.testing.assert_array_equal(model(features).numpy(),reloaded(features).numpy())
    row=dict(method=method,seed=seed,gamma=.2,dt=2e-6,updates=old_iterations+updates,
        added_updates=updates,predecessor_updates=old_iterations,active_seconds=active,
        cumulative_active_seconds=active+member['active_seconds'],budget_seconds=seconds,
        compile_seconds=compile_seconds,parameters=model.count_params(),
        termination='plateau' if plateau else 'time_cap',plateau_reached=plateau,
        model_sha256=sha(out/'model.keras'),predecessor_model_sha256=sha(origin/'model.keras'),
        optimizer_restored_exactly=True,first_probe_loss=first_probe,
        final_learning_rate=float(optimizer.learning_rate.numpy()))
    write_json(out/'training.json',row)
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--short',default='results/temporal_pilot/run_001')
    parser.add_argument('--output',default='results/temporal_pilot_long/run_001')
    parser.add_argument('--seconds',type=float,default=150.)
    args=parser.parse_args();short=Path(args.short);root=Path(args.output)
    if root.exists():
        raise FileExistsError('Never overwrite a continuation')
    if json.loads((short/'complete.json').read_text())['status']!='passed':
        raise ValueError('Short run is not verified')
    diagnostics=diagnose(short)
    root.mkdir(parents=True)
    write_json(root/'loss_diagnosis.json',diagnostics)
    if all(d['plateau_supported'] for d in diagnostics if d['method']!='single_step'):
        write_json(root/'status.json',dict(phase='skipped',reason='new_methods_tail_change_below_1pct'))
        print('CONTINUATION_SKIPPED',flush=True)
        return
    started=time.perf_counter()
    config=json.loads((short/'config.json').read_text())
    config.update(predecessor=str(short),predecessor_complete_sha256=sha(short/'complete.json'),
        training_from_scratch=False,active_seconds_per_member=args.seconds,
        multistep_curriculum=[64],curriculum_active_time_fractions=[1.],
        learning_rate_rule='continue stored LR; fixed train probe every 5 active seconds; reduce x.3 after 3 checks without .5% gain; floor 1e-6',
        plateau_rule='at LR floor 1e-6, 6 further checks without cumulative .5% probe improvement',
        plateau_stopping=True,continuation_reason='loss still changing or insufficient same-K same-LR plateau evidence; all arms extended for matched budgets')
    write_json(root/'config.json',config)
    preprocessing=json.loads((short/'preprocessing.json').read_text())
    write_json(root/'preprocessing.json',preprocessing)
    sources=[Path(__file__),Path(__file__).with_name('temporal_pilot.py'),
             Path(__file__).with_name('temporal_report.py')]
    write_json(root/'source_snapshot.json',{str(p):sha(p) for p in sources})
    source=Path(config['source']);pairs=Path(config['pairs'])
    prep=prepare_training(source,pairs)
    training=[]
    members=json.loads((short/'final_members.json').read_text())
    for member in members:
        training.append(continue_one(member,short,root,prep,args.seconds))
        write_json(root/'final_members.json',training)
    pd.DataFrame(training).to_csv(root/'training.csv',index=False)
    write_json(root/'training_complete.json',dict(models=len(training),
        plateau=sum(m['plateau_reached'] for m in training),
        time_cap=sum(not m['plateau_reached'] for m in training)))
    val=evaluate_split(root,source,config['seeds'],'val',preprocessing)
    freeze_selection(root,val)
    evaluate_split(root,source,config['seeds'],'test',preprocessing)
    produce_report(root,source)
    report=root/'report/REPORT_KO.md'
    text=report.read_text()
    text=text.replace('세 방법을 seed 11/22/33으로 각각 처음부터 학습했다. 각 모델은 활성 학습 계산 150초이며, 기존 10,000회 모델의 연속 학습이 아니다.',
        '세 방법 모두 직전 예비 실험의 가중치와 Adam 상태를 정확히 복원해 모델당 최대 150초 추가 학습했다. 누적 활성 학습 예산은 모델당 약 300초이다. 고정 training probe 정체 시 LR을 줄였으며, LR 하한 1e-6에서 6회 더 개선이 없을 때만 평탄 종료로 판정했다.')
    text=text.replace('다중 스텝은 활성 시간의 각 1/4을 K=1/4/16/64에 배정하고 예측 상태를 되먹여 전체 경로에 역전파한다. 후반 K의 업데이트 수가 적을 수 있다.',
        '다중 스텝은 K=64를 유지해 추가 학습했다. K=1부터 재시작하지 않았고, 입력은 자기 예측을 되먹이며 전체 경로에 역전파했다.')
    report.write_text(text)
    verification=verify(root,source,pairs)
    write_json(root/'verification.json',verification)
    if verification['status']!='passed':
        raise RuntimeError('Continuation verification failed')
    # Matched IC intersection, same gamma, no incomplete full-profile scores.
    comparison=[]
    for split in ('val','test'):
        short_scores=pd.read_csv(short/f'evaluations/{split}/NRMSE.csv')
        long_scores=pd.read_csv(root/f'evaluations/{split}/NRMSE.csv')
        keys=['method','predictor','split','mode','trajectory','variable']
        joined=short_scores.merge(long_scores,on=keys,suffixes=('_short','_long'))
        joined['common_complete']=joined.complete_short&joined.complete_long
        joined.to_csv(root/f'report/{split}_short_long_NRMSE.csv',index=False)
        for (method,predictor,mode,variable),g in joined.groupby(['method','predictor','mode','variable']):
            common=g[g.common_complete]
            comparison.append(dict(split=split,method=method,predictor=predictor,mode=mode,variable=variable,
                common_completed=len(common),short_complete=int(g.complete_short.sum()),long_complete=int(g.complete_long.sum()),
                short_median=float(common.nrmse_short.median()) if len(common) else np.nan,
                long_median=float(common.nrmse_long.median()) if len(common) else np.nan))
    pd.DataFrame(comparison).to_csv(root/'report/short_long_common_summary.csv',index=False)
    write_json(root/'complete.json',dict(status='passed',wall_seconds=time.perf_counter()-started,
        verification_sha256=sha(root/'verification.json')))
    write_json(root/'status.json',dict(phase='complete',status='passed'))
    print('CONTINUATION_COMPLETE',root,flush=True)


if __name__=='__main__':
    main()
