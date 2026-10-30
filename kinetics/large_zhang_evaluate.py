"""Frozen validation selection and 10/20-ms raw evaluation of completed GELU models."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .multiscale_large_data import atomic_json,sha
from .zhang_model import ZhangCommittee
from .zhang_raw_study import raw_rollouts,assess
from .data import samples
from .metrics import evaluate
from .long_committee_study import horizon_summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/zhang_5p3m_gelu/g0.1'))
    parser.add_argument('--seeds',type=int,nargs='+',default=[11])
    parser.add_argument('--source',type=Path,default=Path('results/long10ms_data/trajectories.npz'))
    parser.add_argument('--reference',type=Path,default=Path('results/zhang_multiscale_raw/reference20ms.npz'))
    parser.add_argument('--output',type=Path);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();seeds=sorted(args.seeds);root=args.root
    out=args.output or root/('evaluation_seeds'+'_'.join(map(str,seeds)))
    if out.exists():raise FileExistsError('Use a new evaluation output')
    cfg=json.loads((root/'config.json').read_text());assert cfg['physics'] is False
    records=[];paths=[]
    for seed in seeds:
        folder=root/f'seed{seed}';r=json.loads((folder/'completed.json').read_text())
        assert r['status']=='completed' and r['physics'] is False
        assert r['model_sha256']==sha(folder/'model.keras') and r['metadata_sha256']==sha(folder/'model.json')
        if not args.preflight:assert r['epochs']==10 and r['examples_per_epoch']==5300000
        records.append(r);paths.append(str(folder/'model'))
    preprocessing=json.loads((root/'preprocessing/preprocessing.json').read_text())
    assert all(r['preprocessing_sha256']==sha(root/'preprocessing/preprocessing.json') for r in records)
    out.mkdir(parents=True);(out/'methods').mkdir();(out/'traces').mkdir()
    with np.load(args.source) as d:data={k:d[k] for k in d.files}
    with np.load(args.reference) as d:
        refs={int(i):r for i,r in zip(d['ids'],d['states'])}
        taus={int(i):t for i,t in zip(d['ids'],d['refined_tau400'])}
        assert float(d['dt'])==float(data['dt'])
    # Same physical scale definition as the earlier medoid, fitted on full training data.
    dataset=Path(cfg['data']);dm=json.loads((dataset/'manifest.json').read_text())
    assert sha(dataset/'manifest.json')==preprocessing['data_manifest_sha256']
    raw_x_path=dataset/dm['arrays']['x']['path']
    assert sha(raw_x_path)==dm['arrays']['x']['sha256']
    raw_x=np.load(raw_x_path,mmap_mode='r');gamma=cfg['gamma']
    metric=dict(temperature_scale=max(float(preprocessing['input_scale'][0]),100.),
        species_scale=[max(float(np.quantile(raw_x[:,j],.95)),1e-8) for j in range(1,10)],
        fit_split='train',fit_examples=len(raw_x),derivation='temperature std and species 95th percentiles, training only')
    atomic_json(out/'protocol.json',dict(physics=False,seeds=seeds,gamma=gamma,preflight=args.preflight,
        epochs=cfg['epochs'],training_pairs=preprocessing['examples'],
        validation_selection='all 90000 one-step pairs; original physical-unit selection_score; ties ascending seed',
        test_or_rollout_for_selection=False,horizons=[.01,.02],raw_pressure_self_fed=True,
        decoder_clipping=False,element_projection=False,enthalpy_closure=False,mass_renormalization=False,
        medoid_metric=metric,reference_sha256=sha(args.reference),source_sha256=sha(args.source)))
    began=time.perf_counter();p=ZhangCommittee(paths,seeds,metric,'mean')
    vx,vy,_=samples(data,'val');predictions=np.empty((len(seeds),len(vx),10))
    for i in range(0,len(vx),512):predictions[:,i:i+512]=p.candidates(vx[i:i+512])[:,:,:10]
    ranking=[]
    for seed,pred in zip(seeds,predictions):
        with np.errstate(all='ignore'):m=evaluate(vy,pred,vx,data['element_matrix'])
        if not np.isfinite(m['selection_score']):m['selection_score']=float('inf')
        ranking.append(dict(seed=seed,**m))
    rank=pd.DataFrame(ranking).sort_values(['selection_score','seed']);rank.to_csv(out/'validation_ranking.csv',index=False)
    chosen=int(rank.iloc[0].seed)
    atomic_json(out/'selection_frozen.json',dict(seed=chosen,score=float(rank.iloc[0].selection_score),
        usable_validation_model=bool(np.isfinite(rank.iloc[0].selection_score)),selection_pool=len(seeds),test_used=False,rollout_used=False))
    print('FROZEN_SINGLE',chosen,'from',len(seeds),'models',flush=True)
    del predictions
    definitions=[('single_best',[paths[seeds.index(chosen)]],[chosen],'single')]
    if len(seeds)>1:definitions += [('mean'+str(len(seeds)),paths,seeds,'mean'),('medoid'+str(len(seeds)),paths,seeds,'medoid')]
    rows=[];summaries=[];one=[];checks=[];timings=[]
    for name,pp,ss,rule in definitions:
        p=ZhangCommittee(pp,ss,metric,rule);p.save(out/'methods'/f'{name}.json')
        for split in ('val','test'):
            x,y,ids=samples(data,split);prediction=np.empty_like(y)
            for i in range(0,len(x),512):prediction[i:i+512]=p.predict(x[i:i+512])
            with np.errstate(all='ignore'):one.append(dict(method=name,split=split,**evaluate(y,prediction,x,data['element_matrix'])))
            truth=np.asarray([refs[int(i)] for i in ids]);pressures=data['pressures'][ids]
            first=p.predict_full(np.c_[truth[:,0],pressures]);start=time.perf_counter()
            def progress():
                if time.perf_counter()-began>21600:raise TimeoutError('Evaluation six-hour cap reached')
            traces,statuses,events=raw_rollouts(p,truth,pressures,ids,float(data['dt']),progress)
            timings.append(dict(method=name,split=split,seconds=time.perf_counter()-start))
            np.savez_compressed(out/'traces'/f'{name}_{split}.npz',**{f'trajectory_{int(i)}':a for i,a in zip(ids,traces)})
            pd.DataFrame(events).to_csv(out/f'events_{name}_{split}.csv.gz',index=False)
            for horizon in [.01,.02]:
                rr=[]
                for i,a,t,pressure,status in zip(ids,traces,truth,pressures,statuses):
                    r,_=assess(a,t,pressure,float(data['dt']),data['element_matrix'],horizon,int(i),status,taus[int(i)])
                    rr.append(r);rows.append(dict(method=name,split=split,horizon=horizon,**r))
                summary=horizon_summary(rr)
                summaries.append(dict(method=name,split=split,horizon=horizon,**summary))
                print('RESULT',name,split,horizon,summary,flush=True)
            restored=ZhangCommittee.load(out/'methods'/f'{name}.json')
            np.testing.assert_allclose(first,restored.predict_full(np.c_[truth[:,0],pressures]),atol=1e-9,rtol=1e-8,equal_nan=True)
            for j,a in enumerate(traces):
                if len(a)>1:np.testing.assert_allclose(first[j],a[1],atol=1e-9,rtol=1e-8)
            checks.append(dict(method=name,split=split,reload_match=True,first_step_match=True))
            for filename,values in [('summary',summaries),('trajectories',rows),('one_step',one),('checks',checks),('timing',timings)]:
                pd.DataFrame(values).to_csv(out/f'{filename}.csv',index=False)
    atomic_json(out/'manifest.json',dict(status='completed',preflight=args.preflight,physics=False,
        models=len(seeds),epochs=cfg['epochs'],training_examples=preprocessing['examples'],
        execution_seconds=time.perf_counter()-began,model_records=records,evaluator_sha256=sha(__file__)))


if __name__=='__main__':main()
