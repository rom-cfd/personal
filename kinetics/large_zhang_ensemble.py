"""Train a ten-member GELU ensemble with one two-stage protocol on one GPU.

Stage 1 is large_zhang_training (10 epochs, lr 1e-3 step decay); seed 11 reuses
its existing stage-1 model. Stage 2 is large_zhang_continue to 1000 epochs,
batch 512, halving LR after 20 epochs without a lower epoch-mean training loss;
the last-epoch model is the result. Both programs lock
and bind one config per output root, so every seed gets its own tree. The 5.3M
standardized arrays are read-only symlinks to the original preprocessing cache.
Seeds run as concurrent lanes pulling from one queue; lane i is pinned to
allocated GPU i mod n (one GPU saturates near 3 lanes, so a second GPU about
doubles throughput). Rerunning resumes every seed from its checkpoint.
"""
import argparse,json,os,queue,shutil,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

SOURCE=Path('results/zhang_5p3m_gelu/g0.1')
STAGE1=Path('configs/zhang_5p3m_gelu.json');STAGE2=Path('configs/zhang_5p3m_gelu_1000_trainloss.json')


def prepare(seed,output,configs,source=SOURCE,stage1_cfg=STAGE1,stage2_cfg=STAGE2,reuse_seed11=True):
    tree=output/f'seed{seed}';stage1=tree/'stage10';stage2=tree/'stage1000'
    second={**json.loads(stage2_cfg.read_text()),'output':str(stage2),'seed':seed}
    configs.mkdir(parents=True,exist_ok=True)
    if seed==11 and reuse_seed11:
        assert second['base_root']==str(source);tree.mkdir(parents=True,exist_ok=True)
        return tree,[None,write(configs/f'seed{seed}_stage1000.json',second)]
    cache=stage1/'g0.1/preprocessing';cache.mkdir(parents=True,exist_ok=True)
    for name in ('features.npy','targets.npy'):
        link=cache/name
        if not link.exists():link.symlink_to((source/'preprocessing'/name).resolve())
    for name in ('preprocessing.json','metric.json'):
        if not (cache/name).exists():shutil.copyfile(source/'preprocessing'/name,cache/name)
    first={**json.loads(stage1_cfg.read_text()),'output':str(stage1),'seeds':[seed],'require_gpu':True,
        'scope':'additional ensemble member; same protocol as seed 11 stage 1'}
    second['base_root']=str(stage1/'g0.1')
    return tree,[write(configs/f'seed{seed}_{name}.json',cfg) for name,cfg in [('stage10',first),('stage1000',second)]]


def write(path,cfg):
    if path.exists():assert json.loads(path.read_text())==cfg,f'Config changed: {path}'
    else:path.write_text(json.dumps(cfg,indent=2)+'\n')
    return path


def lane(jobs,gpu):
    env={**os.environ,'TF_FORCE_GPU_ALLOW_GROWTH':'true'}
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=gpu
    while True:
        try:seed,tree,(first,second)=jobs.get_nowait()
        except queue.Empty:return
        print('LANE_START',seed,'gpu',gpu,flush=True)
        for module,config,log in [('large_zhang_training',first,'stage10.log'),('large_zhang_continue',second,'stage1000.log')]:
            if config is None:continue
            with (tree/log).open('a') as f:
                code=subprocess.run([sys.executable,'-u','-m',f'kinetics.{module}','--config',str(config),'--resume'],
                    stdout=f,stderr=subprocess.STDOUT,env=env).returncode
            print('STAGE',seed,module,'exit',code,flush=True)
            if code:raise RuntimeError(f'seed {seed} {module} failed; see {tree/log}')
        print('SEED_COMPLETE',seed,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55,66,77,88,99,110])
    parser.add_argument('--lanes',type=int,default=10)
    parser.add_argument('--output',type=Path,default=Path('results/zhang_5p3m_gelu_ens'))
    parser.add_argument('--configs',type=Path,default=Path('configs/zhang_5p3m_gelu_ens'))
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--source',type=Path,default=SOURCE,help='root whose preprocessing/ (features, targets, json) every seed links to')
    parser.add_argument('--stage1',type=Path,default=STAGE1);parser.add_argument('--stage2',type=Path,default=STAGE2)
    parser.add_argument('--fresh-seed11',action='store_true',help='train seed 11 stage 1 too instead of reusing SOURCE')
    args=parser.parse_args()
    jobs=queue.Queue()
    for s in args.seeds:jobs.put((s,*prepare(s,args.output,args.configs,args.source,args.stage1,args.stage2,not args.fresh_seed11)))
    if args.prepare_only:return
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    gpus=[g for g in visible.split(',') if g] if visible else [None]
    print('GPUS',gpus,'lanes',args.lanes,flush=True)
    with ThreadPoolExecutor(args.lanes) as pool:
        for future in [pool.submit(lane,jobs,gpus[i%len(gpus)]) for i in range(args.lanes)]:future.result()


if __name__=='__main__':main()
