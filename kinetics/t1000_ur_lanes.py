"""Run the ten UR=10 fine-tunes (5 seeds x {raw feedback, physics in loop}) as lanes on the allocated GPUs.

Lane i is pinned to GPU i mod n; each lane pulls (config, seed) tasks from one queue and runs
kinetics.unroll_training --resume as a subprocess (re-running the job resumes every task).
Physics-in-loop tasks are queued first because they take longer.
"""
import argparse,os,queue,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def lane(jobs,gpu,logs):
    env={**os.environ,'TF_FORCE_GPU_ALLOW_GROWTH':'true'}
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=gpu
    while True:
        try:config,seed=jobs.get_nowait()
        except queue.Empty:return
        name=f'{Path(config).stem}_s{seed}';print('TASK_START',name,'gpu',gpu,flush=True)
        with (logs/f'{name}.log').open('a') as f:
            code=subprocess.run([sys.executable,'-u','-m','kinetics.unroll_training','--config',config,'--seed',str(seed),'--resume'],
                stdout=f,stderr=subprocess.STDOUT,env=env).returncode
        print('TASK_END',name,'exit',code,flush=True)
        if code:raise RuntimeError(f'{name} failed; see {logs/(name+".log")}')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configs',nargs='+',default=['configs/t1000/ur10_phys.json','configs/t1000/ur10.json'])
    ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55])
    ap.add_argument('--tasks',nargs='+',help='explicit CONFIG:SEED tasks in queue order (overrides --configs/--seeds)')
    ap.add_argument('--lanes',type=int,default=6);ap.add_argument('--logs',type=Path,default=Path('results/t1000_logs/ur'))
    a=ap.parse_args();a.logs.mkdir(parents=True,exist_ok=True)
    jobs=queue.Queue()
    if a.tasks:
        for t in a.tasks:c,s=t.rsplit(':',1);jobs.put((c,int(s)))
    else:
        for c in a.configs:
            for s in a.seeds:jobs.put((c,s))
    visible=os.environ.get('CUDA_VISIBLE_DEVICES');gpus=[g for g in visible.split(',') if g] if visible else [None]
    print('GPUS',gpus,'lanes',a.lanes,'tasks',jobs.qsize(),flush=True)
    with ThreadPoolExecutor(a.lanes) as pool:
        for f in [pool.submit(lane,jobs,gpus[i%len(gpus)],a.logs) for i in range(a.lanes)]:f.result()


if __name__=='__main__':main()
