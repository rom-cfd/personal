"""One launch, immutable preparation/publication gate, absolute wall deadline."""
import argparse,fcntl,json,os,signal,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import hashlib

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)

def main(root,follow):
    follow.mkdir(parents=True,exist_ok=True)
    lock=(follow/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (follow/'supervisor_started.json').exists() and not (root/'worker_started.json').exists(),'Do not restart'
    cfg=json.loads((root/'config.json').read_text());prep=json.loads((root/'prepared.json').read_text());pub=json.loads((follow/'publication.json').read_text())
    assert pub['status']=='passed' and pub['branch']=='bw' and pub['remote_verified']
    assert pub['prepared_sha256']==digest(root/'prepared.json') and pub['source_manifest_sha256']==digest(root/'source_manifest.json')
    preflight=json.loads((follow/'preflight.json').read_text());assert preflight['status']=='passed'
    remaining=cfg['execution']['total_budget_seconds']-prep['prepare_seconds']-preflight['elapsed_seconds'];assert remaining>cfg['execution']['evaluation_reserve_seconds']
    started=time.perf_counter();deadline=started+remaining
    command=['/workspace/kinetics-venv/bin/python','-c',"import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('kinetics.paper_react_jax',run_name='__main__')",str(root/'source_snapshot'),'run','--root',str(root),'--publication',str(follow/'publication.json'),'--deadline',str(deadline)]
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
    cpus=sorted(os.sched_getaffinity(0))[:cfg['execution']['threads']]
    with root.with_suffix('.log').open('w') as log:
        child=subprocess.Popen(command,cwd='/workspace/personal',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,cpus))
        write(follow/'supervisor_started.json',dict(pid=os.getpid(),worker_pid=child.pid,command=command,at=datetime.now(timezone.utc).isoformat(),deadline_monotonic=deadline,remaining_seconds=remaining,cpus=cpus))
        timeout=False
        try:code=child.wait(timeout=max(1,deadline-time.perf_counter()))
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
    result=dict(returncode=code,timeout=timeout,at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
    if (root/'complete.json').exists():result['complete_sha256']=digest(root/'complete.json')
    write(follow/'exit.json',result)
    if code:write(follow/'failure.json',dict(**result,worker_log=str(root.with_suffix('.log'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--follow',type=Path,required=True);a=p.parse_args()
    try:main(a.root,a.follow)
    except BaseException:
        a.follow.mkdir(parents=True,exist_ok=True);write(a.follow/'failure.json',dict(traceback=traceback.format_exc()));write(a.follow/'exit.json',dict(returncode=1));raise
