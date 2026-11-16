"""Independent single-launch supervisor for the published frozen continuation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,traceback

def write(p,j):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(j,indent=2)+'\n');os.replace(temp,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(root,follow):
    follow.mkdir(parents=True,exist_ok=True)
    lock=(follow/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (follow/'supervisor_started.json').exists() and not (root/'worker_started.json').exists(), 'Do not duplicate or restart'
    cfg=json.loads((root/'config.json').read_text());prep=json.loads((root/'prepared.json').read_text());pub=json.loads((follow/'publication.json').read_text())
    assert not cfg['fixture'] and pub['status']=='passed' and pub['remote_verified'] and pub['branch']=='bw'
    assert pub['prepared_sha256']==sha(root/'prepared.json') and pub['source_manifest_sha256']==sha(root/'source_manifest.json')
    remaining=cfg['total_budget_seconds']-prep['prepare_seconds']-prep['prior_compute_seconds'];assert remaining>0
    started=time.perf_counter();deadline=started+remaining
    program="import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('kinetics.react_stepper_refinement',run_name='__main__')"
    command=['/workspace/kinetics-venv/bin/python','-c',program,str(root/'source_snapshot'),'run','--root',str(root),
        '--publication',str(follow/'publication.json'),'--deadline',str(deadline)]
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',
        MPLCONFIGDIR='/tmp/react-stepper-refinement-mpl',XDG_CACHE_HOME='/tmp/react-stepper-refinement-cache')
    with root.with_suffix('.log').open('w') as log:
        child=subprocess.Popen(command,cwd='/workspace/personal',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(follow/'supervisor_started.json',dict(pid=os.getpid(),worker_pid=child.pid,command=command,
            at=datetime.now(timezone.utc).isoformat(),remaining_seconds=remaining,deadline_monotonic=deadline,
            commit_sha=pub['commit_sha'],source_manifest_sha256=pub['source_manifest_sha256']))
        timed_out=False
        try:code=child.wait(timeout=max(1.,deadline-time.perf_counter()))
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
    result=dict(returncode=code,timeout=timed_out,at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
    if (root/'complete.json').exists():result['complete_sha256']=sha(root/'complete.json')
    write(follow/'exit.json',result)
    if code:write(follow/'failure.json',dict(**result,worker_log=str(root.with_suffix('.log'))))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--follow',type=Path,required=True);a=p.parse_args()
    try:main(a.root,a.follow)
    except Exception:
        a.follow.mkdir(parents=True,exist_ok=True);write(a.follow/'failure.json',dict(traceback=traceback.format_exc()))
        write(a.follow/'exit.json',dict(returncode=1));raise
