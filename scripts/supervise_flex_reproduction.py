"""One independent bounded worker, requiring verified pre-training publication."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--followup',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();follow=args.followup.resolve()
    snapshot=root/'source_snapshot';sys.path.insert(0,str(snapshot))
    from kinetics.flex_data import atomic,sha,stamp
    lock=(follow/'supervisor.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (follow/'exit.json').exists() and not (root/'exit.json').exists()
    publication=json.loads((follow/'publication.json').read_text())
    assert publication['status']=='passed' and publication['remote_verified'] is True
    assert publication['branch']=='bw' and len(publication['commit_sha'])==40
    assert publication['prepared_sha256']==sha(root/'prepared.json')
    launch=json.loads((follow/'launch.json').read_text())
    atomic(follow/'supervisor_started.json',dict(pid=os.getpid(),at=stamp(),command=sys.argv,
        source_manifest_sha256=sha(root/'source_manifest.json')))
    command=['/workspace/kinetics-venv/bin/python','-u','-m','kinetics.flex_study','run','--root',str(root)]
    environment=dict(os.environ,PYTHONPATH=str(snapshot),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR='/tmp/kinetics-mpl',XDG_CACHE_HOME='/tmp/kinetics-cache')
    remaining=launch['deadline_monotonic']-time.monotonic()
    assert remaining>60
    try:
        with (root.parent/'run_001.log').open('a') as log:
            p=subprocess.Popen(command,cwd=snapshot,env=environment,stdout=log,stderr=subprocess.STDOUT)
            atomic(follow/'worker_launch.json',dict(pid=p.pid,command=command,at=stamp(),deadline_monotonic=launch['deadline_monotonic']))
            try:returncode=p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                returncode=124
        result=dict(returncode=returncode,at=stamp(),timeout=returncode==124)
        if returncode==0:
            from scripts.monitor_flex_reproduction import inspect
            assert inspect(root,external=False)['verified_complete'] is True
            result['complete_sha256']=sha(root/'complete.json')
        atomic(follow/'exit.json',result)
    except Exception:
        atomic(follow/'failure.json',dict(at=stamp(),traceback=traceback.format_exc()))
        atomic(follow/'exit.json',dict(returncode=1,at=stamp()))
        raise


if __name__=='__main__':main()
