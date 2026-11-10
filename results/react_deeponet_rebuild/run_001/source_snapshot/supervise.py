"""Independent, no-restart supervisor for the already-published frozen run."""
import os,json,time,signal,subprocess,hashlib,fcntl
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path('/workspace/personal/results/react_deeponet_rebuild/run_001')
AUDIT=Path('/workspace/react_rebuild_followup/run_001')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,value):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(p)


def main():
    AUDIT.mkdir(parents=True,exist_ok=True);lock=(AUDIT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (AUDIT/'supervisor_started.json').exists() or (ROOT/'worker_started.json').exists():raise RuntimeError('Refuse duplicate worker or renewed budget')
    publication=json.loads((AUDIT/'publication.json').read_text());prepared=json.loads((ROOT/'prepared.json').read_text());cfg=json.loads((ROOT/'config.json').read_text());data=json.loads((ROOT/'dataset.json').read_text())
    assert publication['status']=='passed' and publication['remote_verified'] and publication['branch']=='bw'
    assert publication['source_manifest_sha256']==sha(ROOT/'source_manifest.json')==prepared['source_manifest_sha256']
    remaining=cfg['execution']['total_budget_seconds']-prepared['prior_diagnostic_seconds']-data['prepare_seconds']
    assert remaining>cfg['execution']['reserve_evaluation_seconds']
    write(AUDIT/'supervisor_started.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),remaining_budget_seconds=remaining,
        commit_sha=publication['commit_sha'],source_manifest_sha256=publication['source_manifest_sha256']))
    command=['/workspace/kinetics-venv/bin/python',str(ROOT/'source_snapshot/run.py'),'--phase','run','--root',str(ROOT)]
    with (ROOT.parent/'run_001.log').open('x') as log:
        child=subprocess.Popen(command,cwd='/workspace/personal',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(AUDIT/'launch.json',dict(worker_pid=child.pid,command=command,started_at=datetime.now(timezone.utc).isoformat(),no_automatic_restart=True))
        try:
            code=child.wait(timeout=remaining)
            record=dict(returncode=code,finished_at=datetime.now(timezone.utc).isoformat(),timeout=False)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            record=dict(returncode=124,finished_at=datetime.now(timezone.utc).isoformat(),timeout=True,message='Overall compute guard reached; completion is not claimed')
        if record['returncode']==0:
            record['complete_sha256']=sha(ROOT/'complete.json')
        write(AUDIT/'exit.json',record)
        print(json.dumps(record),flush=True)
        raise SystemExit(record['returncode'])


if __name__=='__main__':main()
