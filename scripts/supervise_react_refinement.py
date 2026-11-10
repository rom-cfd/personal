"""Independent supervisor; publication is required and no restart is attempted."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,fcntl,signal,subprocess

ROOT=Path('/workspace/personal/results/react_refinement/run_001')
AUDIT=Path('/workspace/react_refinement_followup/run_001')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,value):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)


def main():
    AUDIT.mkdir(parents=True,exist_ok=True);lock=(AUDIT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (AUDIT/'supervisor_started.json').exists() or (ROOT/'worker_started.json').exists():raise RuntimeError('No duplicate or renewed budget')
    pub=json.loads((AUDIT/'publication.json').read_text());cfg=json.loads((ROOT/'config.json').read_text());prepared=json.loads((ROOT/'prepared.json').read_text())
    assert pub['status']=='passed' and pub['remote_verified'] and pub['branch']=='bw'
    assert pub['source_manifest_sha256']==sha(ROOT/'source_manifest.json')==prepared['source_manifest_sha256']
    assert pub['prepared_sha256']==sha(ROOT/'prepared.json')
    remaining=cfg['budget_seconds']-prepared['prior_seconds']-prepared['prepare_seconds'];assert remaining>cfg['reserve_evaluation_seconds']
    write(AUDIT/'supervisor_started.json',dict(pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),remaining_seconds=remaining,commit_sha=pub['commit_sha']))
    command=['/workspace/kinetics-venv/bin/python',str(ROOT/'source_snapshot/run.py'),'--phase','run','--root',str(ROOT)]
    with (ROOT.parent/'run_001.log').open('x') as log:
        p=subprocess.Popen(command,cwd='/workspace/personal',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(AUDIT/'launch.json',dict(pid=p.pid,command=command,at=datetime.now(timezone.utc).isoformat(),no_restart=True))
        try:code=p.wait(timeout=remaining);timeout=False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=124;timeout=True
        record=dict(returncode=code,timeout=timeout,at=datetime.now(timezone.utc).isoformat())
        if code==0:record['complete_sha256']=sha(ROOT/'complete.json')
        write(AUDIT/'exit.json',record);print(json.dumps(record),flush=True);raise SystemExit(code)


if __name__=='__main__':main()
