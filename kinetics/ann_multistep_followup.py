"""Independent sequential ANN study -> verified publication -> continuation.

No TensorFlow is imported here. Git publication starts only after short-phase
training/evaluation/verification has finished and never changes the checkout.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO=Path('/workspace/personal')
FOLLOW=Path('/workspace/ann_multistep_followup/run_001')
PYTHON='/workspace/kinetics-venv/bin/python'
SHORT=REPO/'results/ann_multistep/run_001'
LONG=REPO/'results/ann_multistep_long/run_001'
SCOPES=['kinetics/ann_multistep_study.py','kinetics/ann_multistep_extend.py',
        'kinetics/ann_multistep_report.py','kinetics/ann_multistep_followup.py',
        'tests/test_ann_multistep.py','results/ann_multistep/run_001','results/ann_multistep/run_001.log']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)


def git(*args,env=None,stdin=None,timeout=180):
    result=subprocess.run(['git','-c','gc.auto=0',*args],cwd=REPO,env=env,input=stdin,
                          text=True,capture_output=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError('git '+args[0]+': '+result.stderr[-4000:])
    return result.stdout.strip()


def check_short():
    for name in ('complete.json','verification.json','report/complete.json'):
        if json.loads((SHORT/name).read_text())['status']!='passed':
            raise RuntimeError('Short phase is not verified: '+name)
    complete=json.loads((SHORT/'complete.json').read_text())
    if complete['verification_sha256']!=digest(SHORT/'verification.json'):
        raise RuntimeError('Short verification SHA mismatch')
    if len(json.loads((SHORT/'final_members.json').read_text()))!=6:
        raise RuntimeError('Expected six ordinary ANN members')
    for filename,expected in json.loads((SHORT/'source_snapshot.json').read_text()).items():
        if digest(filename)!=expected:
            raise RuntimeError('Frozen execution source changed: '+filename)
    for filename,expected in json.loads((SHORT/'report/complete.json').read_text())['output_sha256'].items():
        if digest(SHORT/'report'/filename)!=expected:
            raise RuntimeError('Report changed: '+filename)


def publish():
    check_short()
    if git('branch','--show-current')!='bw':
        raise RuntimeError('Keep the existing bw checkout')
    if git('diff','--cached','--name-only'):
        raise RuntimeError('Unrelated staged changes must not be mixed')
    write(SHORT/'publication_request.json',dict(branch='bw',scope=SCOPES,
        reason='User requests publication before long ANN training',
        excludes=['other experiments','long ANN results','unrelated changes']))
    preflight=FOLLOW/'preflight.json'
    if not preflight.exists() or json.loads(preflight.read_text())['status']!='passed':
        raise RuntimeError('ANN preflight evidence missing')
    for filename,expected in json.loads(preflight.read_text()).get('source_sha256',{}).items():
        if digest(filename)!=expected:
            raise RuntimeError('Preflight source changed: '+filename)
    write(SHORT/'preflight.json',json.loads(preflight.read_text()))
    entries=[]
    for scope in SCOPES:
        path=REPO/scope
        files=sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        for filename in files:
            raw=filename.read_bytes()
            if filename.is_symlink() or len(raw)>=100*1024**2:
                raise RuntimeError('Publication path or GitHub size check failed')
            entries.append(dict(path=str(filename.relative_to(REPO)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                git_blob_sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()))
    write(FOLLOW/'publication_inventory.json',entries)
    parent=git('ls-remote','origin','refs/heads/bw').split()[0]
    # Metadata-only fetch when the remote advanced; no checkout or old blob hydration.
    if subprocess.run(['git','cat-file','-e',parent+'^{commit}'],cwd=REPO,capture_output=True).returncode:
        git('fetch','--no-auto-gc','--no-tags','--filter=blob:none','--depth=1','origin','refs/heads/bw')
        if git('rev-parse','FETCH_HEAD')!=parent:
            raise RuntimeError('Remote changed during metadata fetch; review before publishing')
    env=dict(os.environ,GIT_INDEX_FILE=str(FOLLOW/'publication.index'))
    git('read-tree',parent,env=env)
    git('add','--',*SCOPES,env=env)
    changes=git('diff','--cached','--name-status',parent,env=env).splitlines()
    expected={e['path'] for e in entries}
    git('diff','--cached','--check',parent,env=env)
    tree=git('write-tree','--missing-ok',env=env)
    message=FOLLOW/'commit_message.txt'
    message.write_text('''Add matched-time ordinary ANN multistep study

Train ordinary 64/64/64 MLP single-step and self-fed multistep controls
with three seeds, unchanged gamma=.2 data, and 150 active seconds each.
Include raw/repaired full profiles, NRMSE tables, exact Adam states,
training timings, architecture checks and verified profile scores.
Publish the short phase before the bounded continuation can start.
''')
    if not changes:
        # An earlier push may have succeeded before its acknowledgment was saved.
        commit=parent
    else:
        if len(changes)!=len(expected) or {v.split('\t',1)[1] for v in changes}!=expected or any(not v.startswith('A\t') for v in changes):
            raise RuntimeError('Publication must add only this new ANN experiment')
        commit=git('commit-tree',tree,'-p',parent,'-F',str(message))
        write(FOLLOW/'publication_prepared.json',dict(commit_sha=commit,parent_sha=parent,tree_sha=tree,files=len(entries)))
        # A non-force push rejects concurrent remote changes; no local ref is reset.
        git('-c','pack.threads=2','-c','pack.window=0','push','origin',commit+':refs/heads/bw',timeout=900)
    remote=git('ls-remote','origin','refs/heads/bw').split()[0]
    if remote!=commit:
        raise RuntimeError('Published remote bw SHA mismatch')
    for entry in entries:
        if digest(REPO/entry['path'])!=entry['sha256']:
            raise RuntimeError('Published file changed')
    record=dict(status='passed',remote_verified=True,branch='bw',commit_sha=commit,parent_sha=parent,
        tree_sha=tree,short_complete_sha256=digest(SHORT/'complete.json'),
        commit_url='https://github.com/rom-cfd/personal/commit/'+commit,
        files=len(entries),bytes=sum(e['bytes'] for e in entries),verified_unix=time.time(),
        local_snapshot_preserved=True,whitespace_check='passed')
    # This is the gate: finish all publication work before writing it.
    write(FOLLOW/'publication.json',record)
    print('PUBLISHED',record['commit_url'],flush=True)


def run_phase(module,output,log_path,exit_name,deadline):
    output.parent.mkdir(parents=True,exist_ok=True)
    started=time.time()
    with log_path.open('x') as log:
        process=subprocess.run([PYTHON,'-u','-m',module],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
                               timeout=max(1,deadline-time.monotonic()))
    write(FOLLOW/exit_name,dict(returncode=process.returncode,started_unix=started,ended_unix=time.time()))
    if process.returncode:
        raise RuntimeError(module+' failed; inspect '+str(log_path))


def main():
    FOLLOW.mkdir(parents=True,exist_ok=True)
    if (FOLLOW/'controller_started.json').exists():
        raise FileExistsError('Do not start a duplicate ANN experiment')
    write(FOLLOW/'controller_started.json',dict(pid=os.getpid(),started_unix=time.time()))
    deadline=time.monotonic()+21600
    try:
        run_phase('kinetics.ann_multistep_study',SHORT,SHORT.parent/'run_001.log','short_exit.json',deadline)
        write(FOLLOW/'status.json',dict(phase='publishing_short'))
        publish()
        write(FOLLOW/'status.json',dict(phase='starting_long',publication_verified=True))
        run_phase('kinetics.ann_multistep_extend',LONG,LONG.parent/'run_001.log','long_exit.json',deadline)
        if json.loads((LONG/'complete.json').read_text())['status']!='passed':
            raise RuntimeError('Long completion verification missing')
        write(FOLLOW/'status.json',dict(phase='complete',status='passed'))
        write(FOLLOW/'exit.json',dict(returncode=0,ended_unix=time.time()))
    except Exception as exc:
        write(FOLLOW/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),failed_unix=time.time()))
        write(FOLLOW/'exit.json',dict(returncode=1,ended_unix=time.time()))
        raise


if __name__=='__main__':
    main()
