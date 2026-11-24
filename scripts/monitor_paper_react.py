"""Small-file monitoring while running; full integrity checks only on exit."""
import argparse,hashlib,json
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def monitor(root,follow):
    result={'verified_complete':False,'root':str(root)}
    for label,p in [('status',root/'status.json'),('root_exit',root/'exit.json'),('supervisor_exit',follow/'exit.json'),('failure',follow/'failure.json'),('worker_failure',root/'failure.json')]:
        if p.exists():result[label]=json.loads(p.read_text())
    if not (follow/'exit.json').exists():return result
    assert result['supervisor_exit']['returncode']==0,result
    assert result['root_exit']['returncode']==0,result
    complete=json.loads((root/'complete.json').read_text());assert complete['status']=='passed'
    assert result['supervisor_exit']['complete_sha256']==result['root_exit']['complete_sha256']==digest(root/'complete.json')
    assert complete['verification_sha256']==digest(root/'verification.json')
    assert complete['report_complete_sha256']==digest(root/'report/complete.json')
    for name,expected in complete['output_sha256'].items():
        p=root/name;assert p.resolve().is_relative_to(root.resolve());assert digest(p)==expected,name
    report=json.loads((root/'report/complete.json').read_text());verification=json.loads((root/'verification.json').read_text())
    assert report['status']==verification['status']=='passed'
    for name,expected in report['output_sha256'].items():assert digest(root/'report'/name)==expected,name
    for name in ['all_profiles.pdf','ignition_zoom.pdf','summary.csv','loss_and_lr.png']:assert (root/'report'/name).is_file(),name
    result.update(verified_complete=True,complete_sha256=digest(root/'complete.json'),selection=json.loads((root/'selection_frozen.json').read_text()),verification=verification)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/personal/results/paper_react_jax/run_001'));p.add_argument('--follow',type=Path,default=Path('/workspace/paper_react_jax_followup/run_001'));a=p.parse_args()
    print(json.dumps(monitor(a.root,a.follow),ensure_ascii=False,indent=2))
