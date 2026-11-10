"""Cheap progress reads; completion requires every stored output hash."""
from pathlib import Path
import argparse,json,hashlib


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def inspect(root,external=True):
    root=Path(root);result=dict(root=str(root),verified_complete=False)
    for name in ['status','exit','failure','worker_started']:
        p=root/(name+'.json')
        if p.exists():result[name]=json.loads(p.read_text())
    if result.get('exit',{}).get('returncode')!=0:return result
    complete=json.loads((root/'complete.json').read_text());v=json.loads((root/'verification.json').read_text());r=json.loads((root/'report/complete.json').read_text())
    assert complete['status']==v['status']==r['status']=='passed'
    assert complete['verification_sha256']==sha(root/'verification.json') and complete['report_complete_sha256']==sha(root/'report/complete.json')
    assert all(sha(root/p)==h for p,h in v['output_sha256'].items()) and all(sha(root/'report'/p)==h for p,h in r['output_sha256'].items())
    for name in ['raw_profiles.pdf','repaired_profiles.pdf','ignition_zoom.pdf','representative_curves.pdf']:assert (root/'report'/name).is_file()
    if external:
        p=Path('/workspace/react_refinement_followup/run_001/exit.json')
        if not p.exists():result['awaiting_supervisor_exit']=True;return result
        e=json.loads(p.read_text());assert e['returncode']==0 and e['complete_sha256']==sha(root/'complete.json')
    result.update(verified_complete=True,complete=complete,selection=json.loads((root/'selection_frozen.json').read_text()))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/react_refinement/run_001'));a=p.parse_args()
    try:print(json.dumps(inspect(a.root),ensure_ascii=False,indent=2))
    except (OSError,ValueError,KeyError,AssertionError) as e:print(json.dumps(dict(verified_complete=False,error=type(e).__name__+': '+str(e))));raise SystemExit(1)


if __name__=='__main__':main()
