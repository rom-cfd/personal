"""Read small progress files; verify frozen artifacts only after successful exit."""
from pathlib import Path
import argparse
import json
from kinetics.flex_data import sha, FOLLOW


def inspect(root,external=True):
    root=Path(root);result=dict(root=str(root),verified_complete=False)
    for name in ['status','worker_started','final_training','exit','failure']:
        p=root/(name+'.json')
        if p.exists():result[name]=json.loads(p.read_text())
    if result.get('exit',{}).get('returncode')!=0:return result
    c=json.loads((root/'complete.json').read_text())
    v=json.loads((root/'verification.json').read_text())
    r=json.loads((root/'report/complete.json').read_text())
    assert c['status']==v['status']==r['status']=='passed'
    assert c['verification_sha256']==sha(root/'verification.json')
    assert c['report_complete_sha256']==sha(root/'report/complete.json')
    assert all(sha(root/p)==h for p,h in v['output_sha256'].items())
    assert all(sha(root/'report'/p)==h for p,h in r['output_sha256'].items())
    for name in ['all_profiles.pdf','ignition_zoom.pdf','summary.csv','paper_RMSE_comparison.csv','coverage.png','loss_and_lr.png']:
        assert (root/'report'/name).is_file(),name
    if external:
        e=FOLLOW/'exit.json'
        if not e.exists():result['awaiting_supervisor_exit']=True;return result
        e=json.loads(e.read_text());assert e['returncode']==0 and e['complete_sha256']==sha(root/'complete.json')
    result.update(verified_complete=True,complete=c,report_verification=json.loads((root/'report/verification.json').read_text()))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('results/flex_deeponet_reproduction/run_001'))
    args=parser.parse_args()
    try:print(json.dumps(inspect(args.root),ensure_ascii=False,indent=2))
    except (OSError,ValueError,KeyError,AssertionError) as e:
        print(json.dumps(dict(verified_complete=False,error=type(e).__name__+': '+str(e))));raise SystemExit(1)
