"""Read progress cheaply; validate completion evidence before reporting success."""
import argparse,json,hashlib
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def inspect(root):
    root=Path(root);result={'root':str(root),'verified_complete':False}
    for name in ['status','worker_started','exit','failure']:
        p=root/(name+'.json')
        if p.exists():result[name]=json.loads(p.read_text())
    if result.get('exit',{}).get('returncode')!=0:return result
    for name in ['complete.json','verification.json','report/complete.json']:
        p=root/name
        if not p.is_file():result['missing_evidence']=name;return result
    complete=json.loads((root/'complete.json').read_text());verification=json.loads((root/'verification.json').read_text());report=json.loads((root/'report/complete.json').read_text())
    assert complete['status']==verification['status']==report['status']=='passed'
    assert sha(root/'verification.json')==complete['verification_sha256']
    assert sha(root/'report/complete.json')==complete['report_complete_sha256']
    assert all(sha(root/p)==h for p,h in verification['output_sha256'].items())
    assert all(sha(root/'report'/p)==h for p,h in report['output_sha256'].items())
    for name in ['all_profiles.pdf','ignition_zoom.pdf','paper_representatives.pdf']:
        assert (root/'report'/name).is_file()
    result.update(verified_complete=True,complete=complete,selection=json.loads((root/'selection_frozen.json').read_text()))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/react_deeponet_rebuild/run_001'));a=p.parse_args()
    try:print(json.dumps(inspect(a.root),ensure_ascii=False,indent=2))
    except (OSError,ValueError,AssertionError,KeyError) as e:
        print(json.dumps({'verified_complete':False,'verification_error':type(e).__name__+': '+str(e)},ensure_ascii=False));raise SystemExit(1)


if __name__=='__main__':main()
