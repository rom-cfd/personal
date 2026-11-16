"""Read only small status JSON while training; hash outputs only after exits."""
from pathlib import Path
import argparse,hashlib,json

ROOT=Path('results/react_stepper_refinement/run_001')
FOLLOW=Path('/workspace/react_stepper_refinement_followup/run_001')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(p):return json.loads(p.read_text()) if p.exists() else None

def inspect(root=ROOT,follow=FOLLOW,external=True):
    result=dict(root=str(root),verified_complete=False)
    for name in ['status','worker_started','failure','exit']:
        value=read(root/(name+'.json'))
        if value is not None:result[name]=value
    for name in ['promotion_frozen','final_training','reconstruction_controls','selection_frozen']:
        value=read(root/(name+'.json'))
        if value is not None:
            if name=='promotion_frozen':result[name]=dict(status=value['status'],promoted=value['promoted'])
            elif name=='selection_frozen':result[name]=dict(status=value['status'],combined_choice=value['combined_choice'])
            else:result[name]={key:{k:v for k,v in record.items() if k in ['selected_new_update','last_new_update','selected_K','final_K','active_seconds','last_lr','stop_reason']} for key,record in value.items()}
    outer=read(follow/'exit.json') if external else None
    if outer is not None:result['supervisor_exit']=outer
    complete=read(root/'complete.json')
    if not complete or not result.get('exit') or result['exit'].get('returncode')!=0:return result
    if external and (not outer or outer.get('returncode')!=0):return result
    assert complete['status']=='passed'
    assert result['exit']['complete_sha256']==sha(root/'complete.json')
    if external:assert outer['complete_sha256']==sha(root/'complete.json')
    verification=read(root/'verification.json');report=read(root/'report/complete.json')
    assert verification['status']==report['status']=='passed'
    assert complete['verification_sha256']==sha(root/'verification.json')
    assert complete['report_complete_sha256']==sha(root/'report/complete.json')
    assert complete['selection_sha256']==sha(root/'selection_frozen.json')
    for name,expected in complete['output_sha256'].items():assert sha(root/name)==expected,name
    for name,expected in report['output_sha256'].items():assert sha(root/'report'/name)==expected,name
    for name in ['operator_profiles.pdf','combined_profiles.pdf','ignition_zoom.pdf','final_weights.pdf']:
        assert (root/'report'/name).is_file()
    result.update(verified_complete=True,complete={k:v for k,v in complete.items() if k!='output_sha256'},
        verification=verification,PDF_pages=report['PDF_pages'])
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--follow',type=Path,default=FOLLOW)
    a=p.parse_args();print(json.dumps(inspect(a.root,a.follow),ensure_ascii=False,indent=2))
