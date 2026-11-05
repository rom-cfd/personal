"""Read-only stdlib observer; no TensorFlow or numerical work."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='results/gamma_long/run_001');parser.add_argument('--wait',type=float,default=0.)
    args=parser.parse_args()
    if not 0<=args.wait<=50:raise ValueError('Wait between0 and50 seconds')
    if args.wait:time.sleep(args.wait)
    root=Path(args.root);rows=[]
    if (root/'execution.jsonl').exists():
        for line in (root/'execution.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
    budget=json.loads((root/'budget.json').read_text()) if (root/'budget.json').exists() else {}
    elapsed=(time.time()-budget['started_unix'])/60 if budget else 0.
    endpoints={(r['gamma'],r['family'],r['seed']):r for r in rows if r.get('phase')=='member_finished'}
    flat=sum(r['reason']=='plateau' for r in endpoints.values())
    evaluations={(r['gamma'],r['method'],r['split'],r['mode']) for r in rows if r.get('phase')=='evaluation_finished'}
    pages={(r['split'],r['mode'],r['stage'],r['trajectory']) for r in rows if r.get('phase')=='report_page_finished'}
    latest=rows[-1] if rows else {};phase=latest.get('phase','prepared')
    message=f'γ 장기 {elapsed:.1f}분 · 종료 {len(endpoints)}/240 · 평탄 {flat}/240'
    if phase in ('member_started','member_finished','train_progress'):
        message+=f" · γ={latest['gamma']:g} {latest['family']} seed{latest['seed']}"
        if 'update' in latest:message+=f" {latest['update']:,} update"
    elif phase=='bootstrap_member_finished':
        count=len({(r['family'],r['seed']) for r in rows if r.get('phase')==phase});message+=f' · 누락상태 재생성 {count}/20'
    elif phase=='evaluation_finished':message+=f" · 평가 {len(evaluations)}/168 · {latest['split']} γ={latest['gamma']:g} {latest['method']} {latest['mode']}"
    elif phase=='report_page_finished':message+=f" · 그림·NRMSE {len(pages)}/144 · {latest['split']} {latest['mode']} {latest['stage']}"
    else:message+=' · '+phase
    attempts=sorted((root/'process_launches').glob('attempt_*'))
    end=json.loads((attempts[-1]/'exit.json').read_text()) if attempts and (attempts[-1]/'exit.json').exists() else None
    status='running' if budget else 'prepared'
    if end is not None:
        status='complete' if end['returncode']==0 and (root/'FINAL_ARTIFACTS.json').exists() else 'failed';message+=' · process '+status
    print(json.dumps(dict(message=message,status=status,exit=end,phase=phase,elapsed_minutes=elapsed,finished_members=len(endpoints),plateau_members=flat),ensure_ascii=False))


if __name__=='__main__':main()
