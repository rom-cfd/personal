"""Read-only standard-library observer for the detached gamma sweep."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/gamma_sweep/run_001')
    parser.add_argument('--wait', type=float, default=0.)
    args = parser.parse_args()
    if not 0 <= args.wait <= 50:
        raise ValueError('Observer waits must be between0 and50 seconds')
    if args.wait:
        time.sleep(args.wait)
    root = Path(args.root); rows = []
    if (root / 'execution.jsonl').exists():
        for line in (root / 'execution.jsonl').read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    budget = json.loads((root / 'budget.json').read_text()) if (root / 'budget.json').exists() else {}
    elapsed = max(0., (time.time()-budget['started_unix'])/60) if budget else 0.
    members = {(r['gamma'],r['family'],r['seed']) for r in rows if r.get('phase')=='member_finished'}
    capped = sum(r.get('reason')=='member_time_limit' for r in rows if r.get('phase')=='member_finished')
    evaluations = {(r['gamma'],r['method'],r['split'],r['mode']) for r in rows if r.get('phase')=='evaluation_finished'}
    pages = {(r['split'],r['mode'],r['trajectory']) for r in rows if r.get('phase')=='report_page_finished'}
    latest = rows[-1] if rows else {}; phase = latest.get('phase','prepared')
    message = f'γ 스윕 {elapsed:.1f}분 · 신규 종료 {len(members)}/200 · 시간상한 {capped}'
    if phase in ('member_started','member_finished','train_progress'):
        message += f" · γ={latest['gamma']:g} {latest['family']} seed{latest['seed']}"
        if 'update' in latest:
            message += f" {latest['update']:,} update"
    elif phase == 'evaluation_finished':
        message += f" · 평가 {len(evaluations)}/168 · γ={latest['gamma']:g} {latest['split']} {latest['method']} {latest['mode']}"
    elif phase == 'report_page_finished':
        message += f" · 그림·NRMSE {len(pages)}/72 · {latest['split']} {latest['mode']} {latest['page']}/18"
    else:
        message += ' · '+phase
    attempts = sorted((root/'process_launches').glob('attempt_*'))
    exit_info = json.loads((attempts[-1]/'exit.json').read_text()) if attempts and (attempts[-1]/'exit.json').exists() else None
    status = 'running' if budget else 'prepared'
    if exit_info is not None:
        status = 'complete' if exit_info['returncode']==0 and (root/'FINAL_ARTIFACTS.json').exists() else 'failed'
        message += ' · process '+status
    print(json.dumps(dict(message=message,status=status,exit=exit_info,phase=phase,
        elapsed_minutes=elapsed,finished_members=len(members),evaluations=len(evaluations),pages=len(pages)),ensure_ascii=False))


if __name__ == '__main__':
    main()
