"""Read-only progress observer; not part of the frozen training source."""
import argparse
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--wait', type=float, default=0)
args = parser.parse_args()
if not 0 <= args.wait <= 50:
    raise ValueError('Keep each observer wait under one minute')
time.sleep(args.wait)
root = Path(__file__).resolve().parents[1]
budget = json.loads((root / 'budget.json').read_text()); rows = []
if (root / 'execution.jsonl').exists():
    for line in (root / 'execution.jsonl').read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
endpoints = {(r['lambda_value'], r['family'], r['seed']): r for r in rows if r.get('phase') == 'member_finished'}
flat = sum(r['reason'] == 'joint_plateau' for r in endpoints.values())
elapsed = (time.time() - budget['started_unix']) / 60
e = rows[-1] if rows else {}; phase = e.get('phase', 'initializing')
message = f'λ 장기 {elapsed:.1f}분 · 종료 {len(endpoints)}/120 · 공동 평탄 {flat}/120'
if phase in ('member_started', 'member_finished', 'train_progress'):
    message += f" · λ={int(e['lambda_value'])} {e['family']} seed{e['seed']} {e['update']:,} update"
    if phase == 'member_finished':
        message += ' · ' + e['reason']
    elif phase == 'train_progress' and e.get('loss'):
        message += f" · train MSE {e['loss']['data_loss']:.3e} · 원소L1 {e['loss']['element_loss']:.3e}"
elif phase == 'evaluation_finished':
    n = len({(r['lambda_value'], r['method'], r['mode']) for r in rows if r.get('phase') == phase})
    message += f" · 평가 {n}/42 · λ={int(e['lambda_value'])} {e['method']} {e['mode']}"
elif phase == 'report_page_finished':
    n = len({(r['lambda_value'], r['mode'], r['trajectory']) for r in rows if r.get('phase') == phase})
    message += f" · 그림·NRMSE {n}/108 · λ={int(e['lambda_value'])} {e['mode']} {e['page']}/18"
else:
    message += ' · ' + phase
attempt = sorted((root / 'process_launches').glob('attempt_*'))[-1]
status = 'running'; exit_info = None
if (attempt / 'exit.json').exists():
    exit_info = json.loads((attempt / 'exit.json').read_text())
    status = 'complete' if exit_info['returncode'] == 0 and (root / 'FINAL_ARTIFACTS.json').exists() else 'failed'
    message += ' · process ' + status
print(json.dumps(dict(message=message, status=status, exit=exit_info, phase=phase,
    elapsed_minutes=elapsed, finished_members=len(endpoints), joint_plateau_members=flat)))
