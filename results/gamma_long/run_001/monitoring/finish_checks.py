"""Detached read-only wait, then saved-result numerical checks and summaries."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('results/gamma_long/run_001')
attempt = sorted((ROOT / 'process_launches').glob('attempt_*'))[-1]
while not (attempt / 'exit.json').exists():
    time.sleep(30)
end = json.loads((attempt / 'exit.json').read_text())
if end['returncode'] or not (ROOT / 'FINAL_ARTIFACTS.json').exists():
    raise SystemExit('Production run did not complete: ' + json.dumps(end))
for name in ('postrun_verify.py', 'compact_summary.py'):
    subprocess.run([sys.executable, '-u', str(ROOT / 'monitoring' / name)], check=True)
with (ROOT / 'postprocessing_complete.json').open('x') as f:
    json.dump(dict(status='passed', finished_unix=time.time(), production_exit=end), f, indent=2)
print('INDEPENDENT_CHECKS_AND_SUMMARIES_COMPLETE', flush=True)
