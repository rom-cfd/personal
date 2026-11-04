"""Detached watchdog; fixed original wall clock and one worker at a time."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from scripts.run_weight_long_supervisor import write_exclusive, run_supervised


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/gamma_sweep.json')
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args(); cfg = json.loads(Path(args.config).read_text()); root = Path(cfg['output'])
    attempt = Path(args.attempt)
    if not attempt.resolve().is_relative_to(root.resolve()) or not (root / 'run.json').exists():
        raise ValueError('Prepare first; use a unique launch directory under this run')
    expected = dict(TF_ENABLE_ONEDNN_OPTS='0', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    if any(os.environ.get(k) != v for k, v in expected.items()):
        raise ValueError('Use the frozen CPU/thread environment')
    with (root / 'process.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / 'budget.json'
        if path.exists():
            budget = json.loads(path.read_text())
            if budget['seconds'] != cfg['total_budget_seconds']:
                raise ValueError('Cannot change/reset the original clock')
        else:
            start = time.time()
            budget = dict(started_unix=start, seconds=cfg['total_budget_seconds'],
                expires_unix=start + cfg['total_budget_seconds'],
                training_expires_unix=start + cfg['training_budget_seconds'])
            write_exclusive(path, budget)
        command = [sys.executable, '-u', '-m', 'kinetics.gamma_sweep_study', '--config', args.config, '--phase', 'run', '--resume']
        write_exclusive(attempt / 'supervisor.json', dict(pid=os.getpid(), process_session=os.getsid(0),
            command=command, budget=budget, detached=True, role='watchdog only; no TensorFlow'))
        result = run_supervised(command, budget['expires_unix']) if time.time() < budget['expires_unix'] else dict(returncode=124, timed_out=True, finished_unix=time.time())
        write_exclusive(attempt / 'exit.json', result)
        return 124 if result['timed_out'] else result['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
