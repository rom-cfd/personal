"""Standard-library-only supervisor for a detached six-hour cloud run."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_exclusive(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def run_supervised(command, expires_unix, poll_seconds=5.):
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
    started = time.time(); timed_out = False
    while process.poll() is None:
        remaining = expires_unix - time.time()
        if remaining <= 0:
            timed_out = True; process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            break
        time.sleep(min(poll_seconds, remaining))
    return dict(child_pid=process.pid, started_unix=started, finished_unix=time.time(),
                returncode=process.returncode, timed_out=timed_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/weight_long_study.json')
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args(); cfg = json.loads(Path(args.config).read_text()); root = Path(cfg['output'])
    attempt = Path(args.attempt)
    if not attempt.resolve().is_relative_to(root.resolve()) or not (root / 'run.json').exists():
        raise ValueError('Prepare the study and use a unique launch directory below its output')
    if cfg['total_budget_seconds'] != 21600 or cfg['evaluation_reserve_seconds'] != 3600:
        raise ValueError('Preserve the approved six-hour plan')
    expected = dict(TF_ENABLE_ONEDNN_OPTS='0', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    if any(os.environ.get(k) != v for k, v in expected.items()):
        raise ValueError('Use the frozen CPU/thread environment')
    with (root / 'process.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another supervisor already owns this run') from None
        path = root / 'budget.json'
        if path.exists():
            budget = json.loads(path.read_text())
        else:
            start = time.time()
            budget = dict(started_unix=start, seconds=21600, expires_unix=start + 21600,
                          training_expires_unix=start + 18000)
            write_exclusive(path, budget)
        if budget['seconds'] != 21600:
            raise ValueError('Cannot reset the original approved clock')
        command = [sys.executable, '-u', '-m', 'kinetics.weight_long_study', '--config', args.config,
                   '--phase', 'run', '--resume']
        write_exclusive(attempt / 'supervisor.json', dict(pid=os.getpid(), process_session=os.getsid(0),
            command=command, budget=budget, detached=True, role='watchdog only; no TensorFlow imported'))
        if time.time() >= budget['expires_unix']:
            result = dict(returncode=124, timed_out=True, finished_unix=time.time(), note='Original clock already expired')
        else:
            result = run_supervised(command, budget['expires_unix'])
        write_exclusive(attempt / 'exit.json', result)
        return 124 if result['timed_out'] else result['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
