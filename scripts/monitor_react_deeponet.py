"""Read-only progress/completion checks; never starts or resumes an experiment."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('results/react_deeponet_reconstruction/run_001'))
    parser.add_argument('--followup', type=Path, default=Path('/workspace/react_deeponet_followup/run_001'))
    args = parser.parse_args()
    root, follow = args.root, args.followup
    result = dict(root=str(root), status=read(root/'status.json'),
                  worker_exit=read(root/'exit.json'), controller_exit=read(follow/'exit.json'))
    failure = read(root/'failure.json') or read(follow/'failure.json')
    if failure:
        result.update(classification='failed', failure=failure)
    elif result['worker_exit'] and result['worker_exit']['returncode'] != 0:
        result['classification'] = 'failed'
    elif (root/'complete.json').exists():
        complete = read(root/'complete.json')
        verification = read(root/'verification.json')
        report = read(root/'report/complete.json')
        passed = bool(result['worker_exit'] and result['worker_exit']['returncode'] == 0
                      and result['controller_exit'] and result['controller_exit']['returncode'] == 0
                      and complete['status'] == 'passed'
                      and verification and verification['status'] == 'passed'
                      and report and report['status'] == 'passed')
        if passed:
            passed = (digest(root/'verification.json') == complete['verification_sha256']
                      and digest(root/'report/complete.json') == complete['report_complete_sha256'])
        if passed:
            # Expensive artifact reads only after both processes have exited.
            passed = all((root/'report'/name).is_file() and digest(root/'report'/name) == expected
                         for name, expected in report['output_sha256'].items())
            passed = passed and (root/'report/coverage.png').is_file() and (root/'report/loss_and_lr.pdf').is_file()
        result['classification'] = ('complete' if complete['training_update_target_completed'] else 'complete_time_capped') if passed else 'completion_pending_verification'
        result['training'] = read(root/'selection_frozen.json')
        if passed:
            result['summary'] = list(csv.DictReader((root/'report/test_summary.csv').open()))
    else:
        started = read(root/'worker_started.json')
        if started:
            try:
                os.kill(started['pid'], 0)
                result['classification'] = 'running'
            except ProcessLookupError:
                result['classification'] = 'worker_missing_without_exit_record'
        else:
            result['classification'] = 'not_started'
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
