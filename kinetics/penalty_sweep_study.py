"""Serial matched10k lambda1/100/1000 sweep; preserve all previous studies."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from importlib.metadata import version

import numpy as np
from .ann_deeponet_study import load_inputs
from .data import mechanism, SPECIES
from .p_only_training import sha256, write_json
from .soft_constraint_training import FAMILIES
from .soft_constraint_study import train as train_weight, event
from .study import tf


def weight_root(root, weight):
    return Path(root) / f'lambda_{int(weight)}'


def weight_config(cfg, weight):
    return {**cfg, 'mu': float(weight), 'output': str(weight_root(cfg['output'], weight))}


def validate_config(cfg):
    if (cfg['lambdas'] != [1., 100., 1000.] or cfg['families'] != list(FAMILIES)
            or cfg['updates_per_member'] != 10000 or cfg['modes'] != ['raw', 'repaired']
            or cfg['evaluation_splits'] != ['test'] or cfg['test_used_for_selection']
            or cfg['inference_changed']):
        raise ValueError('Keep the requested three weights,120 independent10k members and original inference')
    if not 0 < cfg['training_wall_safety_seconds'] < cfg['pipeline_wall_safety_seconds']:
        raise ValueError('Training guard must be smaller than pipeline guard')


def prepare(cfg, resume=False):
    validate_config(cfg)
    if subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != 'bw':
        raise ValueError('Use the existing bw checkout')
    if os.environ.get('TF_ENABLE_ONEDNN_OPTS') != '0':
        raise ValueError('Set oneDNN0 before importing TensorFlow')
    root = Path(cfg['output']); namespace = Path('results/penalty_weight_sweep').resolve()
    if root.resolve() == namespace or not root.resolve().is_relative_to(namespace):
        raise ValueError('Use a new output below results/penalty_weight_sweep')
    origin = Path(cfg['origin_mu10']); old = json.loads((origin / 'run.json').read_text())
    base = json.loads(Path(cfg['base_config']).read_text())
    if base != old['base_config']:
        raise ValueError('The lambda controls must use the same base configuration')
    data, pairs, fingerprint = load_inputs(base)
    for key in ('pair_sha256', 'dense_sha256', 'ann_members', 'reference_ignition'):
        if fingerprint[key] != old['fingerprint'][key]:
            raise ValueError(f'Control input changed: {key}')
    gas = mechanism(); elements = list(map(str, data['elements']))
    matrix = np.array([[gas.n_atoms(s, e) * gas.atomic_weight(e) / gas.molecular_weights[k]
                        for k, s in enumerate(SPECIES)] for e in elements])
    if elements != ['H', 'O', 'N']:
        raise ValueError('Expected H/O/N order')
    np.testing.assert_array_equal(matrix, data['element_matrix'])
    records = json.loads((origin / 'final_members.json').read_text())
    if len(records) != 40:
        raise ValueError('Need the frozen40-member lambda10 control')
    for r in records:
        if r['updates'] != 10000 or r['mu'] != 10:
            raise ValueError('Only matched10k controls may enter the sweep')
        cp = Path(r['checkpoint']); marker = json.loads((cp / 'complete.json').read_text())
        for name, expected in marker['output_sha256'].items():
            if sha256(cp / name) != expected:
                raise ValueError('Frozen lambda10 checkpoint changed')
    for family, members in old['baseline_members'].items():
        if [r['seed'] for r in members] != base['seeds']:
            raise ValueError('Baseline seed order changed')
        for r in members:
            if r['updates'] != 10000 or sha256(r['path'] + '.keras') != r['sha256']:
                raise ValueError('Baseline model changed')
    source_files = sorted(Path('kinetics').glob('*.py')) + sorted(Path('scripts').glob('*.py'))
    source_files += [Path('references/user_nrmse/util.py'), Path('references/soft_constraint/step1.py'),
                     Path('references/soft_constraint/step2.py')]
    origin_files = [origin / n for n in ('run.json', 'final_members.json', 'baseline_traces.json',
                    'member_diagnostics/loss_components.csv', 'member_diagnostics/complete.json')]
    snapshot = dict(config=cfg, base_config=base, fingerprint=fingerprint,
                    model_configs=old['model_configs'], element_matrix=matrix.tolist(), elements=elements,
                    baseline_members=old['baseline_members'], lambda10_members=records,
                    origin_sha256={str(p): sha256(p) for p in origin_files},
                    source_sha256={str(p): sha256(p) for p in source_files},
                    authorization='User requested lambda1,100,1000 training and results; matched10000 updates per member',
                    new_members=120, authorized_updates=1200000,
                    initialization='Fresh paired seeds11..110; no fine-tuning from lambda10',
                    comparison='lambda0,1,10,100,1000 at10000 updates; no test-based selection')
    if root.exists():
        if not resume:
            raise FileExistsError('Use a new output or identical --resume')
        if json.loads((root / 'run.json').read_text()) != snapshot:
            raise ValueError('Sweep resume provenance changed')
    else:
        previous = {str(p): sha256(p) for p in Path('results').rglob('*') if p.is_file()}
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / 'run.json', snapshot)
        write_json(root / 'original_results_sha256.json', previous)
        write_json(root / 'environment.json', dict(python=platform.python_version(),
            versions={n: version(n) for n in ('tensorflow_cpu', 'keras', 'numpy', 'cantera', 'scipy', 'matplotlib')},
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            git_status=subprocess.check_output(['git', 'status', '--short'], text=True),
            tensorflow_threads=dict(intra=tf.config.threading.get_intra_op_parallelism_threads(),
                                    inter=tf.config.threading.get_inter_op_parallelism_threads()),
            thread_environment={k: os.environ.get(k) for k in ('TF_ENABLE_ONEDNN_OPTS', 'OMP_NUM_THREADS',
                                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')},
            cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip()))
        for weight in cfg['lambdas']:
            child = weight_root(root, weight); child.mkdir()
            write_json(child / 'run.json', dict(sweep_run_sha256=sha256(root / 'run.json'),
                lambda_value=weight, config=weight_config(cfg, weight), base_config=base))
    return root, data, pairs, snapshot


def deadlines(root, cfg):
    path = Path(root) / 'budget.json'
    if path.exists():
        budget = json.loads(path.read_text())
    else:
        start = time.time()
        budget = dict(started_unix=start, authorized_updates=1200000,
            training_expires_unix=start + cfg['training_wall_safety_seconds'],
            expires_unix=start + cfg['pipeline_wall_safety_seconds'],
            note='Fixed120x10000 updates; wall guards do not authorize more updates or a new resume clock')
        write_json(path, budget)
    now, clock = time.time(), time.perf_counter()
    return clock + max(0, budget['training_expires_unix'] - now), clock + max(0, budget['expires_unix'] - now)


def train(root, cfg, snapshot, pairs, data, deadline):
    if (root / 'final_members.json').exists():
        return
    all_records = []
    for weight in cfg['lambdas']:
        before = time.perf_counter(); child = weight_root(root, weight)
        event(root, phase='weight_training_started', lambda_value=weight)
        train_weight(child, weight_config(cfg, weight), snapshot, pairs, data, deadline)
        records = json.loads((child / 'final_members.json').read_text())
        if len(records) != 40 or any(r['mu'] != weight or r['updates'] != 10000 for r in records):
            raise ValueError('Incomplete or incorrectly weighted member group')
        all_records.extend(records)
        event(root, phase='weight_training_finished', lambda_value=weight, members=40,
              seconds=time.perf_counter() - before)
    write_json(root / 'final_members.json', all_records)
    event(root, phase='training_finished', members=120, updates=sum(r['updates'] for r in all_records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/penalty_weight_sweep.json')
    parser.add_argument('--phase', choices=('prepare', 'run'), default='prepare')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); cfg = json.loads(Path(args.config).read_text())
    root, data, pairs, snapshot = prepare(cfg, args.resume)
    if args.phase == 'prepare':
        print('PENALTY_SWEEP_PREPARED', root, flush=True); return
    train_deadline, total_deadline = deadlines(root, cfg)
    before = time.perf_counter(); train(root, cfg, snapshot, pairs, data, train_deadline)
    event(root, phase='training_wall_finished', seconds=time.perf_counter() - before)
    from .penalty_sweep_evaluation import evaluate
    before = time.perf_counter(); evaluate(root, cfg, snapshot, data, pairs, total_deadline - 900)
    event(root, phase='evaluation_wall_finished', seconds=time.perf_counter() - before)
    from scripts.report_penalty_sweep import report
    report(root, cfg, snapshot, data, total_deadline)
    event(root, phase='pipeline_finished')


if __name__ == '__main__':
    main()
