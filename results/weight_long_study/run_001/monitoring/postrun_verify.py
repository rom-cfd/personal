"""Independent checks of saved reports; run only after the supervisor exits."""
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import zipfile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np

ROOT = Path('results/weight_long_study/run_001')
started = time.time()
final = json.loads((ROOT / 'FINAL_ARTIFACTS.json').read_text())
report = Path(final['report'])
cfg = json.loads(Path('configs/weight_long_study.json').read_text())
snapshot = json.loads((report / 'run_snapshot.json').read_text())
attempts = sorted((ROOT / 'process_launches').glob('attempt_*'))
exit_info = json.loads((attempts[-1] / 'exit.json').read_text())
assert exit_info['returncode'] == 0 and not exit_info.get('timed_out', False)
assert time.time() < json.loads((ROOT / 'budget.json').read_text())['expires_unix']


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def close(actual, expected):
    if not math.isfinite(expected):
        assert actual == '' or math.isnan(float(actual)), (actual, expected)
    else:
        assert actual != '' and math.isclose(float(actual), float(expected), rel_tol=2e-12, abs_tol=2e-14), (actual, expected)


parsed = ast.parse(Path('references/user_nrmse/util.py').read_text())
fn = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == '_nrmse_1d')
scope = {'np': np}
exec(compile(ast.Module(body=[fn], type_ignores=[]), 'references/user_nrmse/util.py', 'exec'), scope)
user_nrmse = scope['_nrmse_1d']
with np.load('results/long10ms_data/trajectories.npz') as f:
    truth = np.asarray(f['states'], dtype=np.float64)
variables = ['T', 'H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'N2']
teams = ['ann10', 'p_only10', 'ann_p_only20', 'tp10', 'ann_tp20', 'tp_phi10', 'ann_tp_phi20']
baseline_root = Path(snapshot['origin_snapshot']['config']['origin_mu10'])
baseline = json.loads((baseline_root / 'baseline_traces.json').read_text())
scores_checked = 0
incomplete_checked = 0
undefined_checked = 0
summary_checked = 0
frames = {}


def check_summaries(folder, rows):
    global summary_checked
    methods = list(dict.fromkeys(r['method'] for r in rows))
    complete = {m: {int(r['trajectory']) for r in rows if r['method'] == m and r['complete'] == 'True'} for m in methods}
    common = sorted(set.intersection(*(complete[m] for m in methods)))
    within = {team: sorted(set.intersection(*(complete[m] for m in methods if next(r['team'] for r in rows if r['method'] == m) == team))) for team in teams}
    lookup = {}
    for r in rows:
        lookup.setdefault((r['method'], r['variable']), []).append(r)
    for filename, per_team in [('common_IC_NRMSE_summary.csv', False), ('within_team_NRMSE_summary.csv', True)]:
        summaries = read_csv(folder / filename)
        assert len(summaries) == len(methods) * 10
        for r in summaries:
            ids = within[r['team']] if per_team else common
            assert json.loads(r['common_ICs']) == ids
            assert int(r['common_IC_count']) == len(ids)
            part = lookup[(r['method'], r['variable'])]
            values = np.array([float(p['nrmse']) for p in part if int(p['trajectory']) in ids and p['nrmse'] != ''])
            assert int(r['valid_nrmse_IC_count']) == len(values)
            assert int(r['completed_all_test_ICs']) == sum(p['complete'] == 'True' for p in part)
            for field, value in [('median_nrmse', np.median(values) if len(values) else np.nan), ('mean_nrmse', np.mean(values) if len(values) else np.nan), ('std_nrmse', np.std(values) if len(values) else np.nan), ('q25_nrmse', np.quantile(values, .25) if len(values) else np.nan), ('q75_nrmse', np.quantile(values, .75) if len(values) else np.nan)]:
                close(r[field], value)
            summary_checked += 1
    return {'common_ICs': common, 'within_team_common_ICs': within}


cohorts = {}
for weight in (1, 100, 1000):
    cohorts[str(weight)] = {}
    for mode in ('repaired', 'raw'):
        folder = report / f'lambda_{weight}' / mode
        rows = read_csv(folder / 'per_IC_NRMSE.csv')
        assert len(rows) == 21 * 18 * 10
        frames[(weight, mode)] = rows
        lookup = {(r['method'], int(r['trajectory']), r['variable']): r for r in rows}
        for team in teams:
            for stage in ('baseline', 'short', 'long'):
                if stage == 'baseline':
                    path = Path(baseline[f'{team}_{mode}']['path'])
                else:
                    origin = Path(cfg['origin']) if stage == 'short' else ROOT
                    markers = list((origin / f'lambda_{weight}' / 'evaluations' / f'{team}_{mode}_test').glob('*/complete.json'))
                    assert len(markers) == 1
                    path = markers[0].parent / 'traces.npz'
                with np.load(path) as traces:
                    for i in sorted({int(r['trajectory']) for r in rows}):
                        prediction = np.asarray(traces[f'trajectory_{i}'], dtype=np.float64)
                        complete = len(prediction) == len(truth[i])
                        np.testing.assert_array_equal(prediction[0], truth[i, 0])
                        for k, variable in enumerate(variables):
                            r = lookup[(f'{stage}_{team}', i, variable)]
                            assert int(r['points']) == len(prediction)
                            assert (r['complete'] == 'True') == complete
                            ref = truth[i, :, k]
                            close(r['reference_rms'], np.sqrt(np.mean(ref ** 2)))
                            close(r['prefix_rmse'], np.sqrt(np.mean((ref[:len(prediction)] - prediction[:, k]) ** 2)))
                            if not complete:
                                assert r['nrmse'] == '' and r['full_rmse'] == ''
                                incomplete_checked += 1
                                continue
                            close(r['full_rmse'], np.sqrt(np.mean((ref - prediction[:, k]) ** 2)))
                            try:
                                expected = user_nrmse(ref, prediction[:, k])
                            except ValueError:
                                assert r['nrmse'] == ''
                                undefined_checked += 1
                            else:
                                close(r['nrmse'], expected)
                                scores_checked += 1
        cohorts[str(weight)][mode] = check_summaries(folder, rows)

cross = {}
cross_temperature = {}
for mode in ('repaired', 'raw'):
    folder = report / f'cross_weight_{mode}'
    rows = read_csv(folder / 'per_IC_NRMSE.csv')
    assert len(rows) == 42 * 18 * 10
    expected = {}
    for weight in (1, 100, 1000):
        for r in frames[(weight, mode)]:
            if r['stage'] != 'baseline':
                expected[(f"lambda{weight}_{r['method']}", r['trajectory'], r['variable'])] = r
    for r in rows:
        source = expected[(r['method'], r['trajectory'], r['variable'])]
        assert all(r[k] == source[k] for k in source if k != 'method')
    cross[mode] = check_summaries(folder, rows)
    cross_temperature[mode] = [r for r in read_csv(folder / 'within_team_NRMSE_summary.csv') if r['variable'] == 'T']

manifest = json.loads((report / 'complete.json').read_text())['output_sha256']
assert digest(report / 'complete.json') == final['report_complete_sha256']
for relative, expected in manifest.items():
    assert digest(report / relative) == expected, relative
pdfs = {}
for p in sorted(report.rglob('*.pdf')):
    count = len(re.findall(rb'/Type\s*/Page\b', p.read_bytes()))
    expected = 36 if p.name == 'profiles_and_NRMSE.pdf' else 7 if p.name == 'within_team_NRMSE.pdf' else 1
    assert count == expected, (str(p), count, expected)
    pdfs[str(p.relative_to(report))] = count
archives = {}
for p in sorted(report.glob('*.zip')):
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        archives[p.name] = {'bytes': p.stat().st_size, 'files': len(z.namelist())}
assert len(archives) == 6
records = json.loads((ROOT / 'final_members.json').read_text())
assert len(records) == 120
for r in records:
    with np.load(Path(r['checkpoint']) / 'adam.npz') as f:
        assert int(f['v0']) == r['updates'] == r['optimizer_iteration']

main_verification = json.loads((report / 'verification.json').read_text())
check = {'report_files_SHA256_verified': len(manifest), 'actual_Adam_iterations_independently_verified': 120,
         'exact_user_util_NRMSE_scores_checked': scores_checked, 'incomplete_variable_rows_without_full_scores': incomplete_checked,
         'undefined_variable_rows': undefined_checked, 'summary_rows_independently_checked': summary_checked,
         'pdf_page_objects': pdfs, 'zip_CRC_checked': archives, 'cohorts': cohorts, 'cross_weight': cross,
         'main_report_verification': main_verification, 'seconds': time.time() - started}
(ROOT / 'postrun_verification.json').write_text(json.dumps(check, ensure_ascii=False, indent=2) + '\n')
analysis = {'cross_weight_temperature_within_team': cross_temperature, 'final_artifacts': final,
            'training': {str(w): {'members': 40, 'updates_min': min(r['updates'] for r in records if r['mu'] == w),
                'updates_max': max(r['updates'] for r in records if r['mu'] == w),
                'updates_median': float(np.median([r['updates'] for r in records if r['mu'] == w])),
                'joint_plateau': sum(r['reason'] == 'joint_plateau' for r in records if r['mu'] == w),
                'under_50k': sum(r['updates'] < 50000 for r in records if r['mu'] == w),
                'under_60k': sum(r['updates'] < 60000 for r in records if r['mu'] == w)} for w in (1, 100, 1000)}}
analysis['per_weight_within_team_comparisons'] = {}
all_temperature_above_baseline = []
for weight in (1, 100, 1000):
    analysis['per_weight_within_team_comparisons'][str(weight)] = {}
    for mode in ('repaired', 'raw'):
        rows = read_csv(report / f'lambda_{weight}' / mode / 'within_team_NRMSE_summary.csv')
        comparisons = []
        for team in teams:
            for variable in variables:
                group = {r['stage']: r for r in rows if r['team'] == team and r['variable'] == variable}
                item = {'team': team, 'variable': variable, 'common_ICs': json.loads(group['long']['common_ICs']),
                        'common_IC_count': int(group['long']['common_IC_count']),
                        'median_NRMSE': {s: float(r['median_nrmse']) if r['median_nrmse'] else None for s, r in group.items()},
                        'completed_of_18': {s: int(r['completed_all_test_ICs']) for s, r in group.items()}}
                comparisons.append(item)
                if mode == 'repaired' and variable == 'T':
                    all_temperature_above_baseline.append(item['median_NRMSE']['long'] > item['median_NRMSE']['baseline'])
        analysis['per_weight_within_team_comparisons'][str(weight)][mode] = comparisons
analysis['repaired_long_temperature_above_baseline_in_all_21_within_team_comparisons'] = all(all_temperature_above_baseline)
diagnostics = read_csv(report / 'all_weight_member_loss_components.csv')
analysis['member_validation_group_means'] = []
for weight in (1, 100, 1000):
    for family in ('ann', 'p_only', 'tp', 'tp_phi'):
        means = {stage: {field: float(np.mean([float(r[field]) for r in diagnostics if float(r['lambda_value']) == weight
                     and r['family'] == family and r['stage'] == stage and r['split'] == 'val']))
                     for field in ('data_MSE', 'physical_element_L1', 'objective')} for stage in ('short', 'long')}
        analysis['member_validation_group_means'].append({'lambda': weight, 'family': family, 'means': means})
events = [json.loads(line) for line in (ROOT / 'execution.jsonl').read_text().splitlines()]
analysis['phase_seconds'] = {r['phase']: r['seconds'] for r in events if r['phase'] in
                            ('training_wall_finished', 'evaluation_wall_finished', 'report_finished')}
analysis['phase_seconds']['pipeline'] = final['pipeline_seconds']
(ROOT / 'analysis_summary.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + '\n')
inventory = {str(p.relative_to(ROOT)): digest(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name != 'all_artifacts_sha256.json'}
(ROOT / 'all_artifacts_sha256.json').write_text(json.dumps({'files': inventory, 'count': len(inventory)}, indent=2) + '\n')
print(json.dumps({'status': 'passed', 'exact_user_NRMSE_scores': scores_checked, 'incomplete_variable_rows': incomplete_checked,
                  'summary_rows': summary_checked, 'PDFs': len(pdfs), 'ZIPs': len(archives), 'inventory_files': len(inventory),
                  'seconds': time.time() - started}, ensure_ascii=False))
