"""Independent numerical checks on saved traces; run after the worker exits0."""
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

ROOT = Path('results/gamma_long/run_001')
TEAMS = ['ann10', 'p_only10', 'ann_p_only20', 'tp10', 'ann_tp20', 'tp_phi10', 'ann_tp_phi20']
VARIABLES = ['T', 'H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'N2']
started = time.time()


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def close(actual, expected):
    if math.isnan(float(expected)):
        assert actual == '' or math.isnan(float(actual)), (actual, expected)
    else:
        assert actual != '' and math.isclose(float(actual), float(expected), rel_tol=3e-12, abs_tol=2e-14), (actual, expected)


def median(rows, ids):
    values = [float(r['nrmse']) for r in rows if int(r['trajectory']) in ids and r['nrmse'] != '']
    return float(np.median(values)) if values else float('nan')


final = json.loads((ROOT / 'FINAL_ARTIFACTS.json').read_text())
report = Path(final['report'])
snapshot = json.loads((ROOT / 'run.json').read_text())
cfg = snapshot['config']
GAMMAS = cfg['gammas']
attempt = sorted((ROOT / 'process_launches').glob('attempt_*'))[-1]
end = json.loads((attempt / 'exit.json').read_text())
assert end['returncode'] == 0 and not end['timed_out']

tree = ast.parse(Path('references/user_nrmse/util.py').read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_nrmse_1d')
namespace = {'np': np}
exec(compile(ast.Module(body=[fn], type_ignores=[]), 'references/user_nrmse/util.py', 'exec'), namespace)
user_nrmse = namespace['_nrmse_1d']
with np.load(snapshot['base_config']['source']) as f:
    truth, splits = f['states'], f['split']
paired_sources = json.loads((ROOT / 'paired_short_traces.json').read_text())
frames = {}
scores = incomplete = undefined = summary_values = 0
cohorts = {}

for split in ('val', 'test'):
    ids = np.flatnonzero(splits == split).astype(int).tolist()
    cohorts[split] = {}
    for mode in ('repaired', 'raw'):
        folder = report / split / mode
        rows = read_csv(folder / 'per_IC_NRMSE.csv')
        assert len(rows) == 15120
        frames[split, mode] = rows
        lookup = {(r['method'], int(r['trajectory']), r['variable']): r for r in rows}
        for stage in ('short', 'long'):
            for gamma in GAMMAS:
                for team in TEAMS:
                    method = f'gamma_{gamma:g}_{team}'
                    if stage == 'short':
                        info = paired_sources[f'{split}_{mode}'][method]
                        path = Path(info['path'])
                        assert digest(path) == info['sha256']
                    else:
                        markers = list((ROOT / f'gamma_{gamma:g}' / 'evaluations' / f'{team}_{mode}_{split}').glob('*/complete.json'))
                        assert len(markers) == 1
                        path = markers[0].parent / 'traces.npz'
                        assert digest(path) == json.loads(markers[0].read_text())['output_sha256']['traces.npz']
                    with np.load(path) as f:
                        for i in ids:
                            pred = f[f'trajectory_{i}']
                            np.testing.assert_array_equal(pred[0], truth[i, 0])
                            complete = len(pred) == len(truth[i])
                            for k, variable in enumerate(VARIABLES):
                                r = lookup[f'{stage}_{method}', i, variable]
                                ref = truth[i, :, k]
                                assert int(r['points']) == len(pred) and (r['complete'] == 'True') == complete
                                close(r['reference_rms'], np.sqrt(np.mean(ref ** 2)))
                                close(r['prefix_rmse'], np.sqrt(np.mean((ref[:len(pred)] - pred[:, k]) ** 2)))
                                if not complete:
                                    assert r['nrmse'] == '' and r['full_rmse'] == ''
                                    incomplete += 1
                                    continue
                                close(r['full_rmse'], np.sqrt(np.mean((ref - pred[:, k]) ** 2)))
                                try:
                                    expected = user_nrmse(ref, pred[:, k])
                                except ValueError:
                                    assert r['nrmse'] == ''
                                    undefined += 1
                                else:
                                    close(r['nrmse'], expected)
                                    scores += 1
        methods = list(dict.fromkeys(r['method'] for r in rows))
        complete = {m: {int(r['trajectory']) for r in rows if r['method'] == m and r['complete'] == 'True'} for m in methods}
        common = sorted(set.intersection(*(complete[m] for m in methods)))
        teams = {r['method']: r['team'] for r in rows}
        within = {t: sorted(set.intersection(*(complete[m] for m in methods if teams[m] == t))) for t in TEAMS}
        cohorts[split][mode] = dict(common_ICs=common, within_team_common_ICs=within)
        parts = {}
        for r in rows:
            parts.setdefault((r['method'], r['variable']), []).append(r)
        for name, local in [('common_IC_NRMSE_summary.csv', False), ('within_team_NRMSE_summary.csv', True)]:
            table = read_csv(folder / name)
            assert len(table) == 840
            for r in table:
                matched = within[r['team']] if local else common
                assert json.loads(r['common_ICs']) == matched and int(r['common_IC_count']) == len(matched)
                part = parts[r['method'], r['variable']]
                values = np.array([float(p['nrmse']) for p in part if int(p['trajectory']) in matched and p['nrmse'] != ''])
                assert int(r['valid_nrmse_IC_count']) == len(values)
                assert int(r['completed_all_ICs']) == sum(p['complete'] == 'True' for p in part)
                for field, val in [('median_nrmse', np.median(values) if len(values) else np.nan),
                                   ('mean_nrmse', np.mean(values) if len(values) else np.nan),
                                   ('std_nrmse', np.std(values) if len(values) else np.nan),
                                   ('q25_nrmse', np.quantile(values, .25) if len(values) else np.nan),
                                   ('q75_nrmse', np.quantile(values, .75) if len(values) else np.nan)]:
                    close(r[field], val)
                    summary_values += 1
        pairs = read_csv(folder / 'same_gamma_short_long_summary.csv')
        assert len(pairs) == 420
        for r in pairs:
            parts2 = [[p for p in rows if p['stage'] == stage and p['team'] == r['team'] and float(p['gamma']) == float(r['gamma']) and p['variable'] == r['variable']] for stage in ('short', 'long')]
            matched = sorted(set.intersection(*({int(p['trajectory']) for p in part if p['complete'] == 'True'} for part in parts2)))
            assert json.loads(r['common_ICs']) == matched and int(r['common_IC_count']) == len(matched)
            for stage, part in zip(('short', 'long'), parts2):
                close(r[stage + '_median_NRMSE'], median(part, matched))
                assert int(r[stage + '_completed']) == sum(p['complete'] == 'True' for p in part)
                summary_values += 1

for stage in ('short', 'long'):
    selection = json.loads((ROOT / f'{stage}_validation_selection.json').read_text())
    assert selection['split'] == 'val' and selection['mode'] == 'repaired' and not selection['test_used']
    val = [r for r in frames['val', 'repaired'] if r['stage'] == stage and r['variable'] == 'T']
    for team in TEAMS:
        completed = {g: {int(r['trajectory']) for r in val if r['team'] == team and float(r['gamma']) == g and r['complete'] == 'True'} for g in GAMMAS}
        maximum = max(map(len, completed.values()))
        eligible = [g for g in GAMMAS if len(completed[g]) == maximum]
        matched = sorted(set.intersection(*(completed[g] for g in eligible)))
        values = {g: median([r for r in val if r['team'] == team and float(r['gamma']) == g], matched) for g in eligible} if matched else {}
        chosen = min(eligible, key=lambda g: (values[g], g)) if values else None
        s = selection['choices'][team]
        assert s['chosen_gamma'] == chosen and s['common_ICs'] == matched and s['maximum_completed_ICs'] == maximum
        assert s['completion_counts'] == {str(g): len(completed[g]) for g in GAMMAS}
        for g, value in values.items():
            close(s['temperature_NRMSE'][str(g)], value)
selection = json.loads((ROOT / 'long_validation_selection.json').read_text())
for mode in ('repaired', 'raw'):
    table = read_csv(report / 'test' / mode / 'validation_selected_same_gamma_short_long.csv')
    for r in table:
        assert float(r['gamma']) == selection['choices'][r['team']]['chosen_gamma']
    assert len(table) == 10 * sum(v['chosen_gamma'] is not None for v in selection['choices'].values())

records = json.loads((ROOT / 'final_members.json').read_text())
origins = json.loads((ROOT / 'paired_short_members.json').read_text())
assert len(records) == len(origins) == 240
assert len({(r['gamma'], r['family'], r['seed']) for r in records}) == 240
assert sum(r['origin_kind'] == 'rebuilt_short_checkpoint' for r in origins) == 20
active = []
for r in records:
    origin = next(o for o in origins if (o['gamma'], o['family'], o['seed']) == (r['gamma'], r['family'], r['seed']))
    assert origin['updates'] == 10000 and r['added_updates'] == r['updates'] - 10000 >= 0
    cp = Path(r['checkpoint'])
    state = json.loads((cp / 'state.json').read_text())
    with np.load(cp / 'adam.npz') as f:
        assert int(f['v0']) == state['optimizer_iteration'] == state['update'] == r['updates']
    for name, expected in json.loads((cp / 'complete.json').read_text())['output_sha256'].items():
        assert digest(cp / name) == expected
    allowance = r['allowance']
    assert allowance['seconds'] == 150 and allowance['open_visit'] is None
    close(allowance['consumed_seconds'], sum(v['charged_seconds'] for v in allowance['visits']))
    charged = 0.
    for visit in allowance['visits']:
        assert 0 < visit['allowed_seconds'] <= max(0., 150 - charged) + 1e-8
        charged += visit['charged_seconds']
    assert allowance['consumed_seconds'] >= 150 or r['reason'] == 'plateau'
    stages = [v['stage'] for v in allowance['visits']]
    assert ('rebuild_short10k' in stages) == (origin['origin_kind'] == 'rebuilt_short_checkpoint')
    active.append(allowance['consumed_seconds'])
    if not r.get('borrowed_short_endpoint'):
        assert state['signature']['loss'] == 'transformed_residual_MSE_only'
        assert (r['reason'] == 'plateau') == state['plateau_monitor']['plateau']

events = [json.loads(line) for line in (ROOT / 'execution.jsonl').read_text().splitlines()]
freeze = next(i for i, e in enumerate(events) if e['phase'] == 'long_validation_selection_frozen')
first_test = next(i for i, e in enumerate(events) if e['phase'] == 'evaluation_finished' and e['split'] == 'test')
assert freeze < first_test
manifest = json.loads((report / 'complete.json').read_text())['output_sha256']
assert digest(report / 'complete.json') == final['report_complete_sha256']
for relative, expected in manifest.items():
    assert digest(report / relative) == expected
pdfs = {}
for p in report.rglob('*.pdf'):
    count = len(re.findall(rb'/Type\s*/Page\b', p.read_bytes()))
    expected = 72 if p.name == 'profiles_and_NRMSE.pdf' else 14 if p.name == 'within_team_NRMSE.pdf' else 1
    assert count == expected, (str(p), count)
    pdfs[str(p.relative_to(report))] = count
archives = {}
for p in report.glob('*.zip'):
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        archives[p.name] = dict(bytes=p.stat().st_size, files=len(z.namelist()))
assert len(archives) == 4
result = dict(status='passed', exact_user_NRMSE_scores=scores, incomplete_variable_rows=incomplete,
              undefined_variable_rows=undefined, summary_values_checked=summary_values,
              actual_Adam_iterations_verified=240, paired_origins_verified=240,
              validation_choices_independently_verified=14, validation_frozen_before_test=True,
              active_seconds=dict(min=min(active), median=float(np.median(active)), max=max(active), total=sum(active)),
              plateau_members=sum(r['reason'] == 'plateau' for r in records),
              report_hashes_verified=len(manifest), pdf_page_objects=pdfs, zip_CRC_checked=archives,
              cohorts=cohorts, seconds=time.time()-started)
(ROOT / 'postrun_verification.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
analysis = dict(short_validation_selection=json.loads((ROOT/'short_validation_selection.json').read_text()),
                long_validation_selection=selection,
                selected_test_comparisons={mode:read_csv(report/'test'/mode/'validation_selected_same_gamma_short_long.csv') for mode in ('repaired','raw')},
                phase_seconds={r['phase']:r['seconds'] for r in events if r['phase'] in ('training_wall_finished','evaluation_wall_finished','report_finished')},
                pipeline_seconds=final['pipeline_seconds'])
(ROOT / 'analysis_summary.json').write_text(json.dumps(analysis, indent=2, allow_nan=False) + '\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('cohorts','pdf_page_objects','zip_CRC_checked')}))
