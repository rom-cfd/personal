"""Same7x10 Cantera comparison and util.py NRMSE for lambda0/1/10/100/1000."""
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from kinetics.ann_deeponet_study import check_deadline
from kinetics.soft_constraint_evaluation import TEAMS
from kinetics.soft_constraint_study import event
from kinetics.penalty_sweep_study import weight_root
from kinetics.p_only_training import sha256, write_json
from scripts.plot_kinetics_7x10_comparison import COLUMNS, STYLE, BLACK
from scripts.report_soft_constraint_comparison import LABELS, limits
from scripts.report_kinetics_nrmse import profile_errors, number, VARIABLES

WEIGHTS = (0, 1, 10, 100, 1000)
COLORS = {0: '#64748b', 1: '#0077bb', 10: '#009988', 100: '#ee7733', 1000: '#aa3377'}
METHODS = [(f'lambda{weight}_{team}', team, weight) for team, _ in TEAMS for weight in WEIGHTS]


def traces_for(root, cfg, data, mode):
    ids = np.flatnonzero(data['split'] == 'test').astype(int).tolist()
    origin = Path(cfg['origin_mu10']); baseline = json.loads((origin / 'baseline_traces.json').read_text())
    traces = {}; sources = {}
    for team, _ in TEAMS:
        for weight in WEIGHTS:
            if weight == 0:
                info = baseline[f'{team}_{mode}']; path = Path(info['path']); expected = info['sha256']
            else:
                parent = origin if weight == 10 else weight_root(root, weight)
                markers = list((parent / 'evaluations' / f'{team}_{mode}_test').glob('*/complete.json'))
                if len(markers) != 1:
                    raise ValueError('Require one frozen complete trace per arm')
                marker = markers[0]; path = marker.parent / 'traces.npz'
                expected = json.loads(marker.read_text())['output_sha256']['traces.npz']
            if sha256(path) != expected:
                raise ValueError('Frozen trace changed')
            sources[str(path)] = expected
            with np.load(path) as f:
                profiles = {i: f[f'trajectory_{i}'] for i in ids}
            for i, trace in profiles.items():
                if trace.shape[1] != 10 or len(trace) > len(data['states'][i]) or not np.isfinite(trace).all():
                    raise ValueError('Invalid saved trajectory')
                np.testing.assert_array_equal(trace[0], data['states'][i, 0])
            traces[f'lambda{weight}_{team}'] = profiles
    return traces, sources


def normalized_metrics(data, traces, ids, mode):
    rows = []
    for i in ids:
        for method, team, weight in METHODS:
            trace = traces[method][i]; r = profile_errors(data['states'][i], trace, float(data['dt']))
            for k, variable in enumerate(VARIABLES):
                rows.append(dict(trajectory=i, method=method, team=team, model=LABELS[team],
                    lambda_value=weight, variable=variable, mode=mode, complete=r['complete'], points=r['points'],
                    last_accepted_ms=(len(trace) - 1) * float(data['dt']) * 1000,
                    nrmse_status=r['status'][k], nrmse=float(r['nrmse'][k]),
                    full_rmse=float(r['full_rmse'][k]), prefix_rmse=float(r['prefix_rmse'][k]),
                    reference_rms=float(r['reference_rms'][k])))
    frame = pd.DataFrame(rows)
    completed = {name: set(p.loc[p.complete, 'trajectory']) for name, p in frame.groupby('method')}
    common = sorted(set.intersection(*completed.values()))
    within = {team: sorted(set.intersection(*[completed[f'lambda{w}_{team}'] for w in WEIGHTS]))
              for team, _ in TEAMS}
    return frame, summary_frame(frame, {team: common for team, _ in TEAMS}, 'all35variants'), \
        summary_frame(frame, within, 'within_team_all5lambdas'), common, within


def summary_frame(frame, selected, scope):
    rows = []
    for method, team, weight in METHODS:
        common = selected[team]
        for variable in VARIABLES:
            part = frame[(frame.method == method) & (frame.variable == variable)]
            values = part.loc[part.trajectory.isin(common), 'nrmse'].dropna()
            rows.append(dict(method=method, team=team, model=LABELS[team], lambda_value=weight,
                variable=variable, mode=part.iloc[0]['mode'], common_scope=scope,
                common_IC_count=len(common), common_ICs=json.dumps(common), valid_nrmse_IC_count=len(values),
                median_nrmse=values.median(), mean_nrmse=values.mean(), std_nrmse=values.std(ddof=0),
                q25_nrmse=values.quantile(.25), q75_nrmse=values.quantile(.75),
                completed_all_test_ICs=int(part.complete.sum())))
    return pd.DataFrame(rows)


def profile_figure(data, traces, i, mode):
    reference = data['states'][i]; dt = float(data['dt']); horizon = (len(reference) - 1) * dt * 1000
    ylimits = limits(reference, traces, i)
    fig, axes = plt.subplots(7, 10, figsize=(31, 20.2), sharex=True, squeeze=False)
    fig.subplots_adjust(left=.087, right=.990, bottom=.062, top=.890, wspace=.36, hspace=.38)
    for row, (team, _) in enumerate(TEAMS):
        failed = [(w, (len(traces[f'lambda{w}_{team}'][i]) - 1) * dt * 1000) for w in WEIGHTS
                  if len(traces[f'lambda{w}_{team}'][i]) < len(reference)]
        for k, ax in enumerate(axes[row]):
            for weight in WEIGHTS:
                trace = traces[f'lambda{weight}_{team}'][i]; tt = np.arange(len(trace)) * dt * 1000
                ax.plot(tt, trace[:, k], color=COLORS[weight], lw=1.0, alpha=.93, zorder=2)
                if len(trace) < len(reference):
                    ax.plot(tt[-1], trace[-1, k], marker='x', color=COLORS[weight], ms=3.5, mew=.9, zorder=5)
            ax.plot(np.arange(len(reference)) * dt * 1000, reference[:, k], color=BLACK,
                    ls=(0, (4, 2.5)), lw=1.1, zorder=3)
            ax.set_xlim(0, horizon); ax.set_ylim(*ylimits[k]); ax.set_xticks([0, 2, 4, 6, 8, 10])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            formatter = ScalarFormatter(useOffset=False, useMathText=True); formatter.set_powerlimits((-3, 3))
            ax.yaxis.set_major_formatter(formatter); ax.yaxis.get_offset_text().set_size(7.5)
            ax.grid(True, which='major', ls=':', lw=.45, color='#b7c1cc', alpha=.65); ax.tick_params(pad=2)
            if row == 0:
                ax.set_title(COLUMNS[k], pad=12, fontweight='semibold')
            if row == 6:
                ax.set_xlabel('Time [ms]', labelpad=5)
        position = axes[row, 0].get_position(); label = LABELS[team].replace(' DeepONet10', '\nDeepONet10')
        fig.text(.009, (position.y0 + position.y1) / 2, f'{row + 1}  {label}',
                 ha='left', va='center', fontsize=11, fontweight='semibold', linespacing=1.5)
        if failed:
            axes[row, 0].text(.985, .97, '\n'.join(f'lambda {w}: stop {t:.3f} ms' for w, t in failed),
                transform=axes[row, 0].transAxes, ha='right', va='top', fontsize=6.6, color='#555555',
                bbox=dict(facecolor='white', alpha=.88, edgecolor='none', pad=1.8))
    t0, p0, phi = data['initials'][i]
    fig.suptitle(f'Element soft-constraint weight sweep | Test IC {i:02d}', x=.54, y=.980,
                 fontsize=21, fontweight='semibold')
    fig.text(.54, .954, f'$T_0$ = {t0:.1f} K     $P_0$ = {p0 / 101325:.3f} atm     '
        + r'$\phi_0$' + f' = {phi:.3f}     |     {mode.capitalize()} self-fed rollout, '
        + r'$\Delta t = 2\,\mu$s', ha='center', va='center', fontsize=12, color='#414b57')
    handles = [Line2D([0], [0], color=BLACK, ls=(0, (4, 2.5)), lw=1.6, label='Cantera reference')]
    handles += [Line2D([0], [0], color=COLORS[w], lw=1.6, label=f'lambda = {w} (10k)') for w in WEIGHTS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .941), ncol=3,
               frameon=False, fontsize=11.5, handlelength=3.5, columnspacing=3)
    fig.text(.54, .021, 'Shared linear y scale within each variable column. Saved 2 us samples; early stops marked x.\n'
        'Every member has10000 updates and the same seed-specific initialization/minibatch stream. '
        'Mixed teams use ANN10 + DeepONet10 at the same lambda. Lambda0/10 are frozen controls; inference is unchanged.',
        ha='center', va='center', fontsize=9.5, color='#4b5563', linespacing=1.5)
    if len(fig.axes) != 70:
        raise AssertionError('Expected7x10 axes')
    return fig


def table_figure(frame, common, mode, *, trajectory=None, initial=None, team=None):
    summary = trajectory is None; methods = [(m, t, w) for m, t, w in METHODS if team is None or t == team]
    fig, ax = plt.subplots(figsize=(22, 18 if team is None else 8.5)); ax.axis('off')
    fig.subplots_adjust(left=.02, right=.98, bottom=.16, top=.83)
    rows = []; colors = []
    for method, model, weight in methods:
        part = frame[frame.method == method].set_index('variable'); values = []
        for variable in VARIABLES:
            r = part.loc[variable]
            values.append(number(r['median_nrmse'] if summary else r['nrmse'])
                          if summary or r['complete'] else '—')
        status = f"{int(part.loc['T', 'completed_all_test_ICs'])}/18" if summary else (
            'Complete' if part.loc['T', 'complete'] else f"Stop {part.loc['T', 'last_accepted_ms']:.3f} ms")
        group = next(j for j, (t, _) in enumerate(TEAMS, 1) if t == model)
        rows.append([f'{group}  {LABELS[model]}', str(weight)] + values + [status])
        colors.append([('#f1f5f9' if group % 2 == 0 else '#ffffff')] * 13)
    table = ax.table(cellText=rows, colLabels=['Model / same-state mean', 'lambda'] + [r'$T$'] + COLUMNS[1:] + [
        'Complete ICs' if summary else '10 ms status'], cellColours=colors, cellLoc='center', colLoc='center',
        bbox=[0, 0, 1, 1], colWidths=[.207, .06] + [.064] * 10 + [.093])
    table.auto_set_font_size(False); table.set_fontsize(9.5 if team is None else 11)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#d8dee6'); cell.set_linewidth(.45)
        if r == 0:
            cell.set_facecolor('#213248'); cell.set_text_props(color='white', fontweight='semibold', fontsize=11)
        elif c == 0:
            cell.set_text_props(ha='left', fontweight='semibold'); cell.PAD = .035
        elif c == 1:
            cell.set_text_props(color=COLORS[methods[r - 1][2]], fontweight='bold')
    if summary:
        title = 'Lambda sweep | Cantera vs NN NRMSE summary'
        scope = 'all35 variants' if team is None else f'all5 lambdas within {LABELS[team]}'
        subtitle = f'Median of per-IC NRMSE on the same {len(common)} complete test ICs for {scope}'
        extra = 'Common ICs: ' + ', '.join(map(str, common)) + '.'
        if team is not None:
            extra += ' IC sets differ between teams; use the all35-variant summary for architecture comparisons.'
    else:
        t, p, phi = initial
        title = f'Lambda sweep | Test IC {trajectory:02d} | Cantera vs NN NRMSE'
        subtitle = f'T0 = {t:.1f} K    P0 = {p / 101325:.3f} atm    phi0 = {phi:.3f}    |    {mode.capitalize()} self-fed, 10 ms'
        extra = 'Every graph row includes all5 lambdas and the same Cantera reference.'
    fig.suptitle(title, x=.5, y=.96, fontsize=22, fontweight='semibold')
    fig.text(.5, .913, subtitle, ha='center', va='center', fontsize=12, color='#475569')
    fig.text(.5, .871, r'$\mathrm{NRMSE}=\sqrt{\mathrm{mean}[(y_{Cantera}-y_{NN})^2]}/\sqrt{\mathrm{mean}[y_{Cantera}^{2}]}$',
             ha='center', va='center', fontsize=14)
    fig.text(.024, .11, 'Exact user util.py defaults: norm="rms", reference=Cantera; equal time-sample weights, dimensionless ratio without x100.\n'
        'N/A: undefined zero-reference RMS or empty common set. —: early stop, so no full-horizon NRMSE. Nonzero constant N2 is valid.\n'
        'Every network has10000 updates. Lambda0/10 controls are reused; lambda1/100/1000 are fresh paired-seed models.\n'
        + extra, ha='left', va='top', fontsize=9.5, color='#475569', linespacing=1.55)
    return fig


def training_diagnostics(root, cfg, destination):
    origin = Path(cfg['origin_mu10']); old = pd.read_csv(origin / 'member_diagnostics/loss_components.csv')
    old['lambda_value'] = old.stage.map({'baseline': 0, 'soft_mu10': 10})
    if old.lambda_value.isna().any():
        raise ValueError('Unknown control diagnostic stage')
    old['objective'] = old.data_MSE + old.lambda_value * old.physical_element_L1
    old['weighted_element_L1'] = old.lambda_value * old.physical_element_L1
    columns = ['lambda_value', 'family', 'seed', 'split', 'pairs', 'objective', 'data_MSE',
               'physical_element_L1', 'weighted_element_L1', 'invalid_inverse_row_fraction']
    frames = [old[columns]]
    for w in cfg['lambdas']:
        frames.append(pd.read_csv(weight_root(root, w) / 'member_diagnostics/loss_components.csv')[columns])
    diagnostics = pd.concat(frames, ignore_index=True)
    diagnostics.to_csv(destination / 'member_loss_components.csv', index=False)
    diagnostics.groupby(['lambda_value', 'family', 'split'])[columns[5:]].mean().to_csv(
        destination / 'mean_member_loss_components.csv')
    rows = []
    for w in (1, 10, 100, 1000):
        parent = origin if w == 10 else weight_root(root, w)
        for r in json.loads((parent / 'final_members.json').read_text()):
            state = json.loads(Path(r['checkpoint'], 'state.json').read_text())
            rows.extend(dict(lambda_value=w, family=r['family'], **h) for h in state['full_train_history'])
    history = pd.DataFrame(rows); history.to_csv(destination / 'observed_full_train_components.csv', index=False)
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    for axrow, family in zip(axes, ('ann', 'p_only', 'tp', 'tp_phi')):
        for ax, key in zip(axrow, ('data_loss', 'element_loss')):
            for w in (1, 10, 100, 1000):
                p = history[(history.family == family) & (history.lambda_value == w) & (history['update'] >= 1000)]
                pivot = p.pivot(index='seed', columns='update', values=key).dropna(axis=1)
                for values in pivot.to_numpy():
                    ax.plot(pivot.columns, values, color=COLORS[w], lw=.55, alpha=.15)
                mean = np.exp(np.log(np.maximum(pivot.to_numpy(), 1e-300)).mean(axis=0))
                ax.plot(pivot.columns, mean, color=COLORS[w], lw=1.5, label=f'lambda={w}')
            ax.set_yscale('log'); ax.set_title(f'{family}: {key} (unweighted)')
            ax.grid(True, which='both', ls=':', alpha=.3); ax.set_xlabel('Optimizer update'); ax.legend(fontsize=8)
    fig.suptitle('Matched10k: faint individual members; thick geometric means. Updates1000 onward; full histories exported.')
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(destination / 'training_loss_components.pdf')
    fig.savefig(destination / 'training_loss_components.png', dpi=160); plt.close(fig)


def verify_training(root, cfg, snapshot):
    records = json.loads((root / 'final_members.json').read_text()); controls = snapshot['lambda10_members']
    if len(records) != 120:
        raise ValueError('Expected120 endpoints')
    for r in records:
        cp = Path(r['checkpoint']); state = json.loads((cp / 'state.json').read_text())
        marker = json.loads((cp / 'complete.json').read_text())
        for name, expected in marker['output_sha256'].items():
            if sha256(cp / name) != expected:
                raise ValueError('Terminal checkpoint changed')
        with np.load(cp / 'adam.npz') as adam:
            if int(adam['v0']) != 10000:
                raise ValueError('Actual Adam iteration differs')
        if not (r['updates'] == state['update'] == state['optimizer_iteration'] == 10000
                and r['mu'] == state['signature']['mu'] and r['mu'] in cfg['lambdas']
                and r['reason'] == 'fixed_update_budget' and not state['plateau_monitor']['enabled']):
            raise ValueError('Incorrect update, loss weight or stop criterion')
        old = next(c for c in controls if c['family'] == r['family'] and c['seed'] == r['seed'])
        if sha256(cp / 'model.json') != old['preprocessing_sha256']:
            raise ValueError('Preprocessing changed between weights')
    for source, expected in snapshot['source_sha256'].items():
        if sha256(source) != expected:
            raise ValueError('Frozen source changed')
    previous = json.loads((root / 'original_results_sha256.json').read_text())
    for source, expected in previous.items():
        if sha256(source) != expected:
            raise ValueError(f'Preexisting result changed: {source}')
    return records, len(previous)


def report(root, cfg, snapshot, data, deadline):
    before = time.perf_counter(); destination = root / 'reports' / f'report_{time.time_ns()}'
    destination.mkdir(parents=True, exist_ok=False); ids = np.flatnonzero(data['split'] == 'test').astype(int).tolist()
    sources = {}; comparisons = {}
    write_json(destination / 'run_snapshot.json', snapshot)
    with plt.rc_context(STYLE):
        training_diagnostics(root, cfg, destination)
        for mode in ('repaired', 'raw'):
            check_deadline(deadline); traces, used = traces_for(root, cfg, data, mode); sources.update(used)
            frame, summary, within_summary, common, within = normalized_metrics(data, traces, ids, mode)
            parent = destination / mode; parent.mkdir(); png = parent / 'png'; png.mkdir(); tables = parent / 'tables'; tables.mkdir()
            frame.to_csv(parent / 'per_IC_NRMSE.csv', index=False)
            summary.to_csv(parent / 'common_IC_NRMSE_summary.csv', index=False)
            within_summary.to_csv(parent / 'within_team_NRMSE_summary.csv', index=False)
            comparisons[mode] = dict(common_ICs=common, common_IC_count=len(common), within_team_common_ICs=within)
            fig = table_figure(summary, common, mode); fig.savefig(parent / 'NRMSE_summary.pdf')
            fig.savefig(parent / 'NRMSE_summary.png', dpi=160); plt.close(fig)
            with PdfPages(parent / 'within_team_NRMSE.pdf') as pdf:
                for team, _ in TEAMS:
                    fig = table_figure(within_summary, within[team], mode, team=team)
                    pdf.savefig(fig); plt.close(fig)
            with PdfPages(parent / 'profiles_and_NRMSE.pdf') as pdf:
                for page, i in enumerate(ids, 1):
                    check_deadline(deadline); fig = profile_figure(data, traces, i, mode)
                    pdf.savefig(fig); fig.savefig(png / f'test_IC{i:02d}_7x10.png', dpi=160); plt.close(fig)
                    fig = table_figure(frame[frame.trajectory == i], common, mode, trajectory=i, initial=data['initials'][i])
                    pdf.savefig(fig); fig.savefig(tables / f'test_IC{i:02d}_NRMSE.png', dpi=160); plt.close(fig)
                    event(root, phase='report_page_finished', mode=mode, page=page, trajectory=i)
            write_json(parent / 'complete.json', dict(pages=36, axes_per_profile=70, metric_rows=len(frame),
                common_ICs=common, within_team_common_ICs=within,
                output_sha256={str(p.relative_to(parent)): sha256(p) for p in parent.rglob('*') if p.is_file()}))
    records, previous_count = verify_training(root, cfg, snapshot)
    write_json(destination / 'trace_sources.json', sources)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ('full_train', 'timing')} for r in records]).to_csv(
        destination / 'training_summary.csv', index=False)
    write_json(destination / 'verification.json', dict(terminal_checkpoints=120, actual_Adam_iterations=10000,
        unchanged_preprocessors=120, fixed_update_budget_members=120, preexisting_files_preserved=previous_count,
        all_source_hashes_verified=len(snapshot['source_sha256']), comparison=comparisons))
    (destination / 'REPORT_KO.md').write_text('\n'.join([
        '원소 soft constraint 가중치 lambda=1,100,1000의 matched10k 비교를 완료했다.',
        'ANN/P-only/T-P/T-P-phi 각10개, 가중치별40개, 총120개를 순차로 새로10000회 학습했다.',
        'lambda0/10은 기존10000회 모델과 평가를 재사용했다. 기존lambda10 장기 모델은 회차가 달라 이 비교에 넣지 않았다.',
        '각seed의 초기화, minibatch stream, 데이터8815쌍, 2us target, 전처리, architecture, Adam/LR/clip 및 추론보정은 동일하다.',
        '전체손실=MSE+lambda*원소L1. lambda가 다르므로 전체손실 수치만으로 가중치 간 성능을 비교하면 안 된다.',
        '120개 종료점을 모두 고정한 뒤 기존18개 test IC에서 raw/repaired를 평가했다. 테스트는 종료점/가중치 선택에 쓰지 않았다.',
        '이미 여러 번 확인한 테스트의 탐색적 가중치 비교이며, 최종 선택을 독립적으로 검증한 결과는 아니다.',
        '그래프는7행×10변수, 검정Cantera와5개lambda 곡선이다. 각IC 그림 바로 뒤에 같은NRMSE표를 넣었다.',
        'NRMSE는util.py기본값 RMSE(Cantera,NN)/RMS(Cantera), 동일 시간점 가중치, 무차원 비율이다.',
        '미완주는전체NRMSE없음. 전체요약은35개구성 공통완주IC 중앙값이며 빈집합은N/A다.',
        '별도 within-team 표는 각팀 내5개가중치가 모두완주한IC만 사용한다. 팀마다IC가 다르므로 architecture 간 순위로 쓰면 안 된다.',
        'Repaired 공통IC: ' + str(comparisons['repaired']['common_ICs']),
        'Raw 공통IC: ' + str(comparisons['raw']['common_ICs']),
        f'기존결과 {previous_count}개 파일을 보존했고, 새120개 실제Adam iteration과전처리를 검증했다.',
        '추론에서 강제한 보존성을 학습된 보존성으로 해석하지 않는다. 개별member 보존진단과raw완주율을 함께 확인한다.',
        '', '[Repaired 그림+표](repaired/profiles_and_NRMSE.pdf) · [요약](repaired/NRMSE_summary.pdf)',
        '[Raw 그림+표](raw/profiles_and_NRMSE.pdf) · [팀별가중치비교](repaired/within_team_NRMSE.pdf)',
        '[손실성분](training_loss_components.pdf)', '']))
    for mode in ('repaired', 'raw'):
        archive = destination / f'{mode}_comparison_bundle.zip'
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            shared = [p for p in destination.iterdir() if p.is_file() and p.suffix != '.zip']
            for p in shared + [p for p in (destination / mode).rglob('*') if p.is_file()]:
                z.write(p, arcname=p.relative_to(destination))
    write_json(destination / 'complete.json', dict(output_sha256={str(p.relative_to(destination)): sha256(p)
        for p in destination.rglob('*') if p.is_file()}))
    finished = time.time(); start = json.loads((root / 'budget.json').read_text())['started_unix']
    write_json(root / 'FINAL_ARTIFACTS.json', dict(report=str(destination),
        report_complete_sha256=sha256(destination / 'complete.json'), new_members=120, updates_per_member=10000,
        lambdas=[0, 1, 10, 100, 1000], comparison=comparisons, finished_unix=finished, pipeline_seconds=finished - start))
    event(root, phase='report_finished', path=str(destination), seconds=time.perf_counter() - before)
    return destination
