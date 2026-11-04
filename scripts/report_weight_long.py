"""Per-lambda baseline/short/long7x10 profiles and matched util.py NRMSE."""
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
from kinetics.element_long_training import joint_monitor, joint_check
from kinetics.penalty_sweep_study import weight_root
from kinetics.soft_constraint_evaluation import TEAMS
from kinetics.soft_constraint_study import event
from kinetics.p_only_training import sha256, write_json
from scripts.plot_kinetics_7x10_comparison import COLUMNS, STYLE, BLACK, BLUE, ORANGE
from scripts.report_kinetics_nrmse import profile_errors, number, VARIABLES
from scripts.report_soft_constraint_comparison import LABELS, limits

GRAY = '#64748b'
STAGE_COLORS = dict(baseline=GRAY, short=BLUE, long=ORANGE)
PER_WEIGHT_METHODS = [(f'{s}_{team}', team, s) for team, _ in TEAMS for s in ('baseline', 'short', 'long')]


def traces_for(root, cfg, snapshot, data, weight, mode):
    origin = Path(cfg['origin']); baseline_root = Path(snapshot['origin_snapshot']['config']['origin_mu10'])
    baseline = json.loads((baseline_root / 'baseline_traces.json').read_text())
    ids = np.flatnonzero(data['split'] == 'test').astype(int).tolist(); traces = {}; sources = {}
    for team, _ in TEAMS:
        for stage in ('baseline', 'short', 'long'):
            if stage == 'baseline':
                info = baseline[f'{team}_{mode}']; path = Path(info['path']); expected = info['sha256']
            else:
                parent = weight_root(origin if stage == 'short' else root, weight)
                markers = list((parent / 'evaluations' / f'{team}_{mode}_test').glob('*/complete.json'))
                if len(markers) != 1:
                    raise ValueError('Require one frozen complete evaluation per arm')
                path = markers[0].parent / 'traces.npz'
                expected = json.loads(markers[0].read_text())['output_sha256']['traces.npz']
            if sha256(path) != expected:
                raise ValueError('Frozen trace changed')
            sources[str(path)] = expected
            with np.load(path) as f:
                profiles = {i: f[f'trajectory_{i}'] for i in ids}
            for i, trace in profiles.items():
                if trace.shape[1] != 10 or len(trace) > len(data['states'][i]) or not np.isfinite(trace).all():
                    raise ValueError('Invalid saved trajectory')
                np.testing.assert_array_equal(trace[0], data['states'][i, 0])
            traces[f'{stage}_{team}'] = profiles
    return traces, sources


def metrics_frame(data, traces, ids, mode, weight):
    rows = []
    for i in ids:
        for method, team, stage in PER_WEIGHT_METHODS:
            trace = traces[method][i]; r = profile_errors(data['states'][i], trace, float(data['dt']))
            for k, variable in enumerate(VARIABLES):
                rows.append(dict(trajectory=i, method=method, team=team, model=LABELS[team], stage=stage,
                    lambda_value=0 if stage == 'baseline' else weight, variable=variable, mode=mode,
                    complete=r['complete'], points=r['points'], last_accepted_ms=(len(trace) - 1) * float(data['dt']) * 1000,
                    nrmse_status=r['status'][k], nrmse=float(r['nrmse'][k]), full_rmse=float(r['full_rmse'][k]),
                    prefix_rmse=float(r['prefix_rmse'][k]), reference_rms=float(r['reference_rms'][k])))
    return pd.DataFrame(rows)


def summaries(frame):
    complete = {m: set(p.loc[p.complete, 'trajectory']) for m, p in frame.groupby('method')}
    common = sorted(set.intersection(*complete.values()))
    within = {team: sorted(set.intersection(*[complete[m] for m in frame.loc[frame.team == team, 'method'].unique()]))
              for team, _ in TEAMS}
    return summarize(frame, {t: common for t, _ in TEAMS}, 'all_variants'), \
        summarize(frame, within, 'within_team'), common, within


def summarize(frame, cohorts, scope):
    rows = []
    for (method, variable), part in frame.groupby(['method', 'variable'], sort=False):
        first = part.iloc[0]; ids = cohorts[first.team]
        values = part.loc[part.trajectory.isin(ids), 'nrmse'].dropna()
        rows.append(dict(method=method, team=first.team, model=first.model, stage=first.stage,
            lambda_value=first.lambda_value, variable=variable, mode=first['mode'], common_scope=scope,
            common_ICs=json.dumps(ids), common_IC_count=len(ids), valid_nrmse_IC_count=len(values),
            median_nrmse=values.median(), mean_nrmse=values.mean(), std_nrmse=values.std(ddof=0),
            q25_nrmse=values.quantile(.25), q75_nrmse=values.quantile(.75), completed_all_test_ICs=int(part.complete.sum())))
    return pd.DataFrame(rows)


def profile_figure(data, traces, i, mode, weight):
    truth = data['states'][i]; dt = float(data['dt']); ylimits = limits(truth, traces, i)
    fig, axes = plt.subplots(7, 10, figsize=(31, 20.2), sharex=True, squeeze=False)
    fig.subplots_adjust(left=.087, right=.990, bottom=.062, top=.900, wspace=.36, hspace=.38)
    for row, (team, _) in enumerate(TEAMS):
        failed = []
        for stage in ('baseline', 'short', 'long'):
            trace = traces[f'{stage}_{team}'][i]
            if len(trace) < len(truth):
                failed.append(f'{stage}: stop {(len(trace) - 1) * dt * 1000:.3f} ms')
        for k, ax in enumerate(axes[row]):
            for stage in ('baseline', 'short', 'long'):
                trace = traces[f'{stage}_{team}'][i]; tt = np.arange(len(trace)) * dt * 1000
                ax.plot(tt, trace[:, k], color=STAGE_COLORS[stage], lw=1.05, alpha=.95, zorder=2)
                if len(trace) < len(truth):
                    ax.plot(tt[-1], trace[-1, k], marker='x', color=STAGE_COLORS[stage], ms=3.5, mew=.9, zorder=5)
            ax.plot(np.arange(len(truth)) * dt * 1000, truth[:, k], color=BLACK,
                    ls=(0, (4, 2.5)), lw=1.1, zorder=3)
            ax.set_xlim(0, 10); ax.set_ylim(*ylimits[k]); ax.set_xticks([0, 2, 4, 6, 8, 10])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            formatter = ScalarFormatter(useOffset=False, useMathText=True); formatter.set_powerlimits((-3, 3))
            ax.yaxis.set_major_formatter(formatter); ax.yaxis.get_offset_text().set_size(7.5)
            ax.grid(True, which='major', ls=':', lw=.45, color='#b7c1cc', alpha=.65); ax.tick_params(pad=2)
            if row == 0:
                ax.set_title(COLUMNS[k], pad=12, fontweight='semibold')
            if row == 6:
                ax.set_xlabel('Time [ms]', labelpad=5)
        position = axes[row, 0].get_position(); label = LABELS[team].replace(' DeepONet10', '\nDeepONet10')
        fig.text(.009, (position.y0 + position.y1) / 2, f'{row + 1}  {label}', ha='left', va='center',
                 fontsize=11, fontweight='semibold', linespacing=1.5)
        if failed:
            axes[row, 0].text(.985, .97, '\n'.join(failed), transform=axes[row, 0].transAxes,
                ha='right', va='top', fontsize=6.8, color='#555555',
                bbox=dict(facecolor='white', alpha=.88, edgecolor='none', pad=1.8))
    t, p, phi = data['initials'][i]
    fig.suptitle(f'Lambda {int(weight)} continuation: Cantera vs NN | Test IC {i:02d}', x=.54, y=.980,
                 fontsize=21, fontweight='semibold')
    fig.text(.54, .954, f'$T_0$ = {t:.1f} K     $P_0$ = {p / 101325:.3f} atm     '
        + r'$\phi_0$' + f' = {phi:.3f}     |     {mode.capitalize()} self-fed rollout, '
        + r'$\Delta t=2\,\mu$s', ha='center', va='center', fontsize=12, color='#414b57')
    handles = [Line2D([0], [0], color=BLACK, ls=(0, (4, 2.5)), lw=1.6, label='Cantera reference'),
        Line2D([0], [0], color=GRAY, lw=1.6, label='Baseline10k (lambda=0)'),
        Line2D([0], [0], color=BLUE, lw=1.6, label=f'Soft10k (lambda={int(weight)})'),
        Line2D([0], [0], color=ORANGE, lw=1.6, label=f'Soft long (lambda={int(weight)})')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .941), ncol=4,
               frameon=False, fontsize=11.5, handlelength=3.5, columnspacing=3)
    fig.text(.54, .021, 'Shared linear y scales by variable column; saved 2 us samples; early stops marked x.\n'
        'Baseline/short10000 updates. Long uses exact Adam/RNG continuation; endpoint updates differ and are exported. '
        'Each mixed team uses the corresponding ANN10 + DeepONet10. Joint loss plateaus and time caps are recorded separately.',
        ha='center', va='center', fontsize=9.5, color='#4b5563', linespacing=1.5)
    if len(fig.axes) != 70:
        raise AssertionError('Expected7x10 axes')
    return fig


def table_figure(frame, common, mode, weight=None, *, trajectory=None, initial=None, team=None):
    summary = trajectory is None
    part = frame if team is None else frame[frame.team == team]
    methods = list(part.method.unique()); height = max(8.5, 7 + .285 * len(methods))
    fig, ax = plt.subplots(figsize=(22, height)); ax.axis('off')
    fig.subplots_adjust(left=.02, right=.98, bottom=.16, top=.83)
    rows = []; colors = []; stages = []
    for method in methods:
        p = part[part.method == method].set_index('variable'); r = p.loc['T']
        group = next(j for j, (t, _) in enumerate(TEAMS, 1) if t == r.team)
        values = [number(p.loc[v, 'median_nrmse'] if summary else p.loc[v, 'nrmse'])
                  if summary or p.loc[v, 'complete'] else '—' for v in VARIABLES]
        status = f"{int(r.completed_all_test_ICs)}/18" if summary else (
            'Complete' if r['complete'] else f"Stop {r.last_accepted_ms:.3f} ms")
        label = dict(baseline='base10k', short='soft10k', long='long')[r.stage]
        if weight is None:
            label = f'{int(r.lambda_value)} / {label}'
        rows.append([f'{group}  {LABELS[r.team]}', label] + values + [status]); stages.append(r.stage)
        colors.append([('#f1f5f9' if group % 2 == 0 else '#ffffff')] * 13)
    table = ax.table(cellText=rows, colLabels=['Model / same-state mean', 'Stage' if weight is not None else 'lambda / stage']
        + [r'$T$'] + COLUMNS[1:] + ['Complete ICs' if summary else '10 ms status'], cellColours=colors,
        cellLoc='center', colLoc='center', bbox=[0, 0, 1, 1], colWidths=[.207, .075] + [.0625] * 10 + [.093])
    table.auto_set_font_size(False); table.set_fontsize(9.2 if len(methods) > 30 else 10)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#d8dee6'); cell.set_linewidth(.45)
        if r == 0:
            cell.set_facecolor('#213248'); cell.set_text_props(color='white', fontweight='semibold', fontsize=11)
        elif c == 0:
            cell.set_text_props(ha='left', fontweight='semibold'); cell.PAD = .035
        elif c == 1:
            cell.set_text_props(color=STAGE_COLORS[stages[r - 1]], fontweight='bold')
    prefix = f'Lambda {int(weight)}' if weight is not None else 'Lambda1/100/1000'
    if summary:
        title = f'{prefix} continuation | Cantera vs NN NRMSE'
        scope = 'all displayed variants' if team is None else f'all stages within {LABELS[team]}'
        subtitle = f'Median of per-IC NRMSE on the same {len(common)} complete ICs for {scope}'
        extra = 'Common ICs: ' + ', '.join(map(str, common)) + '.'
        if team is not None:
            extra += ' Team-specific cohorts differ; do not rank architectures using different IC sets.'
    else:
        t, p, phi = initial
        title = f'{prefix} | Test IC {trajectory:02d} | Cantera vs NN NRMSE'
        subtitle = f'T0={t:.1f} K    P0={p / 101325:.3f} atm    phi0={phi:.3f}    |    {mode.capitalize()} self-fed,10ms'
        extra = 'Each graph row includes Cantera, baseline10k, same-lambda soft10k and soft-long.'
    fig.suptitle(title, x=.5, y=.96, fontsize=22, fontweight='semibold')
    fig.text(.5, .913, subtitle, ha='center', va='center', fontsize=12, color='#475569')
    fig.text(.5, .871, r'$\mathrm{NRMSE}=\sqrt{\mathrm{mean}[(y_{Cantera}-y_{NN})^2]}/\sqrt{\mathrm{mean}[y_{Cantera}^{2}]}$',
             ha='center', va='center', fontsize=14)
    fig.text(.024, .11, 'Exact user util.py RMS normalization: equal time-sample weights; raw dimensionless ratio without x100.\n'
        '—: incomplete, no full-horizon score. N/A: empty common cohort or undefined reference RMS. Constant nonzero N2 is valid.\n'
        'Short10000 updates; long endpoints have different update counts. Both models in every mixed team use the same stage/lambda.\n'
        + extra, ha='left', va='top', fontsize=9.5, color='#475569', linespacing=1.55)
    return fig


def training_outputs(root, cfg, snapshot, destination, weight):
    records = json.loads((weight_root(root, weight) / 'final_members.json').read_text()); history = []; minibatch = []
    for r in records:
        state = json.loads(Path(r['checkpoint'], 'state.json').read_text())
        history.extend(dict(family=r['family'], **v) for v in state['full_train_history'])
        minibatch.extend(dict(family=r['family'], **v) for v in state['history'])
    history = pd.DataFrame(history); history.to_csv(destination / 'observed_full_train_components.csv', index=False)
    pd.DataFrame(minibatch).to_csv(destination / 'observed_minibatch_components.csv', index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ('monitor', 'timing', 'full_train')} for r in records]).to_csv(
        destination / 'training_summary.csv', index=False)
    fig, axes = plt.subplots(4, 3, figsize=(17, 13), sharex=True)
    for axrow, family in zip(axes, ('ann', 'p_only', 'tp', 'tp_phi')):
        selected = history[(history.family == family) & (history['update'] >= 10000)]
        for ax, key in zip(axrow, ('data_loss', 'element_loss', 'full_train_loss')):
            pivot = selected.pivot(index='seed', columns='update', values=key)
            for values in pivot.to_numpy():
                ax.plot(pivot.columns, values, color=BLUE, lw=.6, alpha=.23)
            common = pivot.dropna(axis=1); mean = np.exp(np.log(np.maximum(common.to_numpy(), 1e-300)).mean(axis=0))
            ax.plot(common.columns, mean, color=ORANGE, lw=1.5); ax.set_yscale('log')
            ax.set_title(f'{family}: {key}'); ax.set_xlabel('Optimizer update'); ax.grid(True, which='both', ls=':', alpha=.3)
    fig.suptitle(f'Lambda{int(weight)} continuation from10000: all member tails; means only on common observed updates. Element term is unweighted.')
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(destination / 'training_loss_components.pdf')
    fig.savefig(destination / 'training_loss_components.png', dpi=160); plt.close(fig)
    baseline_root = Path(snapshot['origin_snapshot']['config']['origin_mu10'])
    base = pd.read_csv(baseline_root / 'member_diagnostics/loss_components.csv')
    base = base[base.stage == 'baseline'].copy(); base['lambda_value'] = 0
    base['objective'] = base.data_MSE; base['weighted_element_L1'] = 0
    columns = ['stage', 'lambda_value', 'family', 'seed', 'split', 'pairs', 'objective', 'data_MSE',
               'physical_element_L1', 'weighted_element_L1', 'invalid_inverse_row_fraction']
    frames = [base[columns]]
    for stage, parent in [('short', weight_root(cfg['origin'], weight)), ('long', weight_root(root, weight))]:
        p = pd.read_csv(parent / 'member_diagnostics/loss_components.csv'); p['stage'] = stage; frames.append(p[columns])
    diagnostics = pd.concat(frames, ignore_index=True)
    diagnostics.to_csv(destination / 'member_loss_components.csv', index=False)
    diagnostics.groupby(['stage', 'lambda_value', 'family', 'split'])[columns[6:]].mean().to_csv(
        destination / 'mean_member_loss_components.csv')
    return records, diagnostics


def verify(root, cfg, snapshot):
    records = json.loads((root / 'final_members.json').read_text()); checks = 0
    if len(records) != 120:
        raise ValueError('Expected120 terminal models')
    for r in records:
        cp = Path(r['checkpoint']); state = json.loads((cp / 'state.json').read_text())
        marker = json.loads((cp / 'complete.json').read_text())
        for name, expected in marker['output_sha256'].items():
            if sha256(cp / name) != expected:
                raise ValueError('Terminal checkpoint changed')
        with np.load(cp / 'adam.npz') as adam:
            if int(adam['v0']) != r['updates']:
                raise ValueError('Actual Adam iteration differs')
        origin = next(v for v in snapshot['origin_members'] if (v['mu'], v['family'], v['seed']) == (r['mu'], r['family'], r['seed']))
        before = json.loads(Path(origin['checkpoint'], 'state.json').read_text())
        if (r['updates'] != state['update'] or state['update'] != state['optimizer_iteration']
                or state['signature'] != before['signature'] or r['updates'] < 10000
                or sha256(cp / 'model.json') != origin['preprocessing_sha256']):
            raise ValueError('Continuation config, iteration or preprocessing changed')
        if state['history'][:len(before['history'])] != before['history'] or state['full_train_history'][:len(before['full_train_history'])] != before['full_train_history']:
            raise ValueError('Original observed histories changed')
        saved = state['plateau_monitor']
        if saved['long_run_sha256'] != sha256(root / 'run.json'):
            raise ValueError('Joint-monitor lineage differs')
        replay = joint_monitor(cfg['plateau'], saved['long_run_sha256'], saved['origin_update'], saved['origin_seconds'])
        for row in saved['checks']:
            joint_check(state['full_train_history'], row['update'], replay); checks += 1
        if replay['checks'] != saved['checks'] or replay['components'] != saved['components']:
            raise ValueError('Joint-plateau replay differs')
        if (r['reason'] == 'joint_plateau') != bool(saved['plateau']):
            raise ValueError('A time cap was mislabeled as convergence')
    for p, h in snapshot['source_sha256'].items():
        if sha256(p) != h:
            raise ValueError('Frozen continuation/report source changed')
    old = json.loads((root / 'original_results_sha256.json').read_text())
    for p, h in old.items():
        if sha256(p) != h:
            raise ValueError(f'Preexisting result changed: {p}')
    return records, checks, len(old)


def write_summaries(frame, parent, mode, weight, pdf_metadata):
    summary, within_summary, common, within = summaries(frame)
    frame.to_csv(parent / 'per_IC_NRMSE.csv', index=False)
    summary.to_csv(parent / 'common_IC_NRMSE_summary.csv', index=False)
    within_summary.to_csv(parent / 'within_team_NRMSE_summary.csv', index=False)
    fig = table_figure(summary, common, mode, weight); fig.savefig(parent / 'NRMSE_summary.pdf')
    fig.savefig(parent / 'NRMSE_summary.png', dpi=160); plt.close(fig)
    with PdfPages(parent / 'within_team_NRMSE.pdf', metadata=pdf_metadata) as pdf:
        for team, _ in TEAMS:
            fig = table_figure(within_summary, within[team], mode, weight, team=team); pdf.savefig(fig); plt.close(fig)
    return common, within


def report(root, cfg, snapshot, data, deadline):
    started = time.perf_counter(); destination = root / 'reports' / f'report_{time.time_ns()}'
    destination.mkdir(parents=True, exist_ok=False); ids = np.flatnonzero(data['split'] == 'test').astype(int).tolist()
    sources = {}; comparisons = {}; cross = {'raw': [], 'repaired': []}; diagnostics = []
    write_json(destination / 'run_snapshot.json', snapshot)
    with plt.rc_context(STYLE):
        for weight in cfg['lambdas']:
            parent = weight_root(destination, weight); parent.mkdir()
            records, diag = training_outputs(root, cfg, snapshot, parent, weight); diagnostics.append(diag[diag.stage != 'baseline'])
            comparisons[str(int(weight))] = {}
            for mode in ('repaired', 'raw'):
                check_deadline(deadline); traces, used = traces_for(root, cfg, snapshot, data, weight, mode); sources.update(used)
                frame = metrics_frame(data, traces, ids, mode, weight)
                folder = parent / mode; folder.mkdir(); png = folder / 'png'; png.mkdir(); tables = folder / 'tables'; tables.mkdir()
                common, within = write_summaries(frame, folder, mode, weight, {'Title': f'Lambda{int(weight)} within-team NRMSE'})
                comparisons[str(int(weight))][mode] = dict(common_ICs=common, within_team_common_ICs=within)
                subset = frame[frame.stage != 'baseline'].copy()
                subset['method'] = subset['method'].map(lambda m: f'lambda{int(weight)}_{m}'); cross[mode].append(subset)
                with PdfPages(folder / 'profiles_and_NRMSE.pdf') as pdf:
                    for page, i in enumerate(ids, 1):
                        check_deadline(deadline); fig = profile_figure(data, traces, i, mode, weight)
                        pdf.savefig(fig); fig.savefig(png / f'test_IC{i:02d}_7x10.png', dpi=160); plt.close(fig)
                        fig = table_figure(frame[frame.trajectory == i], common, mode, weight, trajectory=i, initial=data['initials'][i])
                        pdf.savefig(fig); fig.savefig(tables / f'test_IC{i:02d}_NRMSE.png', dpi=160); plt.close(fig)
                        event(root, phase='report_page_finished', lambda_value=weight, mode=mode, page=page, trajectory=i)
                write_json(folder / 'complete.json', dict(pages=36, axes_per_profile=70, common_ICs=common,
                    within_team_common_ICs=within, output_sha256={str(p.relative_to(folder)): sha256(p)
                    for p in folder.rglob('*') if p.is_file()}))
        pd.concat(diagnostics, ignore_index=True).to_csv(destination / 'all_weight_member_loss_components.csv', index=False)
        cross_comparison = {}
        for mode in ('repaired', 'raw'):
            folder = destination / f'cross_weight_{mode}'; folder.mkdir(); frame = pd.concat(cross[mode], ignore_index=True)
            common, within = write_summaries(frame, folder, mode, None, {'Title': 'Across weights and short/long stages'})
            cross_comparison[mode] = dict(common_ICs=common, within_team_common_ICs=within)
    records, checks, preserved = verify(root, cfg, snapshot)
    flat = sum(r['reason'] == 'joint_plateau' for r in records); capped = sum(r['reason'] == 'time_limit' for r in records)
    write_json(destination / 'trace_sources.json', sources)
    write_json(destination / 'verification.json', dict(terminal_checkpoints=120, actual_Adam_iterations_verified=120,
        original_histories_and_preprocessors_preserved=120, joint_checks_recomputed=checks,
        preexisting_files_preserved=preserved, joint_plateau_members=flat, time_limited_members=capped,
        continued_members=sum(r['total_added_updates'] > 0 for r in records), per_weight=comparisons, cross_weight=cross_comparison))
    lines = ['lambda1,100,1000의120개 모델을 기존10000회 체크포인트에서 이어 학습했다.',
        f'공동 평탄 {flat}개, 배정 시간상한 종료 {capped}개. 전체상한6시간 중1시간을 평가/그림에 유보했다.',
        '학습/평가/그림은 순차, CPU intra2/inter1 및 OMP/BLAS2/oneDNN0을 유지했다.',
        '가중치·데이터·전처리·구조·Adam/RNG·절대LR(.0001이후유지)은 그대로며, update상한을 추가하지 않았다.',
        '공동 평탄: 최소50k, 관측1k/검사5k, 연속20k창의q25/median/q75상대변화1%/1%/3%,3회 연속10k지속을 전체/MSE/원소L1 모두 통과해야 한다.',
        '미도달 시간상한을 수렴으로 표시하지 않았다. 모델별 종료update와 실제시간은각 training_summary.csv에 있다.',
        '가중치별7x10 그림은Cantera/무제약10k/동일lambda soft10k/soft장기 네 곡선이다. 각IC 그림 뒤에 같은NRMSE표를 넣었다.',
        'NRMSE=RMSE(Cantera,NN)/RMS(Cantera), util.py기본값·동일샘플가중치·무차원비율. 미완주에는전체점수를 부여하지 않는다.',
        '각요약은명시된공통완주IC 중앙값이다. 별도 within-team 표는각팀내공통IC만쓰며 팀마다집합이다를수있어 architecture순위로쓰면안된다.',
        'cross_weight 표는이번 세lambda의short/long42구성을비교한다. 장기endpoint update가다르므로 완전수렴 후동일회차lambda순위로해석하면안된다.',
        '기존lambda10장기는더큰모델당시간과다른update를사용했으므로이번주요lambda비교에합치지않았다. 이전보고서는원위치에보존했다.',
        '이테스트IC들은이미이전실험에서관찰했다. 본비교는탐색적민감도/추가학습평가다. 테스트로학습종료점이나lambda를조정하지않았다.',
        f'기존결과파일{preserved}개SHA불변,120개endpoint와공동평탄검사{checks}개를검증했다.', '']
    for w in cfg['lambdas']:
        part = [r for r in records if r['mu'] == w]
        lines += [f'lambda{int(w)}: update{min(r["updates"] for r in part)}–{max(r["updates"] for r in part)}, 공동평탄{sum(r["reason"]=="joint_plateau" for r in part)}/40.',
            f'[lambda{int(w)} repaired 그림+표](lambda_{int(w)}/repaired/profiles_and_NRMSE.pdf)',
            f'[lambda{int(w)} raw 그림+표](lambda_{int(w)}/raw/profiles_and_NRMSE.pdf)']
    (destination / 'REPORT_KO.md').write_text('\n'.join(lines) + '\n')
    for w in cfg['lambdas']:
        for mode in ('repaired', 'raw'):
            check_deadline(deadline); parent = weight_root(destination, w)
            archive = destination / f'lambda_{int(w)}_{mode}_bundle.zip'
            with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                files = [p for p in parent.iterdir() if p.is_file()] + [p for p in (parent / mode).rglob('*') if p.is_file()]
                files += [destination / n for n in ('REPORT_KO.md', 'verification.json', 'run_snapshot.json', 'trace_sources.json')]
                for p in files:
                    z.write(p, arcname=p.relative_to(destination))
    write_json(destination / 'complete.json', dict(output_sha256={str(p.relative_to(destination)): sha256(p)
        for p in destination.rglob('*') if p.is_file()}))
    finished = time.time(); start = json.loads((root / 'budget.json').read_text())['started_unix']
    write_json(root / 'FINAL_ARTIFACTS.json', dict(report=str(destination), report_complete_sha256=sha256(destination / 'complete.json'),
        members=120, lambdas=cfg['lambdas'], joint_plateau_members=flat, time_limited_members=capped,
        per_weight=comparisons, cross_weight=cross_comparison, pipeline_seconds=finished - start, finished_unix=finished))
    event(root, phase='report_finished', path=str(destination), seconds=time.perf_counter() - started)
    return destination
