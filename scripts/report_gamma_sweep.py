"""Matched7x10 profiles and exact util.py NRMSE across six Box-Cox gammas."""
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
from kinetics.gamma_sweep_training import gamma_tag
from kinetics.gamma_sweep_evaluation import load_profiles
from kinetics.soft_constraint_evaluation import TEAMS
from kinetics.soft_constraint_study import event
from kinetics.p_only_training import sha256, write_json
from scripts.plot_kinetics_7x10_comparison import COLUMNS, STYLE, BLACK
from scripts.report_soft_constraint_comparison import LABELS, limits
from scripts.report_kinetics_nrmse import profile_errors, number, VARIABLES

GAMMAS = (.05, .1, .2, .3, .5, 1.)
COLORS = {.05: '#aa3377', .1: '#64748b', .2: '#0077bb', .3: '#009988', .5: '#ee7733', 1.: '#cc3311'}
METHODS = [(f'{gamma_tag(g)}_{team}', team, g) for team, _ in TEAMS for g in GAMMAS]


def normalized_metrics(data, traces, ids, split, mode):
    rows = []
    for i in ids:
        for method, team, gamma in METHODS:
            trace = traces[method][i]; r = profile_errors(data['states'][i], trace, float(data['dt']))
            for k, variable in enumerate(VARIABLES):
                rows.append(dict(trajectory=i, method=method, team=team, model=LABELS[team], gamma=gamma,
                    variable=variable, split=split, mode=mode, complete=r['complete'], points=r['points'],
                    last_accepted_ms=(len(trace) - 1) * float(data['dt']) * 1000,
                    nrmse_status=r['status'][k], nrmse=float(r['nrmse'][k]), full_rmse=float(r['full_rmse'][k]),
                    prefix_rmse=float(r['prefix_rmse'][k]), reference_rms=float(r['reference_rms'][k])))
    frame = pd.DataFrame(rows)
    completed = {m: set(p.loc[p.complete, 'trajectory']) for m, p in frame.groupby('method')}
    common = sorted(set.intersection(*completed.values()))
    within = {team: sorted(set.intersection(*(completed[f'{gamma_tag(g)}_{team}'] for g in GAMMAS))) for team, _ in TEAMS}
    return frame, summarize(frame, {t: common for t, _ in TEAMS}, 'all42variants'), summarize(frame, within, 'within_team_all6gammas'), common, within


def summarize(frame, cohorts, scope):
    rows = []
    for (method, variable), part in frame.groupby(['method', 'variable'], sort=False):
        first = part.iloc[0]; ids = cohorts[first.team]; values = part.loc[part.trajectory.isin(ids), 'nrmse'].dropna()
        rows.append(dict(method=method, team=first.team, model=first.model, gamma=first.gamma, variable=variable,
            split=first['split'], mode=first['mode'], common_scope=scope, common_ICs=json.dumps(ids),
            common_IC_count=len(ids), valid_nrmse_IC_count=len(values), median_nrmse=values.median(),
            mean_nrmse=values.mean(), std_nrmse=values.std(ddof=0), q25_nrmse=values.quantile(.25), q75_nrmse=values.quantile(.75),
            completed_all_ICs=int(part.complete.sum())))
    return pd.DataFrame(rows)


def profile_figure(data, traces, i, split, mode):
    truth = data['states'][i]; dt = float(data['dt']); ylimits = limits(truth, traces, i)
    fig, axes = plt.subplots(7, 10, figsize=(31, 20.2), sharex=True, squeeze=False)
    fig.subplots_adjust(left=.087, right=.990, bottom=.062, top=.890, wspace=.36, hspace=.38)
    for row, (team, _) in enumerate(TEAMS):
        failed = []
        for g in GAMMAS:
            trace = traces[f'{gamma_tag(g)}_{team}'][i]
            if len(trace) < len(truth):
                failed.append(f'g={g:g}: stop {(len(trace)-1)*dt*1000:.3f} ms')
        for k, ax in enumerate(axes[row]):
            for g in GAMMAS:
                trace = traces[f'{gamma_tag(g)}_{team}'][i]; tt = np.arange(len(trace)) * dt * 1000
                ax.plot(tt, trace[:, k], color=COLORS[g], lw=1.0, alpha=.93, zorder=2)
                if len(trace) < len(truth):
                    ax.plot(tt[-1], trace[-1, k], marker='x', color=COLORS[g], ms=3.5, mew=.9, zorder=5)
            ax.plot(np.arange(len(truth)) * dt * 1000, truth[:, k], color=BLACK, ls=(0, (4, 2.5)), lw=1.15, zorder=3)
            ax.set_xlim(0, (len(truth)-1)*dt*1000); ax.set_ylim(*ylimits[k]); ax.set_xticks([0, 2, 4, 6, 8, 10])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            formatter = ScalarFormatter(useOffset=False, useMathText=True); formatter.set_powerlimits((-3, 3))
            ax.yaxis.set_major_formatter(formatter); ax.yaxis.get_offset_text().set_size(7.5)
            ax.grid(True, ls=':', lw=.45, color='#b7c1cc', alpha=.65); ax.tick_params(pad=2)
            if row == 0:
                ax.set_title(COLUMNS[k], pad=12, fontweight='semibold')
            if row == 6:
                ax.set_xlabel('Time [ms]', labelpad=5)
        position = axes[row, 0].get_position(); label = LABELS[team].replace(' DeepONet10', '\nDeepONet10')
        fig.text(.009, (position.y0+position.y1)/2, f'{row+1}  {label}', ha='left', va='center', fontsize=11, fontweight='semibold')
        if failed:
            axes[row, 0].text(.985, .97, '\n'.join(failed), transform=axes[row, 0].transAxes,
                ha='right', va='top', fontsize=6.3, bbox=dict(facecolor='white', alpha=.88, edgecolor='none', pad=1.8))
    t0, p0, phi = data['initials'][i]
    fig.suptitle(f'Box-Cox gamma sweep: Cantera vs NN | {split.capitalize()} IC {i:02d}', x=.54, y=.980, fontsize=21, fontweight='semibold')
    fig.text(.54, .954, f'$T_0$={t0:.1f} K    $P_0$={p0/101325:.3f} atm    '+r'$\phi_0$'+f'={phi:.3f}    |    {mode.capitalize()} self-fed rollout, '+r'$\Delta t=2\,\mu$s', ha='center', fontsize=12, color='#414b57')
    handles = [Line2D([0], [0], color=BLACK, ls=(0, (4, 2.5)), lw=1.6, label='Cantera')]
    handles += [Line2D([0], [0], color=COLORS[g], lw=1.6, label=f'gamma={g:g}'+(' (reused reference)' if g==.1 else '')) for g in GAMMAS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .941), ncol=4, frameon=False, fontsize=11, handlelength=3.1)
    fig.text(.54, .021, 'New members: up to10000 updates or150 seconds each, pure data MSE; actual updates/stop reasons exported.\n'
        'Each mixed team uses ANN10 + DeepONet10 at the SAME gamma and its own current state. Shared y scales; saved2us samples; early stops marked x.\n'
        'Gamma=.1 models are frozen controls; historical training time is separate. Raw/repaired retain the original decoding and physical correction.',
        ha='center', va='center', fontsize=9.5, color='#4b5563', linespacing=1.4)
    return fig


def table_figure(frame, common, split, mode, *, trajectory=None, initial=None, team=None):
    summary = trajectory is None; methods = [(m, t, g) for m, t, g in METHODS if team is None or t == team]
    fig, ax = plt.subplots(figsize=(24, 21 if team is None else 8.5)); ax.axis('off')
    fig.subplots_adjust(left=.02, right=.98, bottom=.15, top=.84); rows = []; colors = []
    for method, model, gamma in methods:
        part = frame[frame.method == method].set_index('variable')
        values = [number(part.loc[v, 'median_nrmse' if summary else 'nrmse']) if summary or part.loc[v, 'complete'] else '—' for v in VARIABLES]
        status = f"{int(part.loc['T', 'completed_all_ICs'])}/18" if summary else ('Complete' if part.loc['T', 'complete'] else f"Stop {part.loc['T', 'last_accepted_ms']:.3f} ms")
        group = next(j for j, (t, _) in enumerate(TEAMS, 1) if t == model)
        rows.append([f'{group}  {LABELS[model]}', f'{gamma:g}'] + values + [status])
        colors.append([('#f1f5f9' if group % 2 == 0 else 'white')] * 13)
    table = ax.table(cellText=rows, colLabels=['Model / same-state mean', 'gamma']+VARIABLES+['10 ms status'],
        cellColours=colors, colLoc='center', cellLoc='center', colWidths=[.24, .06]+[.06]*10+[.10], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(8.5 if team is None else 11)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#cbd5e1'); cell.set_linewidth(.45)
        if r == 0:
            cell.set_facecolor('#20344c'); cell.set_text_props(color='white', weight='bold')
        elif c == 0:
            cell.set_text_props(ha='left', weight='semibold')
    label = f'IC {trajectory:02d}' if not summary else (f'{LABELS[team]}: common ICs {common}' if team else f'All42 variants: common ICs {common}')
    fig.suptitle(f'Box-Cox gamma sweep | {split.capitalize()} {label}', fontsize=18, fontweight='semibold', y=.96)
    subtitle = f'{mode.capitalize()} | Cantera vs NN NRMSE | '+('Median of IC-wise scores; completion counts use all18 ICs' if summary else f'T0={initial[0]:.1f}K, P0={initial[1]/101325:.3f}atm, phi0={initial[2]:.3f}')
    fig.text(.5, .91, subtitle, ha='center', fontsize=12, color='#475569')
    fig.text(.03, .09, 'NRMSE = RMSE(Cantera,NN) / RMS(Cantera). Exact user util.py default; equal time-point weights; raw ratio without ×100.\n'
        '—: incomplete, no full-horizon score. N/A: empty common cohort or undefined reference RMS. Nonzero constant N2 is valid.\n'
        'Gamma is the Box-Cox exponent; soft constraint is absent. New models stop at10000 updates or150 seconds; actual endpoints are exported. Cantera + six gamma curves.',
        fontsize=10.5, va='center', linespacing=1.5)
    return fig


def training_outputs(root, destination, records):
    pd.DataFrame(records).drop(columns=['timing'], errors='ignore').to_csv(destination / 'training_summary.csv', index=False)
    rows = []
    for r in records:
        if r.get('reused'):
            continue
        state = json.loads((Path(r['checkpoint']) / 'state.json').read_text())
        if not state['full_train_history'] or state['full_train_history'][0]['full_train_loss'] <= 0:
            continue
        initial = state['full_train_history'][0]['full_train_loss']
        rows.extend(dict(h, gamma=r['gamma'], family=r['family'], seed=r['seed'],
                         loss_ratio_to_initial=h['full_train_loss']/initial) for h in state['full_train_history'])
    history = pd.DataFrame(rows, columns=['gamma','family','seed','update','full_train_loss','loss_ratio_to_initial'])
    history.to_csv(destination / 'observed_full_train_MSE.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, family in zip(axes.flat, ('ann', 'p_only', 'tp', 'tp_phi')):
        for g in GAMMAS:
            part = history[(history.family == family) & (history.gamma == g)]
            if not len(part):
                continue
            pivot = part.pivot(index='seed', columns='update', values='loss_ratio_to_initial').dropna(axis=1)
            for values in pivot.to_numpy():
                ax.plot(pivot.columns, values, color=COLORS[g], lw=.6, alpha=.16)
            mean = np.exp(np.log(np.maximum(pivot.to_numpy(), 1e-300)).mean(axis=0))
            ax.plot(pivot.columns, mean, color=COLORS[g], lw=1.5, label=f'g={g:g}')
        ax.set_title(family); ax.set_yscale('log'); ax.set_xlabel('Optimizer update'); ax.set_ylabel('Full-train MSE / initial MSE')
        ax.grid(True, which='both', ls=':', alpha=.3); ax.legend(fontsize=8)
    fig.suptitle('Pure-MSE gamma screening: loss reduction within each transform. Physical accuracy is compared with Cantera NRMSE.')
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(destination / 'training_loss_ratios.pdf'); fig.savefig(destination / 'training_loss_ratios.png', dpi=150); plt.close(fig)


def verify(root, snapshot):
    records = json.loads((root / 'final_members.json').read_text()); new = [r for r in records if not r.get('reused')]
    if len(records) != 240 or len(new) != 200:
        raise ValueError('Expected200 new and40 cached models')
    initialized = {}
    for r in new:
        cp = Path(r['checkpoint']); marker = json.loads((cp / 'complete.json').read_text()); state = json.loads((cp / 'state.json').read_text())
        for name, expected in marker['output_sha256'].items():
            if sha256(cp / name) != expected:
                raise ValueError('Checkpoint changed')
        with np.load(cp / 'adam.npz') as f:
            if int(f['v0']) != r['updates']:
                raise ValueError('Actual Adam update differs')
        if not (r['updates'] == state['update'] == state['optimizer_iteration'] and 0 <= r['updates'] <= 10000
                and r['reason'] == ('fixed_update_budget' if r['updates'] == 10000 else 'member_time_limit')
                and state['reason'] == r['reason'] and r['member_budget']['seconds'] == 150
                and state['signature']['loss'] == 'transformed_residual_MSE_only' and not state['plateau_monitor']['enabled']):
            raise ValueError('Incorrect training objective or budget')
        info = json.loads((cp / 'model.json').read_text())
        if info['spec']['gamma'] != r['gamma'] or state['signature']['gamma'] != r['gamma']:
            raise ValueError('Gamma provenance differs')
        init = json.loads((root / gamma_tag(r['gamma']) / 'members' / r['family'] / f"seed{r['seed']}" / 'initialization.json').read_text())
        key = (r['family'], r['seed']); paired = (init['initial_weights_sha256'], init['initial_numpy_rng'])
        if key in initialized and initialized[key] != paired:
            raise ValueError('Initialization/minibatch pairing changed across gammas')
        initialized[key] = paired
    for mapping in ('source_sha256', 'origin_sha256'):
        for path, expected in snapshot[mapping].items():
            if sha256(path) != expected:
                raise ValueError('Frozen source changed')
    previous = json.loads((root / 'original_results_sha256.json').read_text())
    for path, expected in previous.items():
        if sha256(path) != expected:
            raise ValueError(f'Preexisting result changed: {path}')
    return records, len(previous)


def selected_comparison(frame, selection):
    rows = []
    for team, _ in TEAMS:
        chosen = selection['choices'][team]['chosen_gamma']
        if chosen is None:
            continue
        a = frame[(frame.team == team) & (frame.gamma == .1)]
        b = frame[(frame.team == team) & (frame.gamma == chosen)]
        common = sorted(set(a.loc[a.complete, 'trajectory']) & set(b.loc[b.complete, 'trajectory']))
        for variable in VARIABLES:
            av, bv = a[a.variable == variable], b[b.variable == variable]
            rows.append(dict(team=team, model=LABELS[team], selected_gamma=chosen, variable=variable,
                common_ICs=json.dumps(common), common_IC_count=len(common),
                reference_median_NRMSE=av.loc[av.trajectory.isin(common), 'nrmse'].median(),
                selected_median_NRMSE=bv.loc[bv.trajectory.isin(common), 'nrmse'].median(),
                reference_completed=int(av.complete.sum()), selected_completed=int(bv.complete.sum())))
    return pd.DataFrame(rows)


def report(root, cfg, snapshot, data, deadline):
    started = time.perf_counter(); destination = root / 'reports' / f'report_{time.time_ns()}'
    destination.mkdir(parents=True, exist_ok=False); comparisons = {}; sources = {}
    write_json(destination / 'run_snapshot.json', snapshot)
    selection = json.loads((root / 'validation_selection.json').read_text())
    write_json(destination / 'validation_selection.json', selection)
    records = json.loads((root / 'final_members.json').read_text())
    with plt.rc_context(STYLE):
        training_outputs(root, destination, records)
        for split in cfg['evaluation_splits']:
            comparisons[split] = {}; ids = np.flatnonzero(data['split'] == split).astype(int).tolist()
            for mode in ('repaired', 'raw'):
                check_deadline(deadline); traces, used = load_profiles(root, cfg, snapshot, data, split, mode)
                sources.update({str(info['path']): info['sha256'] for info in used.values()})
                frame, summary, within_summary, common, within = normalized_metrics(data, traces, ids, split, mode)
                folder = destination / split / mode; folder.mkdir(parents=True); png = folder / 'png'; png.mkdir(); tables = folder / 'tables'; tables.mkdir()
                frame.to_csv(folder / 'per_IC_NRMSE.csv', index=False)
                summary.to_csv(folder / 'common_IC_NRMSE_summary.csv', index=False)
                within_summary.to_csv(folder / 'within_team_NRMSE_summary.csv', index=False)
                comparisons[split][mode] = dict(common_ICs=common, within_team_common_ICs=within)
                fig = table_figure(summary, common, split, mode); fig.savefig(folder / 'NRMSE_summary.pdf'); fig.savefig(folder / 'NRMSE_summary.png', dpi=150); plt.close(fig)
                with PdfPages(folder / 'within_team_NRMSE.pdf') as pdf:
                    for team, _ in TEAMS:
                        fig = table_figure(within_summary, within[team], split, mode, team=team); pdf.savefig(fig); plt.close(fig)
                if split == 'test':
                    selected_comparison(frame, selection).to_csv(folder / 'validation_selected_vs_reference.csv', index=False)
                with PdfPages(folder / 'profiles_and_NRMSE.pdf') as pdf:
                    for page, i in enumerate(ids, 1):
                        check_deadline(deadline); fig = profile_figure(data, traces, i, split, mode)
                        pdf.savefig(fig); fig.savefig(png / f'{split}_IC{i:02d}_7x10.png', dpi=160); plt.close(fig)
                        fig = table_figure(frame[frame.trajectory == i], common, split, mode, trajectory=i, initial=data['initials'][i])
                        pdf.savefig(fig); fig.savefig(tables / f'{split}_IC{i:02d}_NRMSE.png', dpi=150); plt.close(fig)
                        event(root, phase='report_page_finished', split=split, mode=mode, page=page, trajectory=i)
                write_json(folder / 'complete.json', dict(pages=36, axes_per_profile=70, common_ICs=common,
                    within_team_common_ICs=within, output_sha256={str(p.relative_to(folder)): sha256(p) for p in folder.rglob('*') if p.is_file()}))
                del traces
    records, preserved = verify(root, snapshot)
    fresh = [r for r in records if not r.get('reused')]
    completed = sum(r['updates'] == 10000 for r in fresh)
    write_json(destination / 'trace_sources.json', sources)
    write_json(destination / 'verification.json', dict(new_terminal_checkpoints=200, reused_models=40, actual_Adam_iterations_verified=200,
        paired_initialization_and_rng_groups=40, pure_MSE=True, original_files_preserved=preserved,
        completed_10000_updates=completed, time_limited_members=200-completed, member_budget_seconds=150,
        test_used_for_selection=False, comparison=comparisons))
    (destination / 'REPORT_KO.md').write_text('\n'.join([
        'Box-Cox gamma=0.05,0.1,0.2,0.3,0.5,1.0 비교. soft constraint 없이 정규화된 증분 MSE만 학습했다.',
        '기존gamma.1의40개 모델/시험궤적을 재사용하고 나머지5개gamma의200개 모델을10000update 또는모델당150초상한으로 순차 학습했다.',
        f'10000회완료{completed}/200, 모델시간상한{200-completed}/200. 모델별실제update와종료사유를training_summary.csv에기록했다.',
        'ANN10/P-only10/ANN+P20/TP10/ANN+TP20/TPphi10/ANN+TPphi20의7개 팀. 혼합팀의 두 계열은 같은gamma를 쓴다.',
        '데이터8815쌍,2us target,10ms horizon,구조/seed/Adam/배치512/clip5/LR스케줄/추론보정은 고정했다.',
        '각gamma에서 학습current+next 상태의 공통스케일, 학습current 입력스케일을 다시 맞췄다. validation/test로 스케일을 fit하지 않았다.',
        '새gamma들의 초기가중치와미니배치RNG를 seed/계열별로 짝지었다. gamma.1의 과거 ANN grouped time은 신규 serial time과 별개다.',
        'gamma마다 변환/목표척도가 달라 raw training MSE의 크기로 gamma 성능순위를 매기면 안 된다. 학습그래프는 초기MSE대비 비율이다.',
        '모든 endpoint를 고정하고 validation을 먼저 평가했다. 대표gamma는 repaired 완주수 우선, 동률은공통IC 온도NRMSE 중앙값으로 선택했다.',
        'validation_selection.json을 저장한 뒤test를 평가했다. 이미 과거에 관찰한test의 탐색적비교이며 독립최종검증을 주장하지 않는다.',
        'NRMSE=RMSE(Cantera,NN)/RMS(Cantera), user util.py 기본값,동일시간점가중치,무차원비율. 미완주에는 전체점수를 부여하지 않는다.',
        '각IC의7x10그림 바로 뒤에 같은구성의NRMSE표를 넣었다. 검정Cantera+6개gamma곡선이다.',
        '전체요약은42개구성 공통완주IC, 팀별요약은해당팀6개gamma 공통완주IC를 쓴다. 빈집합은N/A이며 팀마다IC집합이 다를수있다.',
        'test 각모드의 validation_selected_vs_reference.csv는 validation에서 고른gamma와.1의 공통완주IC에서10개변수를 비교한다.',
        f'신규200개실제Adam iteration과기록update일치/순수MSE/초기화40그룹 및기존파일{preserved}개의SHA보존을검증했다.',
        '', '[Test repaired 그림+표](test/repaired/profiles_and_NRMSE.pdf)',
        '[Test raw 그림+표](test/raw/profiles_and_NRMSE.pdf)',
        '[팀별gamma NRMSE](test/repaired/within_team_NRMSE.pdf)',
        '[Validation 그림+표](val/repaired/profiles_and_NRMSE.pdf)', ''])+'\n')
    for split in cfg['evaluation_splits']:
        for mode in ('repaired', 'raw'):
            check_deadline(deadline); archive = destination / f'{split}_{mode}_bundle.zip'
            shared = [p for p in destination.iterdir() if p.is_file() and p.suffix != '.zip']
            with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                for p in shared + [p for p in (destination / split / mode).rglob('*') if p.is_file()]:
                    z.write(p, arcname=p.relative_to(destination))
    write_json(destination / 'complete.json', dict(output_sha256={str(p.relative_to(destination)): sha256(p) for p in destination.rglob('*') if p.is_file()}))
    budget = json.loads((root / 'budget.json').read_text())
    write_json(root / 'FINAL_ARTIFACTS.json', dict(report=str(destination), report_complete_sha256=sha256(destination / 'complete.json'),
        new_members=200, reused_models=40, update_ceiling_per_new_member=10000, member_budget_seconds=150,
        completed_10000_updates=completed, time_limited_members=200-completed, gammas=cfg['gammas'],
        comparison=comparisons, validation_selection=selection, pipeline_seconds=time.time()-budget['started_unix']))
    event(root, phase='report_finished', path=str(destination), seconds=time.perf_counter()-started)
    return destination
