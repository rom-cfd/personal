"""Scientific figures and plain-language audit for a frozen React-DeepONet run."""

import csv
import html
import json
from pathlib import Path

import numpy as np

from .react_deeponet_experiment import csv_write, sha, write


MODES = ['representative_raw', 'raw_full', 'repaired_full']
LABELS = ['Representative raw', 'Full state raw', 'Full state repaired']


def summary_rows(rows):
    summary = []
    for split in sorted({r['split'] for r in rows}):
        for mode in MODES:
            selected = [r for r in rows if r['split'] == split and r['mode'] == mode]
            complete = [r for r in selected if r['complete'] == 'True']
            values = [float(r['worst_NRMSE']) for r in complete]
            summary.append(dict(split=split, mode=mode, total=len(selected), complete=len(complete),
                joint_pass=sum(r['joint_pass'] == 'True' for r in selected),
                criterion='joint5' if mode == 'representative_raw' else 'joint9',
                median_worst_NRMSE=float(np.median(values)) if values else None,
                maximum_worst_NRMSE=max(values) if values else None))
    return summary


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    root = Path(root)
    out = root/'report'
    if any(out.iterdir()):
        raise FileExistsError('Do not overwrite a previously generated report')
    (out/'figures').mkdir()
    cfg = json.loads((root/'config.json').read_text())
    metadata = json.loads((root/'dataset.json').read_text())
    selection = json.loads((root/'selection_frozen.json').read_text())
    reconstruction = json.loads((root/'reconstruction_selection.json').read_text())
    cases = {c['case_id']: c for c in json.loads((root/'cases.json').read_text())}
    variables = metadata['variables']
    rep_names = cfg['reported_settings']['representative_variables']
    indices = [variables.index(v) for v in rep_names]
    columns = [j for j, name in enumerate(variables) if name != 'AR']
    dt = metadata['dt']
    rows = list(csv.DictReader((root/'evaluations/rollouts.csv').open()))
    summary = summary_rows(rows)
    csv_write(out/'test_summary.csv', summary)
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .15})
    pdf_paths = []
    with np.load(root/'evaluations/traces.npz') as traces:
        for split in sorted({r['split'] for r in rows}):
            chosen = sorted({int(r['case_id']) for r in rows if r['split'] == split})
            for zoom in [False, True]:
                filename = f'{split}_'+('ignition_zoom' if zoom else 'profiles')+'.pdf'
                pdf_paths.append(filename)
                with PdfPages(out/filename) as pdf:
                    for number in chosen:
                        case = cases[number]
                        reference = traces[f'IC{number}_reference']
                        fig, axes = plt.subplots(3, len(columns), figsize=(25, 7), squeeze=False)
                        for row_index, mode in enumerate(MODES):
                            prediction = traces[f'IC{number}_{mode}']
                            info = next(r for r in rows if int(r['case_id']) == number and r['mode'] == mode)
                            for col_index, column in enumerate(columns):
                                ax = axes[row_index, col_index]
                                ax.plot(np.arange(len(reference))*dt*1000, reference[:, column], color='black', lw=1.5, label='Cantera')
                                if mode != 'representative_raw' or column in indices:
                                    pcol = indices.index(column) if mode == 'representative_raw' else column
                                    ax.plot(np.arange(len(prediction))*dt*1000, prediction[:, pcol], color='#2563eb', ls='--', lw=1.2, label='NN')
                                else:
                                    ax.text(.5, .5, 'Not advanced\nby operator', transform=ax.transAxes, ha='center', fontsize=7)
                                ax.set_xlim(0, .1 if zoom else .5)
                                ax.ticklabel_format(axis='y', style='plain', useOffset=False)
                                if column != 0:
                                    ax.set_ylim(bottom=0)
                                    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useOffset=False)
                                if row_index == 0:
                                    ax.set_title(variables[column])
                                if row_index == 2:
                                    ax.set_xlabel('Time [ms]')
                                if col_index == 0:
                                    text = LABELS[row_index]+'\n'+('complete' if info['complete'] == 'True' else 'incomplete')
                                    ax.set_ylabel(text, fontsize=8)
                                if row_index == 0 and col_index == 0:
                                    ax.legend(fontsize=6)
                        fig.suptitle(f'IC{number}: T0={case["T0"]:g} K, phi={case["phi"]:g}, P=1 atm | {split} | seed11 | 25 self-fed windows', fontsize=13)
                        fig.tight_layout(rect=(0, 0, 1, .95))
                        pdf.savefig(fig)
                        fig.savefig(out/'figures'/f'IC{number}_{"early" if zoom else "full"}.png', dpi=120)
                        plt.close(fig)
    history = list(csv.DictReader((root/'training.csv').open()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    updates = [int(r['update']) for r in history]
    axes[0].plot(updates, [float(r['loss']) for r in history], label='Training batch MSE')
    val = selection['validations']
    axes[0].plot([r['update'] for r in val], [r['window_MSE'] for r in val], 'o-', label='Fixed validation windows')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Future-state MSE in fifth-root coordinates')
    axes[0].legend()
    axes[1].plot(updates, [float(r['lr']) for r in history])
    axes[1].set_ylabel('Adam learning rate')
    for ax in axes:
        ax.set_xlabel('Optimizer updates')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'loss_and_lr.{ext}', dpi=150)
    plt.close(fig)
    tests = [r for r in summary if r['split'].startswith('test_')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for panel, split in enumerate(['test_interpolation', 'test_extrapolation']):
        subset = [next(r for r in tests if r['split'] == split and r['mode'] == mode) for mode in MODES]
        x = np.arange(3)
        axes[panel].bar(x-.18, [r['complete'] for r in subset], .35, label='Completed')
        axes[panel].bar(x+.18, [r['joint_pass'] for r in subset], .35, label='Joint accuracy passed')
        axes[panel].set_xticks(x, ['Rep raw\njoint5', 'Full raw\njoint9', 'Full repaired\njoint9'])
        axes[panel].set_ylim(0, max(r['total'] for r in subset)+1)
        axes[panel].set_title(split)
        axes[panel].set_ylabel('Initial conditions')
        axes[panel].legend()
    fig.tight_layout()
    fig.savefig(out/'coverage.png', dpi=150)
    plt.close(fig)
    training_table = [dict(model='remaining_species_ANN', parameters=5*64+64+64*64+64+64*6+6,
        total_updates=reconstruction['updates'], selected_update=int(Path(reconstruction['selected_checkpoint']).stem.rsplit('_',1)[-1]),
        batch_size=4096, learning_rate=.001, active_seconds=reconstruction['active_seconds'],
        wall_seconds=reconstruction['wall_seconds'], stop_reason=reconstruction['stop_reason']),
        dict(model='React_DeepONet', parameters=selection['parameters'], total_updates=selection['total_training_updates'],
        selected_update=selection['selected_update'], batch_size=20000, learning_rate=.01,
        active_seconds=selection['active_seconds'], wall_seconds=selection['wall_seconds'], stop_reason=selection['stop_reason'])]
    csv_write(out/'training_summary.csv', training_table)
    header = ['평가 집합', '방식', '완주/전체', '정확도 통과/전체', '판정 대상', '완주 조건 중 최대 NRMSE']
    table = '<table><tr>'+''.join(f'<th>{v}</th>' for v in header)+'</tr>'
    for row in summary:
        values = [row['split'], row['mode'], f"{row['complete']}/{row['total']}", f"{row['joint_pass']}/{row['total']}",
                  row['criterion'], f"{row['maximum_worst_NRMSE']:.6g}" if row['maximum_worst_NRMSE'] is not None else '전체 기간 점수 없음']
        table += '<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in values)+'</tr>'
    table += '</table>'
    links = '<ul>'+''.join(f'<li><a href="{name}">{name}</a></li>' for name in pdf_paths)+'</ul>'
    text = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>React-DeepONet 재구현 결과</title>
<style>body{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.8 system-ui;color:#172033}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #d4ddeb}th{background:#eef3f9}img{max-width:100%}a{color:#1d4ed8}</style>
<h1>문헌 구조를 맞춘 수소 React-DeepONet 재구현</h1>
<p>기준: Kumar &amp; Echekki의 <a href="https://arxiv.org/html/2304.12188v1">2023 공개 원문</a>. 저자의 공개 가중치를 실행한 결과가 아니라 명시된 프로토콜을 새로 구현·학습한 결과다. 2024 저널 최종본 전체와 동일하다고 주장하지 않는다.</p>
<p>branch 5→80→80→200, trunk 1→80→80→200, ShiftNet 5→10→10→1을 사용한다. 다섯 대표 변수는 T/H2/H/O2/H2O다. 미래 시간과 상태에 따른 log-time shift를 trunk에 넣고, 다섯 개의 내적으로 미래 상태를 직접 출력한다. 증분 모델·다중 스텝 loss·soft constraint·학습 중 물리 보정은 없다.</p>
<p>1 atm, h2o2.yaml 10종 기구에서 0.1 μs 간격으로 새 참조를 생성했다. 학습 25개, 검증 12개, 보간 테스트 16개, 외삽 테스트 15개 초기 조건은 서로 겹치지 않는다. 문헌의 정확한 분할·기구 버전·활성함수·출력 분할·복원 ANN 설정은 공개 자료에서 완전히 확정되지 않아 config.json의 가정으로 분리했다. 정규화는 학습 집합에만 맞췄다.</p>
<p>실제 예측은 20 μs씩 25번 자기 예측을 되먹여 0.5 ms까지 진행한다. 각 창의 내부도 0.1 μs 간격으로 질의하지만 그 내부값은 다음 창의 시작값으로 사용하지 않는다. Cantera는 초기 상태와 평가 참조만 제공한다. 대표 변수 raw와 전체 상태 raw를 구분하는 이유는 별도 복원 ANN의 오류를 DeepONet 오류와 분리하기 위해서다.</p>
<p>full repaired는 기존 프로젝트의 농도 clipping·비음수 원소 투영·상압 엔탈피 온도 보정을 포함한 별도 시스템이다. 학습에 제약 loss를 넣는 soft constraint가 아니다.</p>
<p>NRMSE = RMSE(Cantera,NN)/RMS(Cantera), 백분율이 아닌 비율이다. joint5는 대표 5변수 각각 NRMSE≤0.1, joint9는 T+반응종 8개 각각 NRMSE≤0.1이다. N2와 AR은 별도로 기록한다. 불완주에는 전체 0.5 ms 점수를 주지 않는다. 초기 0.1 ms 점수도 CSV에 있다. train 행의 1065 K/φ=1은 원문 그림과 같은 학습 중 본 조건을 보여주는 진단이며 미지 테스트가 아니다.</p>'''+table+f'''
<p>DeepONet은 총 {selection['total_training_updates']} / 200000 update를 실행했다. 검증으로 선택한 것은 {selection['selected_update']} update 체크포인트다. 종료 이유: {selection['stop_reason']}. 파라미터 {selection['parameters']}개. 시간상한 종료를 평탄 도달로 부르지 않는다. 동일 검증 조건에서 선택하고 그 뒤에 테스트를 평가했다.</p>
<p><a href="training_summary.csv">모델별 학습시간·update·LR·종료 이유</a> · <a href="../evaluations/NRMSE.csv">변수별 NRMSE</a> · <a href="../evaluations/rollouts.csv">조건별 완주·정확도</a> · <a href="../selection_frozen.json">검증 선택 기록</a> · <a href="../evaluations/reconstruction_given_truth.csv">참 대표 변수를 입력한 복원 ANN 단독 진단</a></p>
<img src="coverage.png" alt="완주와 정확도 통과 수"><img src="loss_and_lr.png" alt="학습 loss와 learning rate">
<p>완주와 정확도 통과는 다르다. 각 방식의 완주 집합이 다르면 중간값만으로 우열을 단정할 수 없다. 이 결과의 범위는 고정 1 atm·0.5 ms이며 기존 1–5 atm·10 ms 실험으로 일반화하지 않는다.</p>
<p>학습 전 게시한 ANN·문헌 조사 커밋: <a href="https://github.com/rom-cfd/personal/commit/94a8eff4d71d9ab469f541aa621073eb495b8e04">94a8eff</a>.</p>
<h2>모든 평가 조건의 그래프</h2>'''+links+'</html>'
    (out/'REPORT.html').write_text(text)
    write(out/'complete.json', dict(status='passed', figures_per_pdf={name: sum(r['mode'] == 'raw_full' and r['split'] == name.split('_ignition')[0].split('_profiles')[0] for r in rows) for name in pdf_paths},
        output_sha256={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}))
