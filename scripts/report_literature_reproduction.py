"""Read frozen author-checkpoint runs; write a new audit and scientific figures."""

import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/literature-report-mpl')
import argparse
import csv
import hashlib
import html
import json
import platform
from pathlib import Path

import numpy as np


MODES = {
    'author_original': 'Released ANN (raw)',
    'published_BCT_projection_common_HP': 'Published BCT projection + HP',
    'current_projection_common_HP': 'Current projection + HP',
}
LABELS_KO = {
    'author_original': '공개 ANN 원본 · 보정 없음',
    'published_BCT_projection_common_HP': '논문 BCT 원소 보정식 + 공통 엔탈피 보정',
    'current_projection_common_HP': '현재 프로젝트 원소 보정 + 공통 엔탈피 보정',
}
DATA_HASHES = {
    'results/long10ms_data/trajectories.npz': '8d41f19f2145e12f9e89ee3eaa88fb0b1b93a84dcaaee65dcaa2e8b264077563',
    'results/long10ms_data/training_pairs.npz': '44db17b16e167ea389631dacb8015528f76cbd5c2072e9fb919701e54d50fa3e',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def csv_out(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def nrmse(a, b):
    den = np.linalg.norm(a, axis=0)
    return np.divide(np.linalg.norm(b-a, axis=0), den,
                     out=np.full_like(den, np.nan), where=den > 0)


def ignition(trace, dt, threshold):
    above = np.flatnonzero(trace[:, 0] >= threshold)
    if not len(above):
        return None
    j = int(above[0])
    if j == 0:
        return 0.0
    fraction = (threshold-trace[j-1, 0])/(trace[j, 0]-trace[j-1, 0])
    return float((j-1+fraction)*dt*1e6)


def table(headers, rows):
    cells = lambda tag, row: ''.join(f'<{tag}>{html.escape(str(v))}</{tag}>' for v in row)
    return '<table><thead><tr>'+cells('th', headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+cells('td', row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new report directory; do not overwrite evidence')

    runs = []
    for root in args.roots:
        complete = json.loads((root/'complete.json').read_text())
        verification = json.loads((root/'verification.json').read_text())
        assert complete['status'] == verification['status'] == 'passed'
        for name, expected in complete['output_sha256'].items():
            assert sha(root/name) == expected, str(root/name)
        runs.append((root, json.loads((root/'config.json').read_text()),
                     list(csv.DictReader((root/'summary.csv').open())),
                     np.load(root/'traces/all_profiles.npz', allow_pickle=False)))
    original_hashes = {name: sha(name) for name in DATA_HASHES}
    assert original_hashes == DATA_HASHES, 'Original research data changed'

    out = args.output
    out.mkdir(parents=True)
    (out/'figures').mkdir()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .15})
    full_pdf = PdfPages(out/'all_profiles.pdf')
    early_pdf = PdfPages(out/'ignition_zoom.pdf')
    metrics = []
    grouped = []
    case_index = []
    for root, config, summary, archive in runs:
        species = config['species']
        active = [0]+[j+2 for j, s in enumerate(species) if s != 'N2']
        dt = config['dt']
        horizon_ms = config['horizon_s']*1000
        for row in summary:
            model, temp, mode = row['model'], int(row['T0']), row['mode']
            prefix = f'{model}_T{temp}'
            reference = archive[prefix+'_reference']
            predicted = archive[prefix+'_'+mode]
            complete = row['complete'] == 'True'
            early_points = int(round(.0001/dt))+1
            early_complete = len(predicted) >= early_points
            early = nrmse(reference[:early_points], predicted[:early_points]) if early_complete else None
            full = nrmse(reference, predicted) if complete else None
            worst = lambda values: float(np.max(values[active])) if values is not None else None
            t_ref = ignition(reference, dt, temp+400)
            t_pred = ignition(predicted, dt, temp+400)
            metrics.append(dict(run=root.name, horizon_ms=horizon_ms, model=model, T0=temp, mode=mode,
                complete=complete, joint_active7_pass=bool(complete and worst(full) <= .1),
                worst7_full=worst(full), early_0p1ms_complete=early_complete,
                worst7_early_0p1ms=worst(early),
                reference_Tplus400_us=t_ref, predicted_Tplus400_us=t_pred,
                ignition_error_us=(t_pred-t_ref if t_ref is not None and t_pred is not None else None)))
        for model in config['models']:
            for mode in MODES:
                subset = [r for r in metrics if r['run'] == root.name and r['model'] == model and r['mode'] == mode]
                full_scores = [r['worst7_full'] for r in subset if r['complete']]
                early_scores = [r['worst7_early_0p1ms'] for r in subset if r['early_0p1ms_complete']]
                grouped.append(dict(run=root.name, horizon_ms=horizon_ms, model=model, mode=mode,
                    complete=sum(r['complete'] for r in subset), total=len(subset),
                    joint_active7_pass=sum(r['joint_active7_pass'] for r in subset),
                    maximum_worst7_full=max(full_scores) if full_scores else None,
                    maximum_worst7_early_0p1ms=max(early_scores) if early_scores else None))
            for temp in config['temperatures']:
                prefix = f'{model}_T{temp}'
                reference = archive[prefix+'_reference']
                for zoom in [False, True]:
                    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
                    handles = None
                    for ax, j in zip(axes.flat, [0]+list(range(2, 9))):
                        ax.plot(np.arange(len(reference))*dt*1e3, reference[:, j], color='black', lw=2, label='Cantera')
                        for mode, color in zip(MODES, ['#2563eb', '#d97706', '#0f766e']):
                            pred = archive[prefix+'_'+mode]
                            ax.plot(np.arange(len(pred))*dt*1e3, pred[:, j], color=color, lw=1.4,
                                    ls='--', label=MODES[mode])
                        ax.set_title('T' if j == 0 else species[j-2])
                        ax.set_xlabel('Time [ms]')
                        ax.set_ylabel('Temperature [K]' if j == 0 else 'Mass fraction [-]')
                        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
                        if j > 1:
                            ax.set_ylim(bottom=0)
                            if species[j-2] == 'N2':
                                ax.set_ylim(0, 1)
                            else:
                                ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useOffset=False)
                        ax.set_xlim(0, .1 if zoom else horizon_ms)
                        handles = ax.get_legend_handles_labels()
                    fig.suptitle(f'{model} | T0={temp} K, P=1 atm, phi=1 | horizon={horizon_ms:g} ms | released weights, no training', fontsize=12)
                    fig.legend(*handles, loc='upper center', bbox_to_anchor=(.5, .95), ncol=4, fontsize=9)
                    fig.tight_layout(rect=(0, 0, 1, .89))
                    name = f'{root.name}_{prefix}'+('_early' if zoom else '')+'.png'
                    fig.savefig(out/'figures'/name, dpi=150)
                    (early_pdf if zoom else full_pdf).savefig(fig)
                    plt.close(fig)
                    if not zoom:
                        case_index.append(dict(horizon_ms=horizon_ms, model=model, T0=temp, figure='figures/'+name))
        archive.close()
    full_pdf.close()
    early_pdf.close()
    csv_out(out/'case_diagnostics.csv', metrics)
    csv_out(out/'aggregate.csv', grouped)
    csv_out(out/'figure_index.csv', case_index)

    # Inspect the existing training pairs without fitting or changing a model.
    pairs = np.load('results/long10ms_data/training_pairs.npz', allow_pickle=False)
    a, b = pairs['x'][:, :10], pairs['y']
    normalization = []
    for gamma in [.1, .2]:
        encode = lambda q: np.c_[q[:, 0]/1000, q[:, 1:]**gamma/gamma]
        u, v = encode(a), encode(b)
        state_sd, increment_sd = np.std(np.r_[u, v], axis=0), np.std(v-u, axis=0)
        for j, name in enumerate(['T', 'H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'N2']):
            relative = float(increment_sd[j]/state_sd[j])
            normalization.append(dict(gamma=gamma, variable=name, state_std=float(state_sd[j]),
                increment_std=float(increment_sd[j]), increment_std_over_state_std=relative,
                independent_increment_MSE_weight_relative_to_current=(1/relative**2 if relative > 0 else None)))
    pairs.close()
    csv_out(out/'normalization_diagnostic.csv', normalization)

    summary_table = table(['기간', '공개 모델', '예측 방식', '완주', '정확도 통과', '최대 NRMSE · 전체', '최대 NRMSE · 초기 0.1 ms'], [
        [f"{r['horizon_ms']:g} ms", r['model'].split('_')[0], LABELS_KO[r['mode']],
         f"{r['complete']}/{r['total']}", f"{r['joint_active7_pass']}/{r['total']}",
         f"{r['maximum_worst7_full']:.6g}" if r['maximum_worst7_full'] is not None else '미완주',
         f"{r['maximum_worst7_early_0p1ms']:.6g}" if r['maximum_worst7_early_0p1ms'] is not None else '미완주'] for r in grouped])
    comparisons = table(['항목', '기존 프로젝트', '공개 ANN 예제 / 논문'], [
        ['반응계', 'H2O2 계열 9종·28반응, HO2/H2O2 포함', '공개 예제 ESH2 7종·16반응, HO2/H2O2 없음'],
        ['평가 기간', '2 μs × 5000 = 10 ms', '공개 예제 1 μs × 500 = 0.5 ms'],
        ['모델', 'ANN 11→64→64→64→10, tanh, 9738개 파라미터', '9→1600→800→400→9, myGELU, 1620809개 파라미터'],
        ['학습 데이터', '선별한 8815개 상태쌍', '체크포인트 설정상 학습 5792898 / 4752020개'],
        ['출력 정규화', '증분을 상태 표준편차로 나눔', '변환 좌표의 유한차분 변화율에 별도 평균·표준편차 적용'],
        ['목적 함수·학습량', 'MSE, 작은 모델 CPU 단기/시간상한 장기 실험', 'L1, 설정상 5000 epochs·단계별 배치/LR. 5000 updates와 다름'],
        ['원소 보정', '가중 Y 좌표 투영 + 비음수 제약, 농도 하한 1e-8', '2025 논문은 BCT 오차 최소화에 따른 농도 가중 투영'],
    ])
    source_links = [
        ('2022 ANN 논문', 'https://doi.org/10.1016/j.combustflame.2022.112319'),
        ('고정된 저자 ANN 코드·가중치', 'https://github.com/tianhanz/DNN-Models-for-Chemical-Kinetics/tree/bc07e5a210ba357b8a5350b460a751a68a002174'),
        ('2025 ANN-hard 논문', 'https://doi.org/10.1016/j.combustflame.2025.114105'),
        ('React-DeepONet 공개 2023 원문', 'https://arxiv.org/html/2304.12188v1'),
        ('React-DeepONet 2024 저널 논문', 'https://doi.org/10.1016/j.fuel.2024.131212'),
        ('Extended DeepONet 논문', 'https://doi.org/10.1016/j.cma.2023.116674'),
        ('Extended DeepONet 공개 코드', 'https://github.com/somdattagoswami/Approximating-StiffKinetics'),
    ]
    source_list = '<ul>'+''.join(f'<li><a href="{url}">{label}</a></li>' for label, url in source_links)+'</ul>'
    figures = ''.join(f'<li>{r["horizon_ms"]:g} ms · {r["model"]} · T0={r["T0"]} K: <a href="{r["figure"]}">그림</a></li>' for r in case_index)
    report = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>수소 0D 문헌 비교와 공개 모델 재현</title>
<style>body{max-width:1150px;margin:40px auto;padding:0 22px;font:16px/1.8 system-ui;color:#172033}h1{font-size:28px}h2{margin-top:40px;font-size:22px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d6deea;padding:9px;vertical-align:top}th{background:#edf2f9}a{color:#1d4ed8}.box{padding:18px;background:#edf7f5;border-left:4px solid #0f766e}code{font-family:monospace}img{max-width:100%}@media print{body{margin:0}h2{break-after:avoid}tr{break-inside:avoid}}</style>
<h1>수소 0D: 기존 실험과 성공 문헌의 차이, 공개 ANN 재현</h1>
<p>2026-11-09 · 원본 데이터와 기존 결과는 보존했다. 새 학습 없이 저자가 공개한 두 ANN의 가중치를 고정해 실행했다.</p>
<div class="box"><b>확인된 결론:</b> 같은 환경에서 공개 ANN의 좋은 예측과 물리 보정의 효과를 재현했다. 기존 작은 ANN/DeepONet의 실패를 “0D에서는 NN이나 물리 보정이 안 된다”는 결론으로 일반화할 근거는 없다. 다만 어떤 학습 차이가 기존 실패를 주도했는지는 통제된 재학습으로 분리해야 한다.</div>
<h2>1. 실제로 재현한 범위</h2>
<p>Zhang 등(2022)과 연결된 공개 저장소의 코드·기구·두 체크포인트를 고정했다. 1100–1600 K의 6개 초기 온도, 1 atm, 당량비 1에서 1 μs씩 자기 예측을 되먹였다. 첫 0.5 ms는 공개 데모 조건이며, 같은 가중치의 10 ms는 이번에 추가한 기간 진단이다. 재학습, Cantera 정답의 중간 주입, 화학 ODE로의 대체는 없다.</p>
<p>정확도 통과는 온도와 반응종 6개 각각의 NRMSE가 모두 0.1 이하인 경우다. N2는 별도다. NRMSE = RMSE(Cantera, NN)/RMS(Cantera)이며 백분율이 아니다. 기존 9종 문제의 joint9 통과 수나 18개 미지 테스트 성능과 합산하지 않는다. 공개 데모 6개는 새로 분리한 미지 테스트가 아니다.</p>
<p>보정 진단은 동일 ANN에 두 원소 투영을 각각 적용하고, 두 방식 모두 현재 엔탈피·상압으로 온도를 복원했다. “현재 보정”은 기존 project_species와 역변환 가드를 적용한 것이다. 앙상블 후보 필터는 이번 단일 모델에 없으며, 2025 ANN-hard의 원래 9종 학습 모델을 재현했다고 주장하지 않는다.</p>
<h2>2. 모든 초기 조건의 결과</h2>'''+summary_table+'''
<p>최대 NRMSE는 해당 행의 완주한 조건들과 온도·6반응종 전체에서 가장 큰 값이다. 불완주에는 전체 기간 점수를 부여하지 않았다. 초기 0.1 ms와 점화시각(T가 초기보다 400 K 상승하는 최초 시각)도 별도 CSV로 보존했다. 통과 여부는 그림의 눈대중이 아니라 같은 시각의 값들로 계산했다.</p>
<p><a href="all_profiles.pdf">전체 곡선 PDF</a> · <a href="ignition_zoom.pdf">초기 0.1 ms 확대 PDF</a> · <a href="aggregate.csv">집계 CSV</a> · <a href="case_diagnostics.csv">조건별 정확도·점화 CSV</a> · <a href="normalization_diagnostic.csv">기존 학습 정규화 진단</a></p>
<h2>3. 기존 구현과 무엇이 다른가</h2>'''+comparisons+'''
<p>증분 대신 변화율을 출력한다는 표현만으로 차이를 설명하면 부정확하다. 고정 시간 간격에서는 둘은 선형 변환 관계다. 실제 차이는 <b>출력 변화량의 별도 표준화와 목적 함수</b>다. 기존 γ=0.2 학습쌍에서 증분 표준편차/상태 표준편차는 온도 약 0.0544, HO2 약 0.187이다. 출력 증분을 각자 표준화하면 현재 MSE에 비해 온도 항은 약 338배, HO2는 약 28.6배의 가중치가 된다. 이는 loss의 변수 간 균형이 같지 않다는 진단이며, 이 차이가 주원인이라는 인과 증명은 아니다.</p>
<p>현재 보정은 원소 수지와 엔탈피를 맞추지만 반응 경로, 반응 속도, 점화시각을 결정하지 않는다. 서로 다른 많은 조성 상태가 같은 원소량과 엔탈피를 가질 수 있다. 틀린 궤적을 보존 법칙 위로 옮겨도 Cantera 궤적이 되지는 않는다. 반면 좋은 공개 ANN에 현재 보정을 적용했을 때는 정확한 궤적을 유지할 수 있었다. 따라서 기존 학습된 상태 전이, 표본 범위, 학습량을 먼저 점검할 근거가 생겼다.</p>
<h2>4. 물리 보정 논문과의 차이</h2>
<p>Wang 등(2025)의 ANN-hard는 학습 loss에 제약을 넣는 방법이 아니라, 학습된 ANN의 원소 오차를 예측 뒤에 보정하는 방법이다. 논문 식(6)–(9)은 BCT 좌표의 보정 거리를 최소화한다. γ=0.1이면 Y 좌표의 가중 분산은 예측 농도 Y의 1.8승에 해당한다. 현재 코드의 max(예측 농도, 현재 농도, 1e-8)의 제곱 및 비음수 active-set과는 다르다. 그렇다고 현재 코드가 단순 무가중 투영이라는 뜻도 아니다.</p>
<p>해당 논문의 수소 사례는 9종·19반응, 110만 표본, 1600/800/400 GELU 모델이다. 이번에 확보한 공개 체크포인트는 7종·16반응이다. 논문 식을 이 이전 공개 모델에 적용한 것은 <b>보정식 진단</b>이며 논문 전체 재현이 아니다. 논문이 선택한 soft-constraint λ=100도 모든 문제의 최적값을 의미하지 않는다.</p>
<h2>5. DeepONet이 성공한 문헌과의 차이</h2>
<p>React-DeepONet의 공개 2023 원문은 현재 상태를 branch, 미래 시간을 trunk에 넣고 ShiftNet으로 시간 좌표를 조정한다. 이동하는 0–20 μs 창을 촘촘히 학습하며, 대표 변수 T/H2/H/O2/H2O를 진행하고 나머지는 별도 ANN으로 복원한다. 수소 평가는 20 μs×25회=0.5 ms다. 구조는 약 4.7만 파라미터, 학습은 배치 2만·20만 update로 보고된다. 이 상세 설정은 2024 저널 최종본과 동일하다고 검증한 것이 아니다.</p>
<p>기존 P-only/P-T/P-T-phi는 미래 시간을 trunk에 넣지 않은 조건부 단일 전이 모델이었다. 최근 interval 모델은 미래 간격을 넣었지만 작은 구조·다른 표본·ShiftNet 없음·2 μs×5000 평가를 사용했다. 따라서 지금까지의 실험은 논문 방식의 정밀 재현이 아니다. 또한 논문의 정규화 곡선 MAE와 현재의 모든 반응종 개별 NRMSE 기준은 다르다.</p>
<p>Goswami 등(2024)의 extended DeepONet 저장소도 확인했다. 공개된 것은 syngas 예제의 일부 MAT 데이터와 AE/DeepONet 소스이며, 전체 데이터·사전학습 가중치는 없다. 제공된 예제는 참 초기 상태에서 네 미래 시점을 질의한다. 5000번 자기 예측을 되먹이는 기존 수소 평가와 구분해야 한다. 코드 기본 학습량·테스트 기반 AE 선택 등도 따로 감사했다. 명시적 라이선스를 찾지 못해 이 소스·데이터는 저장소에 재배포하지 않았다.</p>
<h2>6. 재현을 먼저 하는 연구 순서</h2>
<ol><li><b>공개 가중치로 성공 기준 확보:</b> 이번에 완료했다. 원본 코드와 보정 없는 결과를 기준점으로 보존한다.</li>
<li><b>기간만 늘리기:</b> 동일 기구·가중치·시간 간격을 유지한 10 ms 진단을 별도 수행했다. 원래 논문의 평가 결과로 표시하지 않는다.</li>
<li><b>학습 재현:</b> 출력 정규화·loss·모델 규모·표본 생성 규칙을 함께 감사하고, 저자 데이터가 없으면 자체 생성 데이터라는 사실을 명시한다. 기존 8815개 표본과 150초를 그대로 적용한 결과를 논문 재현이라고 부르지 않는다.</li>
<li><b>DeepONet 프로토콜 재구성:</b> 미래 시간·ShiftNet·이동 시간창·대표 변수 복원·논문 평가 기간을 먼저 맞춘다. 공개 자료에 없는 활성함수, 기구 버전, 분할과 ShiftNet 세부 식은 가정으로 표시한다.</li>
<li><b>그 다음 연구 아이디어:</b> 재현 기준이 확보된 동일 조건에서 하나씩 바꾸고, 이후 9종·넓은 P/φ 범위·10 ms로 확장한다. 그림, 반응종별 NRMSE, 점화, 완주를 함께 평가한다.</li></ol>
<p>새 장기 학습은 시작하지 않았다. 실행한 것은 두 공개 체크포인트의 추론과 보정 진단이다. 계획의 기계 판독 사양은 <a href="../../../configs/literature_baseline_plan.json">configs/literature_baseline_plan.json</a>에 있다.</p>
<h2>7. 검증과 제한</h2>
<p>원본 코드의 501개 상태 전체와 직접 호출한 결과가 두 모델 모두 배열 단위로 일치한다. 저장된 곡선에서 모든 NRMSE를 다시 계산했고, 독립적인 연속 Cantera 적분과 원본 데모의 재시작 적분도 비교했다. 원본 소스·가중치 해시와 기존 두 데이터 해시는 유지했다. 런타임은 현재 Cantera 3.2/PyTorch 2.6 CPU이며, 저자의 과거 환경까지 완전히 복원한 것은 아니다.</p>
<p>첫 시도 run_001은 초기 상태 생성의 부동소수점 차이로 엄격한 원본 일치 검사에 실패했다. 원본 코드의 초기화 방식을 그대로 따라 run_002에서 일치를 검증했다. 실패 기록은 삭제하지 않았으며, 이 도구 초기화 문제를 NN 예측 실패로 집계하지 않는다.</p>
<h2>1차 출처</h2>'''+source_list+'''<h2>모든 조건 그림</h2><ul>'''+figures+'''</ul></html>'''
    (out/'REPORT.html').write_text(report)
    import cantera as ct
    import torch
    audit = dict(status='passed', roots=[str(r) for r in args.roots],
                 original_data_sha256=original_hashes, trained_new_models=0,
                 versions=dict(python=platform.python_version(), numpy=np.__version__, cantera=ct.__version__, torch=torch.__version__),
                 new_source_sha256={str(p): sha(p) for p in [Path(__file__), Path('kinetics/literature_reproduction.py'), Path('kinetics/physics.py')]},
                 frozen_run_complete_sha256={str(r): sha(r/'complete.json') for r in args.roots},
                 scope='Released 7-species checkpoint inference and separately labelled correction/horizon diagnostics; no full training or DeepONet reproduction claim')
    dump(out/'verification.json', audit)
    dump(out/'complete.json', dict(status='passed', output_sha256={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(grouped, ensure_ascii=False))


if __name__ == '__main__':
    main()
