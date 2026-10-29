"""Render the completed gamma/tanh ablation; never modify original results."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/kinetics-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def table(frame):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return '—' if not np.isfinite(v) else f'{v:.5g}'
        return str(v)
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
        '|' + '|'.join(['---']*len(frame.columns)) + '|'] +
        ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/preprocess_tanh10'))
    parser.add_argument('--report', type=Path, default=Path('PREPROCESS_REPORT.md'))
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    cfg = json.loads((root / 'config.json').read_text())
    frozen = json.loads((root / 'all_selections_frozen.json').read_text())
    # This common control was fixed from validation scores before any rollout.
    control = json.loads((root / 'global_single_frozen.json').read_text())
    summary = pd.read_csv(root / 'summary.csv')
    detail = pd.read_csv(root / 'trajectories.csv')
    priori = pd.read_csv(root / 'one_step.csv')
    checks = pd.read_csv(root / 'reload_checks.csv')
    verification = json.loads((root / 'verification.json').read_text())
    assert verification['pytest_failures'] == verification['pytest_errors'] == verification['pytest_skipped'] == 0
    assert len(summary) == len(priori) == len(checks) == 12
    assert checks[['reload_match', 'first_step_match', 'singleton_seed_fixed']].all().all()
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    with np.load(cfg['source']) as data:
        states, split, names = data['states'], data['split'], data['species']
    ids = np.flatnonzero(split == 'test')
    colors = dict(single_best='tab:red', mean10='tab:blue', medoid10='tab:orange')
    for gamma in cfg['gammas']:
        label = f'g{gamma:g}'
        traces = {}
        for method in colors:
            with np.load(root / label / 'traces' / f'{method}_test.npz') as archive:
                traces[method] = {int(i): archive[f'trajectory_{int(i)}'] for i in ids}
        def draw(ax, i, column):
            truth = states[i]
            ax.plot(np.arange(1, len(truth))*cfg['dt']*1e3, truth[1:, column], 'k--', lw=1.6, label='Cantera')
            for name, color in colors.items():
                t = traces[name][i]
                ax.plot(np.arange(1, len(t))*cfg['dt']*1e3, t[1:, column], color=color, lw=1, label=name)
            ax.set_xscale('log')
            ax.set_xlim(cfg['dt']*1e3, cfg['horizon']*1e3)
            ax.set_xlabel('Time (ms)')
            ax.grid(alpha=.2)
        fig, axes = plt.subplots(3, 6, figsize=(22, 11))
        for i, ax in zip(ids, axes.flat):
            draw(ax, int(i), 0)
            ax.set_title(f'IC {i}, T0={states[i, 0, 0]:.0f} K')
            ax.set_ylabel('T (K)')
        axes.flat[-1].legend(fontsize=7)
        fig.suptitle(f'Gamma {gamma:g}: 10 ms self-fed held-out temperature profiles')
        fig.tight_layout()
        fig.savefig(figures / f'{label}_temperature.png', dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(3, 3, figsize=(14, 11))
        for j, ax in enumerate(axes.flat):
            draw(ax, 44, j+1)
            ax.set_yscale('symlog', linthresh=1e-15)
            ax.set_title(str(names[j]))
            ax.set_ylabel('Mass fraction')
        axes.flat[-1].legend(fontsize=8)
        fig.suptitle(f'Gamma {gamma:g}: nine species on previously difficult IC 44')
        fig.tight_layout()
        fig.savefig(figures / f'{label}_species_ic44.png', dpi=150)
        plt.close(fig)
    sections = ['# Box–Cox γ=0.2 / 0.5: 10-member tanh ANN 재훈련', '',
        '최신 `kisung` 커밋 `b5ecd33`에서 만든 `pre-process` 브랜치의 실험이다. '
        '각 γ에서 10개를 새로 훈련했고, 전체 20개 중 validation a priori 최상위 단일 모델 하나를 공통 대조군으로 고정했다. '
        '각 γ 내부의 최상위 singleton도 보조 대조군으로 함께 평가했다. '
        '기존 데이터·가중치·결과는 덮어쓰지 않았다.', '',
        '## 조건과 선택 규칙', '',
        '- 구조: **64–64–64 tanh**, 모델당 9,738개 파라미터. GELU/1600–800–400 paper 모델은 사용하지 않았다.',
        '- 기존 9종/28반응 수소계, train/validation/test IC 54/18/18개, 10 ms, dt=2 μs. '
        'Train-only 평형 downsampling으로 고정된 **8,815쌍**을 그대로 재사용했다. Augmentation은 없다.',
        '- γ별 seed 11,22,…,110의 10개, 모델당 10,000 Adam update, batch512, '
        'lr .001→.0003→.0001, 멤버별 norm clip5. 총 20개/200,000 member-updates. '
        '5개씩 계산만 묶고 가중치·Adam state·minibatch RNG는 독립적이다.',
        '- γ 사이에는 같은 seed 목록을 매칭하여 초기화·minibatch 차이를 통제했다. '
        'γ만 바꾼 두 집합이지 서로 독립적으로 재추출한 ensemble 반복은 아니다.',
        '- 입력은 T(t), log P, Box–Cox(Y(t)). 다음 상태의 정규화된 transformed residual을 학습한다. '
        'Box–Cox는 Y^γ/γ이며 표준화하면 (Y^γ−1)/γ와 affine-equivalent다. '
        '전처리 평균/표준편차는 train 입력·target만으로 적합했다.',
        '- Wang 보정의 역미분 가중치도 **Y^(1−γ)**로 일치시켰다. '
        'ANN T 출력은 보정 추론에서 버리고 현재 엔탈피와 다음 Y, 고정 P로 T를 복원한다. '
        'Decoder의 기존 [0,1] domain guard를 유지하고, 보정 후 음수/불능 후보는 거부한다.',
        '- Mean10: 멤버별 보정 후 유효 후보 평균, 원소/엔탈피 복원. Medoid10: train-fitted scaled 거리로 실제 보정 후보 하나 선택. '
        'ODE fallback·hysteresis는 없다.',
        '- 단일 모델: γ별 10개 모두를 **validation 18×5,000=90,000쌍 전부**에서 평가하여 기존 물리 단위 selection_score 최소 모델을 선택. '
        '동률은 seed 오름차순. 두 γ의 선택 기록을 모두 저장한 뒤에만 rollout/test 평가를 시작했다.', '',
        '`selection_score = mean(log1p(min(|Ypred−Ytrue|/max(Ytrue,1e−8),1e12))) '
        '+ 10×major_Y_RMSE + 10×mass_sum_MAE + 100×invalid_row_fraction`', '',
        'Major 기준은 Y≥1e−3이다. 이 score에는 T RMSE가 직접 들어가지 않는다. '
        '따라서 “가장 좋은 모델”은 이 명시적 a priori score상 최상위라는 뜻이다. '
        'Validation은 평형 후기 구간도 원래 시간 비중으로 포함하므로 그 구간이 순위에 큰 영향을 줄 수 있다. '
        'A posteriori 결과나 test 오차로 seed·γ·aggregation을 재선택하지 않았다. '
        '기존과 같은 이미 관측된 test split의 ablation이며 새로운 외부 일반화 검증은 아니다.', '',
        '## 고정한 단일 대조군', '']
    selection_rows = []
    for gamma in cfg['gammas']:
        chosen = frozen[f'g{gamma:g}']
        rank = pd.read_csv(root / f'g{gamma:g}' / 'validation_member_ranking.csv')
        best = rank.iloc[0]
        selection_rows.append(dict(gamma=gamma, seed=chosen['selected_seed'], validation_score=chosen['validation_score'],
            Y_RMSE=best.rmse, T_RMSE_K=best.temperature_rmse, invalid_fraction=best.invalid_row_fraction))
    sections += [table(pd.DataFrame(selection_rows)), '',
        f'**공통 단일 대조군: γ={control["gamma"]:g}, seed {control["selected_seed"]}**, '
        f'validation score={control["validation_score"]:.8g}. 두 γ의 최상위 중 같은 score로 선택했으므로 전체 20개 중 최상위다. '
        '각 γ 전체 순위는 각 하위 폴더의 `validation_member_ranking.csv`에 있다.', '',
        '## A priori: 동일한 전체 90,000쌍 비교', '',
        '모든 방법에 동일한 truth 입력을 주는 one-step 오차다. 아래 test와 validation 모두 full-pair 평가이며 '
        '희소 probe가 아니다. Self-fed a posteriori와 구분한다.', '']
    for partition in ('val', 'test'):
        sections += [f'### {partition}', '', table(priori[priori.split == partition][['gamma', 'method', 'samples', 'selection_score', 'rmse', 'major_rmse', 'minor_rmse', 'temperature_rmse', 'invalid_row_fraction']]), '']
    sections += ['## A posteriori: 10 ms 전체 self-fed 예측', '',
        '각 IC의 초기 상태만 입력하고 이후 5,000 step을 자기 예측으로 전진한다. '
        '중간 Cantera 상태 주입이나 rollout downsampling은 없다. '
        '점화는 초기 T 대비 400 K 상승 시점이며 reference는 기존 0.2 μs 정밀 Cantera τ를 재사용했다. '
        '점화오차는 양쪽 모두 점화한 쌍의 조건부 중앙값으로, 쌍 수를 함께 표시한다. '
        '미완주 full-profile 오차는 NaN으로 제외하고 coverage/실패를 따로 보고한다.', '']
    for partition in ('val', 'test'):
        frame = summary[summary.split == partition].copy()
        frame['completion_%'] = frame.completion_fraction*100
        frame['physical_%'] = frame.physical_fraction*100
        frame['ignition_accuracy_%'] = frame.ignition_classification*100
        frame['ignition_error_%'] = frame.median_ignition_error_refined*100
        sections += [f'### {partition}', '', table(frame[['gamma', 'method', 'completed', 'completion_%', 'physical_%',
            'ignition_accuracy_%', 'ignition_error_%', 'ignition_pairs', 'temperature_profile_mean',
            'temperature_profile_pooled', 'species_profile_mean']]), '']
    sections += ['### 공통 최상위 단일 대조군 대비', '']
    delta = []
    base = summary[(summary.gamma == control['gamma']) & (summary.method == 'single_best') & (summary.split == 'test')].iloc[0]
    for gamma in cfg['gammas']:
        group = summary[(summary.gamma == gamma) & (summary.split == 'test')].set_index('method')
        for method in ('mean10', 'medoid10'):
            r = group.loc[method]
            delta.append(dict(gamma=gamma, method=method, completion_delta_pp=100*(r.completion_fraction-base.completion_fraction),
                ignition_accuracy_delta_pp=100*(r.ignition_classification-base.ignition_classification),
                T_RMSE_delta_K=r.temperature_profile_mean-base.temperature_profile_mean,
                Y_RMSE_delta=r.species_profile_mean-base.species_profile_mean))
    sections += [table(pd.DataFrame(delta)), '',
        '차이는 ensemble−single이다. RMSE 차이가 음수면 완주 궤적 평균 오차가 더 작다. '
        '완주 집합이 다르면 이 차이를 전체 성능 개선으로 단정할 수 없다.', '',
        '### 기존 γ=0.1의 10-member 결과: 역사적 참고', '']
    historical_root = Path('results/wang_hard10ms')
    historical_manifest = json.loads((historical_root / 'manifest.json').read_text())
    assert historical_manifest['source_sha256'] == manifest['input_hashes']['source']
    historical = pd.read_csv(historical_root / 'summary.csv')
    historical = historical[(historical.split == 'test') & historical.method.isin(['mean10', 'medoid10'])].copy()
    old_single_root = Path('results/single_apriori_wang')
    old_single_manifest = json.loads((old_single_root / 'manifest.json').read_text())
    assert old_single_manifest['source_sha256'] == manifest['input_hashes']['source']
    old_single = pd.read_csv(old_single_root / 'summary.csv')
    old_single = old_single[old_single.split == 'test'].copy()
    old_single['method'] = 'historical_single_best_of_100_seed11'
    historical = pd.concat([historical, old_single], ignore_index=True)
    historical['gamma'] = .1
    sections += [table(historical[['gamma', 'method', 'completed', 'ignition_classification',
        'median_ignition_error_refined', 'temperature_profile_mean', 'species_profile_mean']]), '',
        '기존 Wang 실행의 같은 데이터 hash·64/64/64 tanh·첫 10 seed·모델당 10,000-update 결과를 그대로 재사용했다. '
        '이전 a priori 최상위 γ=.1/seed11 단일 모델도 기존 결과로 함께 표시했다. '
        '이 모델은 과거 100개 중 고른 모델로 이번 10개당 선택과 후보 풀 크기가 다르다. '
        'γ=.1은 이번에 재훈련하지 않았고, 새 20개의 singleton 공통 대조군 선택 후보에도 넣지 않았다. '
        '위 역사적 표의 ignition 지표는 비율(0–1)이다. '
        '새 γ는 5개씩 grouped training했고 기존 γ=.1은 8개씩 계산했으므로 수치 연산 순서의 작은 차이가 있을 수 있다.', '',
        '### Test 실패·점화 불일치', '']
    bad = detail[(detail.split == 'test') & (~detail.complete | ~detail.ignition_classification_correct)]
    sections += [table(bad[['gamma', 'method', 'trajectory', 'complete', 'coverage', 'completed_steps',
        'reference_ignites', 'prediction_ignites', 'ignition_status', 'full_temperature_rmse', 'prefix_temperature_rmse']]) if len(bad) else '실패나 점화 분류 불일치 없음.', '',
        '## 물리 진단과 실행 검증', '',
        '물리 허용 완주는 기존 metric 기준(질량합·원소 drift ≤0.01, Y≤1.01)이다. '
        '실제 drift 최대값은 아래 CSV 수치를 확인한다. '
        '질량·원소·엔탈피는 출력 보정이 강제하므로 ANN이 보존법칙이나 올바른 반응 경로를 학습했다는 뜻은 아니다. '
        'Entropy 비감소는 보장하지 않는다.', '',
        table(summary[['gamma', 'method', 'split', 'mass_sum_max', 'elemental_max']]), '',
        table(pd.read_csv(root / 'correction_diagnostics.csv')), '',
        '후보 통계는 rollout만 포함하며 mean 방식의 마지막 보정 호출도 포함한다. '
        '실패 원인은 `negative`, `inconsistent`, `thermal_invalid` 등을 함께 해석한다.', '',
        f'훈련·a priori 선택·전체 평가 총 실행시간: **{manifest["execution_seconds"]:.1f}초**. '
        '훈련/rollout wall-time을 기록했고 GPU/HPC는 사용하지 않았다.', '',
        table(pd.read_csv(root / 'timing.csv')[['gamma', 'method', 'split', 'seconds']]), '',
        '모든 방법/분할 12개에서 모델 재로딩 예측, 첫 self-fed step 일치, singleton seed 고정을 확인했다. '
        'Ranking의 all-member 경로와 singleton 추론도 일치 검증했다. '
        '입력 데이터 hash와 train parent/next-state identity를 확인했다. '
        f'전체 pytest: **{verification["pytest_tests"]} passed**, '
        f'실패/오류/skip 0개. [실행 로그]({root}/pytest.log), [검증 기록]({root}/verification.json). '
        'Keras/NumPy의 copy 키워드 deprecation 경고는 남아 있다.', '',
        '## 그림과 재현', '']
    for gamma in cfg['gammas']:
        label = f'g{gamma:g}'
        sections += [f'- γ={gamma:g}: [전체 test 온도]({figures}/{label}_temperature.png), '
            f'[IC44의 9종]({figures}/{label}_species_ic44.png)', '']
    sections += ['그림의 로그 시간축은 t=0 표시만 제외한다. 오차 계산은 t=0부터 전체 구간을 사용한다. '
        '종 그림은 symlog를 사용하며 원자료를 floor/clipping하지 않았다.', '',
        f'- [전체 요약]({root}/summary.csv), [IC별 결과]({root}/trajectories.csv), [종별 오차]({root}/species_profiles.csv)',
        f'- [Full-pair a priori]({root}/one_step.csv), [고정 선택]({root}/all_selections_frozen.json), [검증]({root}/reload_checks.csv)', '',
        '```bash', '/workspace/kinetics-venv/bin/python -m kinetics.preprocess_study --config configs/preprocess.json',
        '/workspace/kinetics-venv/bin/python -m pytest -q --junitxml=results/preprocess_tanh10/pytest.xml > results/preprocess_tanh10/pytest.log',
        '/workspace/kinetics-venv/bin/python -m kinetics.preprocess_verify',
        '/workspace/kinetics-venv/bin/python -m kinetics.preprocess_report', '```', '',
        '반복 실험은 config를 복사하여 새 output 경로를 사용한다. 기존 output은 기본 거부하며 '
        '`--resume`는 config·입력·모델 hash와 고정 선택을 검증한 뒤 이어간다. '
        '데이터/seed를 늘리거나 budget을 늘리지 않은 한 번의 matched ablation이다. '
        '10-member ensemble의 훈련 예산은 single의 10배이며 독립 ensemble 반복에 대한 통계적 유의성을 주장하지 않는다.']
    test = summary[summary.split == 'test']
    baseline = test[(test.gamma == control['gamma']) & (test.method == 'single_best')].iloc[0]
    findings = ['## 핵심 결과', '',
        f'공통 단일 대조군 γ={control["gamma"]:g}/seed{control["selected_seed"]}: '
        f'test 완주 {int(baseline.completed)}/18, 점화 분류 {100*baseline.ignition_classification:.1f}%, '
        f'완주 궤적 평균 T RMSE {baseline.temperature_profile_mean:.1f} K.', '']
    for gamma in cfg['gammas']:
        for method in ('mean10', 'medoid10'):
            r = test[(test.gamma == gamma) & (test.method == method)].iloc[0]
            findings += [f'- γ={gamma:g} {method}: 완주 {int(r.completed)}/18, '
                f'점화 분류 {100*r.ignition_classification:.1f}%, '
                f'T RMSE {r.temperature_profile_mean:.1f} K, '
                f'점화지연 조건부 중앙오차 {100*r.median_ignition_error_refined:.2f}% ({int(r.ignition_pairs)}쌍).']
    findings += ['', 'A priori 순위와 a posteriori 성능은 다른 지표다. '
        '아래 결과로 seed나 γ를 다시 선택하지 않았고, 개선 여부는 완주율·점화·전체 profile을 함께 비교해야 한다.', '']
    sections[4:4] = findings
    args.report.write_text('\n'.join(sections) + '\n')
    print('WROTE', args.report)


if __name__ == '__main__':
    main()
