"""Compare the frozen a priori-selected singleton with the Wang committees."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/kinetics-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/kinetics-cache')
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .wang_report import table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/single_apriori_wang'))
    parser.add_argument('--report', type=Path, default=Path('SINGLE_MODEL_REPORT.md'))
    args = parser.parse_args(); root = args.root
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    frozen = json.loads((root / 'selection_frozen.json').read_text())
    baseline = Path(manifest['baseline']); seed = frozen['selected_seed']
    with np.load(manifest['source']) as data:
        states, split, dt, species = data['states'], data['split'], float(data['dt']), data['species']
    ids = np.flatnonzero(split == 'test'); times = np.arange(states.shape[1])*dt*1e3
    figures = root / 'figures_log_time'; figures.mkdir(exist_ok=True)
    colors = dict(mean100='tab:blue', medoid100='tab:orange', single_best='tab:red')
    labels = dict(mean100='Mean 100', medoid100='Medoid 100', single_best=f'Single seed {seed}')
    traces = {}
    for name in colors:
        source = root if name == 'single_best' else baseline
        with np.load(source / 'traces' / f'{name}_test.npz') as archive:
            traces[name] = {int(i): archive[f'trajectory_{int(i)}'] for i in ids}

    def draw(ax, i, column):
        def line(values, **kwargs):
            v = values[1:, column]
            ax.plot(times[1:len(values)], np.maximum(v, 1e-20) if column else v, **kwargs)
        line(states[i], color='black', ls='--', lw=1.8, label='Cantera', zorder=5)
        for name, color in colors.items():
            trace = traces[name][i]
            line(trace, color=color, lw=1, label=labels[name])
            if 1 < len(trace) < len(times):
                ax.plot(times[len(trace)-1], max(trace[-1, column], 1e-20) if column else trace[-1, column],
                        marker='x', color=color, ms=6)
        ax.set(xscale='log', xlim=(times[1], times[-1]), xlabel='Time (ms, log scale)',
               ylabel='Y' if column else 'T (K)')
        ax.grid(True, alpha=.2)
        if column:
            ax.set(yscale='log', ylim=(1e-20, 1.1))
        else:
            ax.set_ylim(150, 3500)

    def save(fig, name):
        fig.tight_layout(rect=(0, .025, 1, .96))
        fig.text(.5, .008, 't=0 omitted from log display only; x marks an incomplete rollout endpoint.', ha='center')
        fig.savefig(figures / name, dpi=170); plt.close(fig)

    fig, axes = plt.subplots(3, 6, figsize=(22, 11), sharex=True)
    for i, ax in zip(ids, axes.flat):
        draw(ax, int(i), 0); ax.set_title(f'IC {i}: T0={states[i, 0, 0]:.0f} K')
    axes.flat[-1].legend(fontsize=7)
    fig.suptitle(f'Wang correction: best validation a priori single (seed {seed}) vs 100-member committees')
    save(fig, 'all_temperature_profiles.png')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for i, ax in zip([4, 44, 64, 73], axes.flat):
        draw(ax, i, 0); ax.set_title(f'IC {i}')
    axes.flat[-1].legend(fontsize=8)
    fig.suptitle('Same fixed 2 us step, same Wang correction; no model switching in single arm')
    save(fig, 'difficult_initial_conditions.png')
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True); i = int(ids[0])
    for j, ax in enumerate(axes.flat):
        draw(ax, i, j+1); ax.set_title(str(species[j]))
    axes.flat[-1].legend(fontsize=7)
    fig.suptitle(f'Nine species for held-out IC {i}; display-only mass-fraction floor 1e-20')
    save(fig, 'species_profiles.png')

    summary = pd.read_csv(root / 'comparison_summary.csv')
    def result_table(partition):
        frame = summary[summary.split == partition].copy()
        for name, column in [('완주 %', 'completion_fraction'), ('점화 구분 %', 'ignition_classification'),
                              ('점화오차 중앙값 %', 'median_ignition_error_refined')]:
            frame[name] = (100*frame[column]).round(2)
        frame['T profile 평균 K'] = frame.temperature_profile_mean.round(2)
        frame['T pooled K'] = frame.temperature_profile_pooled.round(2)
        return table(frame[['method', '완주 %', '점화 구분 %', '점화오차 중앙값 %', 'ignition_pairs',
                            'T profile 평균 K', 'T pooled K', 'species_profile_mean']])
    ranking = pd.read_csv(root / 'validation_member_ranking.csv')
    probe = pd.read_csv(root / 'comparison_one_step_probes.csv')
    probe = probe[probe.split == 'test'][['method', 'rmse', 'major_rmse', 'minor_rmse',
              'relative_log_mean', 'invalid_row_fraction', 'temperature_rmse']]
    timing = pd.read_csv(root / 'comparison_timing.csv')
    detail = pd.read_csv(root / 'trajectories.csv')
    test = detail[detail.split == 'test']
    worst = test.sort_values(['complete', 'prefix_temperature_rmse'], ascending=[True, False]).head(6)
    entropy = pd.read_csv(root / 'entropy_diagnostics.csv')
    entropy = entropy[entropy.split == 'test'].groupby('method').agg(
        min_temperature=('min_temperature', 'min'),
        entropy_decreasing_fraction=('decreasing_fraction', 'mean'),
        max_enthalpy_drift_J_kg=('max_enthalpy_drift_J_kg', 'max')).reset_index()
    s = summary[(summary.method == 'single_best') & (summary.split == 'test')].iloc[0]
    tests_path = root / 'tests.log'
    tests = tests_path.read_text().splitlines()[-1] if tests_path.exists() else 'See pytest execution log.'
    cases = []
    for row in test.itertuples():
        if row.reference_ignites and not row.prediction_ignites:
            cases.append(f'IC {row.trajectory}는 reference 점화를 놓쳤다.')
        if not row.complete:
            cases.append(f'IC {row.trajectory}는 {row.completed_steps:,} step({row.completed_steps*dt*1e3:.3f} ms) 후 중단했다. '
                         'Reference가 미점화여도 미완주 ANN을 정상 미점화/TN으로 세지 않는다.')
    correction = pd.read_csv(root / 'correction_diagnostics.csv')
    correction = correction[correction.split == 'test'].iloc[0]
    text = f'''# A priori 최상위 단일 모델 vs 100개 mean/medoid

동일한 100개 모델 중 **seed {seed} 모델을** validation one-step 물리 오차로 한 번 선택하고 고정했다.
Test 완주 {int(s.completed)}/18, 점화 구분 {100*s.ignition_classification:.1f}%,
조건부 점화지연 중앙오차 {100*s.median_ignition_error_refined:.2f}%다.

## 선택과 비교 조건

기존 학습 가중치는 그대로다. 18개 validation IC × 5,000개 고정 dt 쌍 = **90,000쌍 전부**에서 100개 모델을 각각 평가했다.
희소 one-step probe로 모델을 고른 것이 아니다. Prediction에는 앙상블과 동일한 Wang Box–Cox γ=.1 보정·decoder guard·엔탈피 복원을 적용했다.
`metrics.evaluate`의 기존 물리 단위 selection_score 최소 모델을 선택하고 동률은 seed 오름차순으로 처리했다.

`score = mean(log1p(min(|Ypred−Ytrue|/max(Ytrue,1e-8),1e12))) + 10×major_Y_RMSE + 10×mass_sum_MAE + 100×invalid_row_fraction`

Major 기준은 Y≥1e-3이다. Transformed-space MSE도 CSV에 기록했지만 선택에는 사용하지 않았다.
Temperature RMSE는 별도로 기록하며 이 기존 score에는 직접 들어가지 않는다.
이는 ‘모든 가능한 a priori 기준에서 가장 좋은 모델’이라는 뜻이 아니라 **명시한 score상 최상위 모델**이다.
Full validation은 평형 구간도 시간 비중대로 포함하므로 조용한 후기 구간이 순위에 큰 영향을 줄 수 있다.
Train downsampling을 validation에 적용하거나 test에 맞춰 선택 기준을 바꾸지 않았다.

선택 protocol을 평가 전에, 선택 모델/seed/hash를 모든 rollout 전에 저장했다.
**A posteriori 점화지연·profile·test 정확도는 모델 선택에 사용하지 않았고 결과가 나빠도 다른 seed로 바꾸지 않았다.**
단일 모델은 매 timestep 같은 한 네트워크만 실행하며 다른 99개 모델을 호출하거나 후보를 바꾸지 않는다.
Wang 보정으로 무효 후보가 생기면 실패한다. ODE/ensemble fallback은 없다.
이 결과는 이미 공개된 같은 test split에 대한 추가 ablation이며 독립적인 새 외부 검증은 아니다.

Validation 상위 5개:

{table(ranking[['seed', 'selection_score', 'rmse', 'major_rmse', 'minor_rmse', 'invalid_row_fraction', 'temperature_rmse', 'transformed_mse']].head(5))}

## 10 ms a posteriori

고정 dt=2 μs, 각 IC 최대 5,000 self-fed step이다. 정답 상태를 중간에 입력하지 않는다.
Mean100/medoid100은 앞서 완료한 동일 Wang 보정 실행 결과를 그대로 재사용했다.

Validation:

{result_table('val')}

Held-out test:

{result_table('test')}

점화지연은 400 K 상승 기준이며 reference만 0.2 μs로 정밀 측정했다. 오차 중앙값은 양쪽 모두 점화한 쌍에서 계산한다.
미점화·미완주를 작은 점화오차로 치환하지 않는다. 미완주의 full-profile RMSE는 NaN으로 제외하고 완주율/coverage를 함께 보고한다.
미완주 모델이 낮은 full-profile 평균을 보이더라도 성공으로 해석할 수 없다.

단일 모델의 큰 오차/실패 사례:

{table(worst[['trajectory', 'complete', 'coverage', 'completed_steps', 'reference_ignites', 'prediction_ignites', 'status', 'full_temperature_rmse', 'prefix_temperature_rmse']].round(3))}

{' '.join(cases)}
Rollout 보정 진단상 음수 후보 {int(correction.negative)}개, 열역학 무효 후보 {int(correction.thermal_invalid)}개였다.
보정 진단과 실패를 함께 해석해야 한다. 이번 singleton 실패를 음수 질량분율 때문이라고 해석할 수 없다.
반대로 IC 64는 단일 모델이 점화를 감지했지만 medoid100이 놓쳤다. 앙상블이 모든 조건에서 우월하다는 뜻은 아니다.

## 같은 probe에서의 a priori test 비교

아래 표만 기존 ensemble 평가와 같은 IC당 31개 시점 probe를 사용했다. 위의 90,000쌍 선택 ranking과 다른 표본이다.
전체 a posteriori profile은 downsampling하지 않았다.

{table(probe)}

One-step 최상위여도 self-fed trajectory가 학습 궤적을 벗어나면 성능이 달라질 수 있다.
Ensemble은 분산을 줄이거나 무효 후보를 제외할 여지가 있지만, 단일 모델은 그러한 여유가 없다.
이 비교는 한 번 선택한 seed의 결과이며 다수의 독립 ensemble 반복 통계가 아니다.
물리 단위 정확도와 transformed-space loss 개선을 구분해야 한다.

## 물리 진단과 시간

{table(entropy.round(6))}

질량·원소·엔탈피는 보정으로 강제된다. Entropy 비감소나 올바른 반응 경로는 보장되지 않는다.

{table(timing[['method', 'split', 'seconds']].round(4))}

시간은 같은 CPU에서 warmup을 뺀 진단 포함 rollout 시간이다. 미완주면 실행 step 자체가 짧으므로 전체 표와 함께 해석해야 한다.
모델 재훈련 없이 선택/단일 모델 평가 전체 {manifest['execution_seconds']:.1f}초를 사용했다.
저장 모델 재로딩, 첫 self-fed step 일치, 선택 seed 고정, ranking/runtime prediction 일치를 실행 중 확인했다.
전체 테스트: **{tests}**.

## 결과와 재현

- [로그 시간축 전체 온도](results/single_apriori_wang/figures_log_time/all_temperature_profiles.png), [어려운 조건](results/single_apriori_wang/figures_log_time/difficult_initial_conditions.png), [9종](results/single_apriori_wang/figures_log_time/species_profiles.png)
- [전체 100개 validation 순위](results/single_apriori_wang/validation_member_ranking.csv), [고정 선택 기록](results/single_apriori_wang/selection_frozen.json)
- [비교 CSV](results/single_apriori_wang/comparison_summary.csv), [초기조건별 결과](results/single_apriori_wang/trajectories.csv), [재로딩 검증](results/single_apriori_wang/reload_checks.csv)

```bash
python -m kinetics.single_apriori_study --output results/single_repeat
python -m kinetics.single_apriori_report --root results/single_repeat --report SINGLE_REPEAT_REPORT.md
```

기존 데이터·100개 가중치·앙상블 결과를 덮어쓰지 않았다.
'''
    args.report.write_text(text.replace('results/single_apriori_wang', str(root)))
    print(f'{args.report} written')


if __name__ == '__main__':
    main()
