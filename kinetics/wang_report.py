"""Create a comparison report from completed frozen-weight ANN-hard rollouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import mechanism


def table(frame):
    # Avoid an optional tabulate dependency.
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                      '|' + '|'.join(['---'] * len(frame.columns)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |'
                      for row in frame.itertuples(index=False, name=None)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/wang_hard10ms'))
    parser.add_argument('--report', type=Path, default=Path('WANG_HARD_REPORT.md'))
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    cfg = json.loads((root / 'config.json').read_text())
    with np.load(cfg['source']) as data:
        pressures = data['pressures']
        states, split, dt = data['states'], data['split'], float(data['dt'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = root / 'figures_log_time'
    figdir.mkdir(exist_ok=True)
    times = np.arange(states.shape[1]) * dt * 1e3
    fig, axes = plt.subplots(3, 6, figsize=(22, 11), sharex=True)
    with np.load(root / 'traces/medoid100_test.npz') as new_traces, np.load(
            Path(manifest['baseline']) / 'traces/medoid100_test.npz') as old_traces:
        for i, ax in zip(np.flatnonzero(split == 'test'), axes.flat):
            ax.plot(times[1:], states[i, 1:, 0], 'k--', lw=1.8, label='Cantera', zorder=5)
            for trace, color, label in [(old_traces[f'trajectory_{i}'], 'tab:blue', 'Previous QP'),
                                         (new_traces[f'trajectory_{i}'], 'tab:red', 'Wang hard')]:
                ax.plot(times[1:len(trace)], trace[1:, 0], color=color, lw=1, label=label)
            ax.set(xscale='log', xlim=(times[1], times[-1]), ylim=(150, 3500),
                   title=f'IC {i}', xlabel='Time (ms, log scale)', ylabel='T (K)')
            ax.grid(True, alpha=.2)
    axes.flat[-1].legend(fontsize=7)
    fig.suptitle('100-member medoid: same weights, previous QP vs Wang hard correction')
    fig.tight_layout(rect=(0, .025, 1, .96))
    fig.text(.5, .008, 't=0 omitted on log axis; first displayed time 0.002 ms.', ha='center')
    fig.savefig(figdir / 'medoid100_before_after.png', dpi=170)
    plt.close(fig)
    rows = []
    gas = mechanism()
    for partition in ('val', 'test'):
        for name in manifest['definitions']:
            with np.load(root / 'traces' / f'{name}_{partition}.npz') as traces:
                for key in traces.files:
                    i = int(key.split('_')[-1]); trace = traces[key]
                    entropy, enthalpy = [], []
                    for state in trace:
                        gas.TPY = state[0], pressures[i], state[1:]
                        entropy.append(gas.entropy_mass)
                        enthalpy.append(gas.enthalpy_mass)
                    entropy = np.asarray(entropy); difference = np.diff(entropy)
                    rows.append(dict(method=name, split=partition, trajectory=i,
                        points=len(trace), min_temperature=float(trace[:, 0].min()),
                        max_enthalpy_drift_J_kg=float(np.max(np.abs(np.asarray(enthalpy)-enthalpy[0]))),
                        decreasing_fraction=float(np.mean(difference < -1e-6)) if len(difference) else np.nan,
                        largest_step_drop=float(max(0., -difference.min())) if len(difference) else np.nan,
                        minimum_delta_from_initial=float((entropy-entropy[0]).min())))
    entropy = pd.DataFrame(rows)
    entropy.to_csv(root / 'entropy_diagnostics.csv', index=False)
    comp = pd.read_csv(root / 'comparison.csv')
    def comparison(partition):
        frame = comp[comp.split == partition].copy()
        frame['correction'] = frame.correction.str.replace('previous_Y_weighted_nonnegative_QP', '기존 QP').str.replace('Wang_G_linearized', 'Wang hard')
        frame['완주 %'] = (100 * frame.completion_fraction).round(1)
        frame['점화 구분 %'] = (100 * frame.ignition_classification).round(1)
        frame['점화오차 중앙값 %'] = (100 * frame.median_ignition_error_refined).round(2)
        frame['T profile RMSE 평균 K'] = frame.temperature_profile_mean.round(2)
        frame['T pooled RMSE K'] = frame.temperature_profile_pooled.round(2)
        frame['Y profile RMSE 평균'] = frame.species_profile_mean.map(lambda x: f'{x:.4e}')
        frame['점화 쌍 수'] = frame.ignition_pairs
        return table(frame[['method', 'correction', '완주 %', '점화 구분 %', '점화오차 중앙값 %',
                            '점화 쌍 수', 'T profile RMSE 평균 K', 'T pooled RMSE K', 'Y profile RMSE 평균']])
    diag = pd.read_csv(root / 'correction_diagnostics.csv')
    diag['negative %'] = (100 * diag.negative / diag.proposals).round(3)
    conservation = comp[comp.split == 'test'][['method', 'correction', 'mass_sum_max', 'elemental_max']]
    entropy_summary = entropy[entropy.split == 'test'].groupby('method').agg(
        min_temperature=('min_temperature', 'min'),
        max_enthalpy_drift_J_kg=('max_enthalpy_drift_J_kg', 'max'),
        entropy_decreasing_fraction=('decreasing_fraction', 'mean'),
        worst_entropy_step_drop=('largest_step_drop', 'max')).reset_index()
    test = comp[comp.split == 'test']
    findings = []
    for name in ('mean100', 'medoid100'):
        pair = test[test.method == name].set_index('correction')
        old, new = pair.loc['previous_Y_weighted_nonnegative_QP'], pair.loc['Wang_G_linearized']
        findings.append(f'- {name}: T profile 평균 RMSE {old.temperature_profile_mean:.1f}→{new.temperature_profile_mean:.1f} K; '
                        f'점화 구분 {100*old.ignition_classification:.1f}→{100*new.ignition_classification:.1f}%; '
                        f'조건부 점화오차 중앙값 {100*old.median_ignition_error_refined:.2f}→{100*new.median_ignition_error_refined:.2f}%.')
    detail = pd.read_csv(root / 'trajectories.csv')
    worst = detail[(detail.split == 'test') & (detail.method == 'medoid100')].nlargest(5, 'full_temperature_rmse')
    worst_table = table(worst[['trajectory', 'reference_ignites', 'prediction_ignites', 'full_temperature_rmse']].round(2))
    tests = (root / 'tests.log').read_text().splitlines()[-1] if (root / 'tests.log').exists() else 'See pytest execution log.'
    text = f'''# Wang ANN-hard: 동일 가중치의 10 ms a posteriori 비교

첨부 논문: Wang et al., *Enforcing physical conservation in neural network surrogate models for complex chemical kinetics*, Combustion and Flame 275 (2025) 114105. DOI: https://doi.org/10.1016/j.combustflame.2025.114105

## 주요 관찰

{chr(10).join(findings)}

점화오차 중앙값과 전체 profile 오차는 별개이며 한 metric의 감소가 나머지의 개선을 뜻하지 않는다.
Medoid100은 profile RMSE의 궤적별 평균은 줄었으나 pooled RMSE는 456.4→459.0 K로 소폭 커졌다.
따라서 가장 큰 실패가 개선됐다고 해석할 수 없다. Mean100의 test 개선도 validation에서는 T 오차가 증가하므로 일관된 우위로 단정하지 않는다.

## 구현과 기존 방법의 차이

논문 §2.2 식 (5)–(9)를 구현했다. A는 H/O/N 원소 질량분율 행렬이고,
v는 ANN의 다음 질량분율 후보다. γ=.1에서 D=diag(v^(1−γ)), B=A D,
E=A(Y현재−v), ΔG=Bᵀ(BBᵀ)⁺E, Y보정=v+DΔG를 사용한다.
최소화하는 것은 **표준화하지 않은 G 공간의 보정 L2 norm**이다.
G에 ΔG를 더하고 비선형 inverse Box–Cox를 적용하는 알고리즘이 아니다.
선형 Y 보정을 적용하므로 풀 수 있는 제약은 반올림 오차 내에서 정확히 보존된다.

기존 방법은 Y 공간 가중 QP이며 가중 scale=max(|v|,Y현재,1e-8), 비음수 제약을 포함한다.
Wang은 predicted v만으로 scale=v^.9를 사용하고 논문 식에는 비음수 제약이 없다.
따라서 기존 결과에 보정을 중복 적용하지 않고 **원소 보정 단계만 교체**했다.
공통 decoder [0,1] guard와 NASA/Cantera 엔탈피 온도 복원은 그대로다.
Wang 식에서 생긴 음수 후보는 clip하거나 기존 QP로 재보정하지 않고 전체 후보를 제외했다.
전부 무효면 실패한다. 이 거부 정책은 본 benchmark의 물리 허용 정책이며 논문의 추가 식이 아니다.
0의 correction weight는 0으로 두며 임의 epsilon floor는 추가하지 않았다.

Appendix A pseudocode의 첫 residual 부호는 식 (7) 및 마지막 덧셈과 일치하지 않는다.
본 구현은 식 (7)의 E=A(Y현재−v)를 따른다. 행 scaling과 pseudoinverse는 수치 해법이며
제약/목적함수를 바꾸지 않는다. 불충족 residual>1e-10은 제외한다.

## 고정된 비교 조건

동일한 새 10-ms 데이터로 이미 훈련된 100개 모델을 재사용했다. **재학습·parameter tuning은 없다.**
10개 nested control, 100개, mean-after-correction/medoid의 네 policy를 모두 유지했다.
Box–Cox γ=.1, T/P/Y 입력, 다음 Y 출력, dt=2 μs, 5,000 self-fed step이다.
온도는 다음 Y와 현재 엔탈피·압력에서 복원한다. 각 후보를 보정한 다음 평균 또는 대표를 정한다.
모든 policy는 같은 IC에서 독립적으로 적분하고 중간 정답은 입력하지 않는다.
기존 train-only 거리 scale과 모델 SHA256을 확인했다. Validation/test 각각 18개이며 test tuning은 하지 않았다.
논문 자체의 큰 network, 19-reaction mechanism, dt=1 μs, 훈련 dataset를 재현한 실험은 아니다.
이번 실험은 우리 9-species/28-reaction 시스템의 correction ablation이다.
Chemical ODE fallback과 entropy constraint는 없다. 전체 실행 {manifest['execution_seconds']:.1f}초, CPU만 사용했다.

## Validation

{comparison('val')}

## Held-out test

{comparison('test')}

점화오차는 400 K 상승 기준, reference만 0.2 μs로 측정한 값과 비교했다.
양쪽 모두 점화한 쌍의 조건부 중앙값이며 쌍 수도 표시했다. 미점화·실패를 작은 오차로 치환하지 않는다.
전체 profile RMSE는 선형 실제 시간의 모든 2 μs 점을 사용하며 로그 그림으로 바꿔도 metric은 같다.
미완주 궤적의 full-profile RMSE는 NaN이며 완주율과 함께 해석해야 한다.

## 보정 후보 진단

{table(diag)}

proposal 수에는 평균 후 공통 엔탈피 재복원 호출도 포함된다. 위 진단은 rollout만 포함하고 warmup과 one-step probe는 제외했다.
negative/inconsistent 집계는 중복 가능하다. 음수 후보를 제외하기 때문에 accepted output의 음수율만 보면 위험을 놓칠 수 있다.

## 보존성과 장기 거동

{table(conservation)}

{table(entropy_summary.round(6))}

질량·원소·엔탈피 보존은 hard correction으로 강제되며 신경망이 학습한 정확도로 해석하지 않는다.
이 제약만으로 entropy 증가, 반응 방향, 평형 고정점 또는 올바른 점화지연을 보장하지 않는다.
특히 trace에서 v^.9 weight가 작아 trace correction은 작고 major species가 제약을 맞추는 역할을 한다.
이는 미량종에 과도한 절대량 보정을 주는 문제를 줄이지만, 잘못 예측된 radical과 off-manifold 상태를 올바른 반응 경로로 돌리는 장치는 아니다.
기존 Y² weighting과 지수/zero floor/비음수 제약이 함께 다르므로 변화 전체를 한 가지 원인으로 단정하지 않는다.
Entropy 감소와 후기 profile 오차가 남으면 보존성 개선만으로 안정성을 주장할 수 없다.
특히 medoid100 test의 큰 오차 사례는 다음과 같다.

{worst_table}

IC 64는 reference 점화를 놓치고 냉각되며, IC 4는 점화 후 후기 profile이 무너진다.
이는 원소/엔탈피 제약을 만족하는 상태도 잘못된 chemistry 경로에 있을 수 있음을 보여준다.
학습 가중치/목적함수는 바꾸지 않았으므로 transformed-space 학습 loss 개선을 주장하지 않는다.
판단 근거는 실제 물리 단위 one-step CSV, 점화/완주율과 full-profile 오차다.

## 그림과 재현

테스트: **{tests}**. 식 (6)–(9)의 직접 선형 해와 일치, residual 부호, 0/1e-300 농도,
불충족 zero-support와 음수 보정, 질량·원소·엔탈피를 검사했다.
저장된 네 policy를 다시 불러와 첫 step과 대표 seed/후보 identity가 기록과 같은지 확인했다.
[재로딩 검증](results/wang_hard10ms/reload_checks.csv), [소스 SHA256](results/wang_hard10ms/source_hashes.json)을 남겼다.

- [로그 시간축 온도](results/wang_hard10ms/figures_log_time/all_temperature_profiles.png)
- [100개 medoid 보정 전후 비교](results/wang_hard10ms/figures_log_time/medoid100_before_after.png)
- [어려운 초기조건](results/wang_hard10ms/figures_log_time/difficult_initial_conditions.png)
- [9종 profile](results/wang_hard10ms/figures_log_time/species_profiles.png)
- [비교 CSV](results/wang_hard10ms/comparison.csv), [초기조건별](results/wang_hard10ms/trajectories.csv), [one-step 물리 단위](results/wang_hard10ms/one_step_probes.csv)
- [보정 진단](results/wang_hard10ms/correction_diagnostics.csv), [엔트로피](results/wang_hard10ms/entropy_diagnostics.csv), [시간](results/wang_hard10ms/timing.csv)

```bash
/workspace/kinetics-venv/bin/python -m kinetics.wang_study --output results/wang_hard10ms
/workspace/kinetics-venv/bin/python -m kinetics.log_time_profiles --root results/wang_hard10ms
/workspace/kinetics-venv/bin/python -m kinetics.wang_report
/workspace/kinetics-venv/bin/python -m kinetics.verify_wang
/workspace/kinetics-venv/bin/python -m pytest -q
```

기존 dataset·weights·baseline 결과는 그대로 보존했다. 새 결과는 `results/wang_hard10ms`에 분리했다.
'''
    args.report.write_text(text.replace('results/wang_hard10ms', str(root)))
    print(f'{args.report} written')


if __name__ == '__main__':
    main()
