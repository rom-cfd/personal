# A priori 최상위 단일 모델 vs 100개 mean/medoid

동일한 100개 모델 중 **seed 11 모델을** validation one-step 물리 오차로 한 번 선택하고 고정했다.
Test 완주 17/18, 점화 구분 88.9%,
조건부 점화지연 중앙오차 8.94%다.

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

| seed | selection_score | rmse | major_rmse | minor_rmse | invalid_row_fraction | temperature_rmse | transformed_mse |
|---|---|---|---|---|---|---|---|
| 11 | 0.0041195204982243 | 4.887641584446024e-05 | 6.568185191610393e-05 | 3.4196813350933216e-06 | 0.0 | 1.6045880581555867 | 1.4617255863060152e-05 |
| 1001 | 0.0042158423684576 | 5.526687365414989e-05 | 7.431122453539261e-05 | 2.7038457273028065e-06 | 0.0 | 1.594323290034375 | 7.335333553593726e-06 |
| 968 | 0.0042205391770435 | 5.116247625696901e-05 | 6.880105881805392e-05 | 2.19332083558608e-06 | 0.0 | 1.4071234527341785 | 7.335235262962877e-06 |
| 22 | 0.0043167033053948 | 5.444186335519673e-05 | 7.320777639053851e-05 | 2.456883387890135e-06 | 0.0 | 1.1727386303196283 | 1.00542110786705e-05 |
| 957 | 0.004330040570548 | 4.576012394265837e-05 | 6.149675424131377e-05 | 3.1373129702666585e-06 | 0.0 | 1.2805814795838095 | 1.0367610909908372e-05 |

## 10 ms a posteriori

고정 dt=2 μs, 각 IC 최대 5,000 self-fed step이다. 정답 상태를 중간에 입력하지 않는다.
Mean100/medoid100은 앞서 완료한 동일 Wang 보정 실행 결과를 그대로 재사용했다.

Validation:

| method | 완주 % | 점화 구분 % | 점화오차 중앙값 % | ignition_pairs | T profile 평균 K | T pooled K | species_profile_mean |
|---|---|---|---|---|---|---|---|
| mean100 | 100.0 | 94.44 | 4.19 | 15 | 310.32 | 542.4 | 0.0149714203430411 |
| medoid100 | 100.0 | 100.0 | 4.37 | 16 | 195.63 | 404.2 | 0.0096491798263073 |
| single_best | 94.44 | 83.33 | 7.35 | 13 | 546.33 | 820.93 | 0.0224339261125255 |

Held-out test:

| method | 완주 % | 점화 구분 % | 점화오차 중앙값 % | ignition_pairs | T profile 평균 K | T pooled K | species_profile_mean |
|---|---|---|---|---|---|---|---|
| mean100 | 100.0 | 100.0 | 6.24 | 15 | 175.12 | 242.1 | 0.0120377640107066 |
| medoid100 | 100.0 | 94.44 | 5.13 | 14 | 252.75 | 459.0 | 0.0108538473858414 |
| single_best | 94.44 | 88.89 | 8.94 | 14 | 561.2 | 821.05 | 0.0288819714765523 |

점화지연은 400 K 상승 기준이며 reference만 0.2 μs로 정밀 측정했다. 오차 중앙값은 양쪽 모두 점화한 쌍에서 계산한다.
미점화·미완주를 작은 점화오차로 치환하지 않는다. 미완주의 full-profile RMSE는 NaN으로 제외하고 완주율/coverage를 함께 보고한다.
미완주 모델이 낮은 full-profile 평균을 보이더라도 성공으로 해석할 수 없다.

단일 모델의 큰 오차/실패 사례:

| trajectory | complete | coverage | completed_steps | reference_ignites | prediction_ignites | status | full_temperature_rmse | prefix_temperature_rmse |
|---|---|---|---|---|---|---|---|---|
| 78 | False | 0.462 | 2311 | False | False | no_valid_prediction | nan | 184.249 |
| 44 | True | 1.0 | 5000 | True | False | completed | 1934.209 | 1934.209 |
| 45 | True | 1.0 | 5000 | True | True | completed | 1718.282 | 1718.282 |
| 68 | True | 1.0 | 5000 | True | True | completed | 1667.191 | 1667.191 |
| 5 | True | 1.0 | 5000 | True | True | completed | 673.847 | 673.847 |
| 64 | True | 1.0 | 5000 | True | True | completed | 616.266 | 616.266 |

IC 44는 reference 점화를 놓쳤다. IC 78는 2,311 step(4.622 ms) 후 중단했다. Reference가 미점화여도 미완주 ANN을 정상 미점화/TN으로 세지 않는다.
Rollout 보정 진단상 음수 후보 0개, 열역학 무효 후보 1개였다.
보정 진단과 실패를 함께 해석해야 한다. 이번 singleton 실패를 음수 질량분율 때문이라고 해석할 수 없다.
반대로 IC 64는 단일 모델이 점화를 감지했지만 medoid100이 놓쳤다. 앙상블이 모든 조건에서 우월하다는 뜻은 아니다.

## 같은 probe에서의 a priori test 비교

아래 표만 기존 ensemble 평가와 같은 IC당 31개 시점 probe를 사용했다. 위의 90,000쌍 선택 ranking과 다른 표본이다.
전체 a posteriori profile은 downsampling하지 않았다.

| method | rmse | major_rmse | minor_rmse | relative_log_mean | invalid_row_fraction | temperature_rmse |
|---|---|---|---|---|---|---|
| mean100 | 0.0001190697164997 | 0.0001702112351555 | 5.36790415433796e-06 | 0.0073471445816666 | 0.0 | 7.466694157777852 |
| medoid100 | 0.0001327910220942 | 0.0001898684295373 | 4.5186430663851005e-06 | 0.0077664789815333 | 0.0 | 8.042040960207817 |
| single_best | 0.000131846458432 | 0.0001885030774883 | 5.045861284316431e-06 | 0.0080705389177397 | 0.0 | 6.18756723749034 |

One-step 최상위여도 self-fed trajectory가 학습 궤적을 벗어나면 성능이 달라질 수 있다.
Ensemble은 분산을 줄이거나 무효 후보를 제외할 여지가 있지만, 단일 모델은 그러한 여유가 없다.
이 비교는 한 번 선택한 seed의 결과이며 다수의 독립 ensemble 반복 통계가 아니다.
물리 단위 정확도와 transformed-space loss 개선을 구분해야 한다.

## 물리 진단과 시간

| method | min_temperature | entropy_decreasing_fraction | max_enthalpy_drift_J_kg |
|---|---|---|---|
| single_best | 200.067702 | 0.533674 | 0.016546 |

질량·원소·엔탈피는 보정으로 강제된다. Entropy 비감소나 올바른 반응 경로는 보장되지 않는다.

| method | split | seconds |
|---|---|---|
| mean100 | val | 69.2125 |
| medoid100 | val | 108.0704 |
| mean100 | test | 72.601 |
| medoid100 | test | 136.1147 |
| single_best | val | 10.0913 |
| single_best | test | 10.1636 |

시간은 같은 CPU에서 warmup을 뺀 진단 포함 rollout 시간이다. 미완주면 실행 step 자체가 짧으므로 전체 표와 함께 해석해야 한다.
모델 재훈련 없이 선택/단일 모델 평가 전체 97.6초를 사용했다.
저장 모델 재로딩, 첫 self-fed step 일치, 선택 seed 고정, ranking/runtime prediction 일치를 실행 중 확인했다.
전체 테스트: **62 passed, 20 warnings in 7.51s**.

## 결과와 재현

- [로그 시간축 전체 온도](results/single_apriori_wang/figures_log_time/all_temperature_profiles.png), [어려운 조건](results/single_apriori_wang/figures_log_time/difficult_initial_conditions.png), [9종](results/single_apriori_wang/figures_log_time/species_profiles.png)
- [전체 100개 validation 순위](results/single_apriori_wang/validation_member_ranking.csv), [고정 선택 기록](results/single_apriori_wang/selection_frozen.json)
- [비교 CSV](results/single_apriori_wang/comparison_summary.csv), [초기조건별 결과](results/single_apriori_wang/trajectories.csv), [재로딩 검증](results/single_apriori_wang/reload_checks.csv)

```bash
python -m kinetics.single_apriori_study --output results/single_repeat
python -m kinetics.single_apriori_report --root results/single_repeat --report SINGLE_REPEAT_REPORT.md
```

기존 데이터·100개 가중치·앙상블 결과를 덮어쓰지 않았다.
