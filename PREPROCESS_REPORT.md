# Box–Cox γ=0.2 / 0.5: 10-member tanh ANN 재훈련

최신 `kisung` 커밋 `b5ecd33`에서 만든 `pre-process` 브랜치의 실험이다. 각 γ에서 10개를 새로 훈련했고, 전체 20개 중 validation a priori 최상위 단일 모델 하나를 공통 대조군으로 고정했다. 각 γ 내부의 최상위 singleton도 보조 대조군으로 함께 평가했다. 기존 데이터·가중치·결과는 덮어쓰지 않았다.

## 핵심 결과

공통 단일 대조군 γ=0.2/seed66: test 완주 18/18, 점화 분류 100.0%, 완주 궤적 평균 T RMSE 236.8 K.

- γ=0.2 mean10: 완주 18/18, 점화 분류 83.3%, T RMSE 387.4 K, 점화지연 조건부 중앙오차 6.75% (13쌍).
- γ=0.2 medoid10: 완주 18/18, 점화 분류 88.9%, T RMSE 412.5 K, 점화지연 조건부 중앙오차 8.92% (14쌍).
- γ=0.5 mean10: 완주 18/18, 점화 분류 83.3%, T RMSE 594.6 K, 점화지연 조건부 중앙오차 4.47% (15쌍).
- γ=0.5 medoid10: 완주 18/18, 점화 분류 88.9%, T RMSE 343.3 K, 점화지연 조건부 중앙오차 5.31% (15쌍).

A priori 순위와 a posteriori 성능은 다른 지표다. 아래 결과로 seed나 γ를 다시 선택하지 않았고, 개선 여부는 완주율·점화·전체 profile을 함께 비교해야 한다.

## 조건과 선택 규칙

- 구조: **64–64–64 tanh**, 모델당 9,738개 파라미터. GELU/1600–800–400 paper 모델은 사용하지 않았다.
- 기존 9종/28반응 수소계, train/validation/test IC 54/18/18개, 10 ms, dt=2 μs. Train-only 평형 downsampling으로 고정된 **8,815쌍**을 그대로 재사용했다. Augmentation은 없다.
- γ별 seed 11,22,…,110의 10개, 모델당 10,000 Adam update, batch512, lr .001→.0003→.0001, 멤버별 norm clip5. 총 20개/200,000 member-updates. 5개씩 계산만 묶고 가중치·Adam state·minibatch RNG는 독립적이다.
- γ 사이에는 같은 seed 목록을 매칭하여 초기화·minibatch 차이를 통제했다. γ만 바꾼 두 집합이지 서로 독립적으로 재추출한 ensemble 반복은 아니다.
- 입력은 T(t), log P, Box–Cox(Y(t)). 다음 상태의 정규화된 transformed residual을 학습한다. Box–Cox는 Y^γ/γ이며 표준화하면 (Y^γ−1)/γ와 affine-equivalent다. 전처리 평균/표준편차는 train 입력·target만으로 적합했다.
- Wang 보정의 역미분 가중치도 **Y^(1−γ)**로 일치시켰다. ANN T 출력은 보정 추론에서 버리고 현재 엔탈피와 다음 Y, 고정 P로 T를 복원한다. Decoder의 기존 [0,1] domain guard를 유지하고, 보정 후 음수/불능 후보는 거부한다.
- Mean10: 멤버별 보정 후 유효 후보 평균, 원소/엔탈피 복원. Medoid10: train-fitted scaled 거리로 실제 보정 후보 하나 선택. ODE fallback·hysteresis는 없다.
- 단일 모델: γ별 10개 모두를 **validation 18×5,000=90,000쌍 전부**에서 평가하여 기존 물리 단위 selection_score 최소 모델을 선택. 동률은 seed 오름차순. 두 γ의 선택 기록을 모두 저장한 뒤에만 rollout/test 평가를 시작했다.

`selection_score = mean(log1p(min(|Ypred−Ytrue|/max(Ytrue,1e−8),1e12))) + 10×major_Y_RMSE + 10×mass_sum_MAE + 100×invalid_row_fraction`

Major 기준은 Y≥1e−3이다. 이 score에는 T RMSE가 직접 들어가지 않는다. 따라서 “가장 좋은 모델”은 이 명시적 a priori score상 최상위라는 뜻이다. Validation은 평형 후기 구간도 원래 시간 비중으로 포함하므로 그 구간이 순위에 큰 영향을 줄 수 있다. A posteriori 결과나 test 오차로 seed·γ·aggregation을 재선택하지 않았다. 기존과 같은 이미 관측된 test split의 ablation이며 새로운 외부 일반화 검증은 아니다.

## 고정한 단일 대조군

| gamma | seed | validation_score | Y_RMSE | T_RMSE_K | invalid_fraction |
|---|---|---|---|---|---|
| 0.2 | 66 | 0.0046807 | 3.792e-05 | 1.2368 | 0 |
| 0.5 | 77 | 0.027614 | 4.9188e-05 | 1.4102 | 0 |

**공통 단일 대조군: γ=0.2, seed 66**, validation score=0.0046806669. 두 γ의 최상위 중 같은 score로 선택했으므로 전체 20개 중 최상위다. 각 γ 전체 순위는 각 하위 폴더의 `validation_member_ranking.csv`에 있다.

## A priori: 동일한 전체 90,000쌍 비교

모든 방법에 동일한 truth 입력을 주는 one-step 오차다. 아래 test와 validation 모두 full-pair 평가이며 희소 probe가 아니다. Self-fed a posteriori와 구분한다.

### val

| gamma | method | samples | selection_score | rmse | major_rmse | minor_rmse | temperature_rmse | invalid_row_fraction |
|---|---|---|---|---|---|---|---|---|
| 0.2 | single_best | 90000 | 0.0046807 | 3.792e-05 | 5.0959e-05 | 2.6225e-06 | 1.2368 | 0 |
| 0.2 | mean10 | 90000 | 0.0059672 | 3.8831e-05 | 5.2209e-05 | 1.987e-06 | 0.96057 | 0 |
| 0.2 | medoid10 | 90000 | 0.0053777 | 3.9065e-05 | 5.2531e-05 | 1.7694e-06 | 0.89905 | 0 |
| 0.5 | single_best | 90000 | 0.027614 | 4.9188e-05 | 6.6091e-05 | 3.6742e-06 | 1.4102 | 0 |
| 0.5 | mean10 | 90000 | 0.033331 | 4.683e-05 | 5.9437e-05 | 2.3211e-05 | 1.1302 | 0 |
| 0.5 | medoid10 | 90000 | 0.025251 | 4.2104e-05 | 5.4299e-05 | 1.7916e-05 | 1.2031 | 0 |

### test

| gamma | method | samples | selection_score | rmse | major_rmse | minor_rmse | temperature_rmse | invalid_row_fraction |
|---|---|---|---|---|---|---|---|---|
| 0.2 | single_best | 90000 | 0.0053229 | 4.6397e-05 | 6.4409e-05 | 3.4901e-06 | 1.4241 | 0 |
| 0.2 | mean10 | 90000 | 0.0036463 | 3.2989e-05 | 4.5832e-05 | 1.615e-06 | 1.0575 | 0 |
| 0.2 | medoid10 | 90000 | 0.0048716 | 3.6989e-05 | 5.1386e-05 | 1.9411e-06 | 1.1185 | 0 |
| 0.5 | single_best | 90000 | 0.022284 | 5.0182e-05 | 6.9672e-05 | 3.6044e-06 | 1.45 | 0 |
| 0.5 | mean10 | 90000 | 0.022533 | 5.1499e-05 | 7.1527e-05 | 3.0992e-06 | 1.2805 | 0 |
| 0.5 | medoid10 | 90000 | 0.026552 | 5.4168e-05 | 7.522e-05 | 3.5829e-06 | 1.2813 | 0 |

## A posteriori: 10 ms 전체 self-fed 예측

각 IC의 초기 상태만 입력하고 이후 5,000 step을 자기 예측으로 전진한다. 중간 Cantera 상태 주입이나 rollout downsampling은 없다. 점화는 초기 T 대비 400 K 상승 시점이며 reference는 기존 0.2 μs 정밀 Cantera τ를 재사용했다. 점화오차는 양쪽 모두 점화한 쌍의 조건부 중앙값으로, 쌍 수를 함께 표시한다. 미완주 full-profile 오차는 NaN으로 제외하고 coverage/실패를 따로 보고한다.

### val

| gamma | method | completed | completion_% | physical_% | ignition_accuracy_% | ignition_error_% | ignition_pairs | temperature_profile_mean | temperature_profile_pooled | species_profile_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | single_best | 18 | 100 | 100 | 83.333 | 5.0419 | 13 | 452.64 | 680.32 | 0.020001 |
| 0.2 | mean10 | 18 | 100 | 100 | 83.333 | 4.3487 | 14 | 391.13 | 669.77 | 0.021272 |
| 0.2 | medoid10 | 18 | 100 | 100 | 94.444 | 9.4884 | 16 | 750.5 | 952.86 | 0.035219 |
| 0.5 | single_best | 17 | 94.444 | 94.444 | 88.889 | 4.2716 | 14 | 356.49 | 630.53 | 0.021062 |
| 0.5 | mean10 | 17 | 94.444 | 94.444 | 83.333 | 4.6061 | 14 | 613.94 | 916.45 | 0.033691 |
| 0.5 | medoid10 | 17 | 94.444 | 94.444 | 83.333 | 4.5533 | 13 | 330.34 | 632.08 | 0.018803 |

### test

| gamma | method | completed | completion_% | physical_% | ignition_accuracy_% | ignition_error_% | ignition_pairs | temperature_profile_mean | temperature_profile_pooled | species_profile_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | single_best | 18 | 100 | 100 | 100 | 6.4757 | 15 | 236.78 | 318.96 | 0.012086 |
| 0.2 | mean10 | 18 | 100 | 100 | 83.333 | 6.7522 | 13 | 387.41 | 635.19 | 0.021343 |
| 0.2 | medoid10 | 18 | 100 | 100 | 88.889 | 8.9224 | 14 | 412.51 | 631.61 | 0.022795 |
| 0.5 | single_best | 18 | 100 | 100 | 100 | 5.3646 | 15 | 208.19 | 381.1 | 0.010668 |
| 0.5 | mean10 | 18 | 100 | 100 | 83.333 | 4.4684 | 15 | 594.61 | 789.57 | 0.031211 |
| 0.5 | medoid10 | 18 | 100 | 100 | 88.889 | 5.3116 | 15 | 343.27 | 516.71 | 0.019114 |

### 공통 최상위 단일 대조군 대비

| gamma | method | completion_delta_pp | ignition_accuracy_delta_pp | T_RMSE_delta_K | Y_RMSE_delta |
|---|---|---|---|---|---|
| 0.2 | mean10 | 0 | -16.667 | 150.63 | 0.0092565 |
| 0.2 | medoid10 | 0 | -11.111 | 175.73 | 0.010709 |
| 0.5 | mean10 | 0 | -16.667 | 357.82 | 0.019125 |
| 0.5 | medoid10 | 0 | -11.111 | 106.49 | 0.0070278 |

차이는 ensemble−single이다. RMSE 차이가 음수면 완주 궤적 평균 오차가 더 작다. 완주 집합이 다르면 이 차이를 전체 성능 개선으로 단정할 수 없다.

### 기존 γ=0.1의 10-member 결과: 역사적 참고

| gamma | method | completed | ignition_classification | median_ignition_error_refined | temperature_profile_mean | species_profile_mean |
|---|---|---|---|---|---|---|
| 0.1 | mean10 | 18 | 1 | 0.06135 | 172.51 | 0.012057 |
| 0.1 | medoid10 | 18 | 1 | 0.054458 | 292.2 | 0.016637 |
| 0.1 | historical_single_best_of_100_seed11 | 17 | 0.88889 | 0.089434 | 561.2 | 0.028882 |

기존 Wang 실행의 같은 데이터 hash·64/64/64 tanh·첫 10 seed·모델당 10,000-update 결과를 그대로 재사용했다. 이전 a priori 최상위 γ=.1/seed11 단일 모델도 기존 결과로 함께 표시했다. 이 모델은 과거 100개 중 고른 모델로 이번 10개당 선택과 후보 풀 크기가 다르다. γ=.1은 이번에 재훈련하지 않았고, 새 20개의 singleton 공통 대조군 선택 후보에도 넣지 않았다. 위 역사적 표의 ignition 지표는 비율(0–1)이다. 새 γ는 5개씩 grouped training했고 기존 γ=.1은 8개씩 계산했으므로 수치 연산 순서의 작은 차이가 있을 수 있다.

### Test 실패·점화 불일치

| gamma | method | trajectory | complete | coverage | completed_steps | reference_ignites | prediction_ignites | ignition_status | full_temperature_rmse | prefix_temperature_rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | mean10 | 5 | True | 1 | 5000 | True | False | missed_ignition | 1197.9 | 1197.9 |
| 0.2 | mean10 | 64 | True | 1 | 5000 | True | False | missed_ignition | 1073.7 | 1073.7 |
| 0.2 | mean10 | 78 | True | 1 | 5000 | False | True | false_ignition_within_horizon | 1676.4 | 1676.4 |
| 0.2 | medoid10 | 64 | True | 1 | 5000 | True | False | missed_ignition | 1098.9 | 1098.9 |
| 0.2 | medoid10 | 78 | True | 1 | 5000 | False | True | false_ignition_within_horizon | 1741.8 | 1741.8 |
| 0.5 | mean10 | 7 | True | 1 | 5000 | False | True | false_ignition_within_horizon | 1286.5 | 1286.5 |
| 0.5 | mean10 | 69 | True | 1 | 5000 | False | True | false_ignition_within_horizon | 1027.8 | 1027.8 |
| 0.5 | mean10 | 78 | True | 1 | 5000 | False | True | false_ignition_within_horizon | 1538.6 | 1538.6 |
| 0.5 | medoid10 | 7 | True | 1 | 5000 | False | True | false_ignition_within_horizon | 752.33 | 752.33 |
| 0.5 | medoid10 | 78 | True | 1 | 5000 | False | True | false_ignition_within_horizon | 1425.2 | 1425.2 |

## 물리 진단과 실행 검증

물리 허용 완주는 기존 metric 기준(질량합·원소 drift ≤0.01, Y≤1.01)이다. 실제 drift 최대값은 아래 CSV 수치를 확인한다. 질량·원소·엔탈피는 출력 보정이 강제하므로 ANN이 보존법칙이나 올바른 반응 경로를 학습했다는 뜻은 아니다. Entropy 비감소는 보장하지 않는다.

| gamma | method | split | mass_sum_max | elemental_max |
|---|---|---|---|---|
| 0.2 | single_best | val | 1.199e-14 | 1.2031e-14 |
| 0.2 | single_best | test | 4.7851e-14 | 4.473e-14 |
| 0.2 | mean10 | val | 1.3323e-15 | 1.1203e-15 |
| 0.2 | mean10 | test | 3.4972e-14 | 3.4002e-14 |
| 0.2 | medoid10 | val | 6.6613e-15 | 5.7991e-15 |
| 0.2 | medoid10 | test | 1.5765e-14 | 1.7143e-14 |
| 0.5 | single_best | val | 4.885e-15 | 6.269e-15 |
| 0.5 | single_best | test | 8.2157e-15 | 8.0299e-15 |
| 0.5 | mean10 | val | 1.7764e-15 | 1.5583e-15 |
| 0.5 | mean10 | test | 1.5543e-15 | 1.4472e-15 |
| 0.5 | medoid10 | val | 1.3323e-15 | 1.193e-15 |
| 0.5 | medoid10 | test | 1.7764e-15 | 1.5834e-15 |

| gamma | method | split | horizon | proposals | nonfinite | negative | inconsistent | thermal_invalid | accepted |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | single_best | val | 0.01 | 90000 | 0 | 0 | 0 | 0 | 90000 |
| 0.2 | single_best | test | 0.01 | 90000 | 0 | 0 | 0 | 0 | 90000 |
| 0.2 | mean10 | val | 0.01 | 990000 | 0 | 0 | 0 | 0 | 990000 |
| 0.2 | mean10 | test | 0.01 | 990000 | 0 | 0 | 0 | 0 | 990000 |
| 0.2 | medoid10 | val | 0.01 | 900000 | 0 | 0 | 0 | 0 | 900000 |
| 0.2 | medoid10 | test | 0.01 | 900000 | 0 | 0 | 0 | 0 | 900000 |
| 0.5 | single_best | val | 0.01 | 85900 | 0 | 0 | 0 | 1 | 85899 |
| 0.5 | single_best | test | 0.01 | 90000 | 0 | 0 | 0 | 0 | 90000 |
| 0.5 | mean10 | val | 0.01 | 943096 | 0 | 0 | 0 | 10 | 943086 |
| 0.5 | mean10 | test | 0.01 | 990000 | 0 | 0 | 0 | 0 | 990000 |
| 0.5 | medoid10 | val | 0.01 | 851560 | 0 | 50 | 0 | 17 | 851493 |
| 0.5 | medoid10 | test | 0.01 | 900000 | 0 | 0 | 0 | 0 | 900000 |

후보 통계는 rollout만 포함하며 mean 방식의 마지막 보정 호출도 포함한다. 실패 원인은 `negative`, `inconsistent`, `thermal_invalid` 등을 함께 해석한다.

훈련·a priori 선택·전체 평가 총 실행시간: **543.1초**. 훈련/rollout wall-time을 기록했고 GPU/HPC는 사용하지 않았다.

| gamma | method | split | seconds |
|---|---|---|---|
| 0.2 | single_best | val | 9.7039 |
| 0.2 | single_best | test | 10.215 |
| 0.2 | mean10 | val | 17.803 |
| 0.2 | mean10 | test | 18.096 |
| 0.2 | medoid10 | val | 15.802 |
| 0.2 | medoid10 | test | 15.471 |
| 0.5 | single_best | val | 8.8871 |
| 0.5 | single_best | test | 9.1489 |
| 0.5 | mean10 | val | 17.572 |
| 0.5 | mean10 | test | 17.958 |
| 0.5 | medoid10 | val | 15.739 |
| 0.5 | medoid10 | test | 15.891 |

모든 방법/분할 12개에서 모델 재로딩 예측, 첫 self-fed step 일치, singleton seed 고정을 확인했다. Ranking의 all-member 경로와 singleton 추론도 일치 검증했다. 입력 데이터 hash와 train parent/next-state identity를 확인했다. 전체 pytest: **77 passed**, 실패/오류/skip 0개. [실행 로그](results/preprocess_tanh10/pytest.log), [검증 기록](results/preprocess_tanh10/verification.json). Keras/NumPy의 copy 키워드 deprecation 경고는 남아 있다.

## 그림과 재현

- γ=0.2: [전체 test 온도](results/preprocess_tanh10/figures/g0.2_temperature.png), [IC44의 9종](results/preprocess_tanh10/figures/g0.2_species_ic44.png)

- γ=0.5: [전체 test 온도](results/preprocess_tanh10/figures/g0.5_temperature.png), [IC44의 9종](results/preprocess_tanh10/figures/g0.5_species_ic44.png)

그림의 로그 시간축은 t=0 표시만 제외한다. 오차 계산은 t=0부터 전체 구간을 사용한다. 종 그림은 symlog를 사용하며 원자료를 floor/clipping하지 않았다.

- [전체 요약](results/preprocess_tanh10/summary.csv), [IC별 결과](results/preprocess_tanh10/trajectories.csv), [종별 오차](results/preprocess_tanh10/species_profiles.csv)
- [Full-pair a priori](results/preprocess_tanh10/one_step.csv), [고정 선택](results/preprocess_tanh10/all_selections_frozen.json), [검증](results/preprocess_tanh10/reload_checks.csv)

```bash
/workspace/kinetics-venv/bin/python -m kinetics.preprocess_study --config configs/preprocess.json
/workspace/kinetics-venv/bin/python -m pytest -q --junitxml=results/preprocess_tanh10/pytest.xml > results/preprocess_tanh10/pytest.log
/workspace/kinetics-venv/bin/python -m kinetics.preprocess_verify
/workspace/kinetics-venv/bin/python -m kinetics.preprocess_report
```

반복 실험은 config를 복사하여 새 output 경로를 사용한다. 기존 output은 기본 거부하며 `--resume`는 config·입력·모델 hash와 고정 선택을 검증한 뒤 이어간다. 데이터/seed를 늘리거나 budget을 늘리지 않은 한 번의 matched ablation이다. 10-member ensemble의 훈련 예산은 single의 10배이며 독립 ensemble 반복에 대한 통계적 유의성을 주장하지 않는다.
