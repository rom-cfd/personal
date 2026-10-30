# Zhang multi-scale sampling + full-state NN, physics OFF

첨부된 Zhang et al., Combustion and Flame 245 (2022) 112319의 §2.3과 §2.5를 직접 읽고 구현한 **축소 비교 실험**이다. 기존 데이터 perturbation을 중단하고 독립적인 multi-scale log sampling으로 바꿨다. NN은 T/P/BCT(Y)의 변화량을 출력하고 온도·압력까지 raw 출력으로 self-feed한다. 원소 보정, 엔탈피 복원, decoder clipping, 질량합 재정규화는 모두 없다.

## 최종 해석 시간과 핵심 결과

목표 해석 시간은 **10 ms와 20 ms**, dt=2 μs이다. 20 ms까지 한 번 전진하고 10 ms 시점도 평가했다. 도중 중단되면 실제 도달 시간을 표시하며, 짧은 구간만 맞춘 결과를 10/20-ms 성공으로 세지 않는다. 정상 숫자로 끝난 완주와 물리적으로 허용되는 완주를 구분한다. **Test의 물리 허용 완주는 모든 방법에서 0/18**이다. Multi-scale sampling이 수치 완주율을 높여도 질량합·원소 보존을 만족하는 장시간 kinetics 예측으로 이어지지는 않았다.

- multiscale, γ=0.1, single_best: 20-ms 수치 완주 1/18, 물리 허용 완주 0/18, 실제 도달시간 중앙값 0.217 ms.
- multiscale, γ=0.1, mean10: 20-ms 수치 완주 16/18, 물리 허용 완주 0/18, 실제 도달시간 중앙값 20 ms.
- multiscale, γ=0.1, medoid10: 20-ms 수치 완주 16/18, 물리 허용 완주 0/18, 실제 도달시간 중앙값 20 ms.
- multiscale, γ=0.2, single_best: 20-ms 수치 완주 13/18, 물리 허용 완주 0/18, 실제 도달시간 중앙값 20 ms.
- multiscale, γ=0.2, mean10: 20-ms 수치 완주 13/18, 물리 허용 완주 0/18, 실제 도달시간 중앙값 20 ms.
- multiscale, γ=0.2, medoid10: 20-ms 수치 완주 7/18, 물리 허용 완주 0/18, 실제 도달시간 중앙값 0.614 ms.
- multiscale, γ=0.5, single_best: 20-ms 수치 완주 1/18, 물리 허용 완주 0/18, 실제 도달시간 중앙값 0 ms.
- multiscale, γ=0.5, mean10: 20-ms 수치 완주 12/18, 물리 허용 완주 0/18, 실제 도달시간 중앙값 20 ms.
- multiscale, γ=0.5, medoid10: 20-ms 수치 완주 9/18, 물리 허용 완주 0/18, 실제 도달시간 중앙값 11.291 ms.
- manifold_control, γ=0.1, single_best: 20-ms 수치 완주 0/18, 물리 허용 완주 0/18, 실제 도달시간 중앙값 0.385 ms.
- manifold_control, γ=0.1, mean10: 20-ms 수치 완주 0/18, 물리 허용 완주 0/18, 실제 도달시간 중앙값 1.347 ms.
- manifold_control, γ=0.1, medoid10: 20-ms 수치 완주 0/18, 물리 허용 완주 0/18, 실제 도달시간 중앙값 0.416 ms.

## 원문과 구현의 대응

| 항목 | 논문 | 이번 구현 |
|---|---|---|
| sampling | §2.5: T 800–3100 K, P .5–2 atm, 종별 독립 log sampling | 같은 규칙; 주 비교는 기존 test를 덮도록 P .5–5 atm로 확장 |
| log scales | N2 .2–.9, H2/O2 1e−5–1, 나머지 하한 1e−k (k=1…25) | 원문 주요 종 분류 유지; H2O도 small-scale 그룹 |
| normalization | §2.5는 simplex 정규화의 구체적 순서 미명시 | sampled N2 보존, 나머지 raw log weights를 1−N2로 비례 정규화한 명시적 가정 |
| main samples | 180,000 × 25 = 4,500,000 | 300 × 25 = 7,500 |
| hot supplement | T 1800–3100 K; 다른 종 1e−4–1; 400,000 초기 상태의 연속 두 step | 667 초기 상태, 연속 두 step 1,334쌍 |
| total pairs | 5,300,000 | 8,834: 기존 8,815쌍과 거의 같은 크기 |
| input/output | §2.3: x=[T,P,F(Y)], u=x_next−x; m0=mL=n+2 | 9종의 11입력·11출력; linear P(atm), literal BCT |
| target distinction | F(Ynext)−F(Ynow) | F(Ynext−Ynow) 또는 signed physical increment가 아님 |
| normalization/loss | 입력·출력 각각 Z-score, L1 | train input/target만으로 별도 적합; standardized increment L1 |
| temperature inference | 본문은 species 예측 뒤 엔탈피 기반 T 복원도 설명 | 사용자 요청으로 제외: T와 P를 포함한 모든 NN 출력을 그대로 사용 |
| architecture | 원문은 GELU, 큰 DNN | 이전 요청의 64–64–64 tanh 유지, 9,803 parameters |
| chemistry | 8종/16 reversible reactions로 기술 | 기존 9종/28반응 유지 |
| step size | 논문에서 사용한 kinetics step과 현재 benchmark 차이 | 이전 비교와 동일하게 dt=2 μs 유지 |
| HRR screening | §2.5에 HRR 비교식/제외 임계값 미명시 | 임의 HRR 필터를 추가하지 않음; HRR은 기록/진단만 수행 |

따라서 데이터 규모·반응기구·압력 범위·정규화 가정·작은 tanh 구조·physics-off 차이가 있는 방법론 비교이며, 논문의 전체 수치 결과를 재현했다는 주장이 아니다. log 범위는 정규화 전 raw weights에 적용되므로 최종 비질소 질량분율은 비례 재분배로 달라질 수 있다. N2 범위는 그대로 유지한다.

## 압력 범위와 데이터 생성 검증

논문 그대로 .5–2 atm인 데이터도 `results/zhang_multiscale_data`에 별도로 생성·보존했다. 기존 test 18개 중 16개가 2 atm보다 높으므로 주 실험은 .5–5 atm에서 새로 생성한 `results/zhang_multiscale_matched_data`를 사용했다. 좁은 범위 데이터의 NN 결과라고 혼동하면 안 된다.

독립 초기 상태 8167개, 학습 쌍 8834개, 제외 초기 상태 0개. 정상 Cantera 해이면서 초기 HRR<0인 쌍 2783개는 유지했다. 라벨 원소 잔차 최대 1.41e-12. 유한성, Cantera 음수 roundoff 한계, 기존 모델 온도 지원 범위 200–6000 K만 라벨 적합성에 사용했다. 이번에는 이 조건으로 실제 제외된 표본이 없다.

독립 sampling 상태에는 기존 manifold의 시간 좌표가 없다. 각 라벨의 예측 간격은 2 μs이고, 고온 보충은 연속된 두 2-μs 쌍이다. 이것이 곧 10/20-ms 후기 평형 분포를 충분히 포함했다는 뜻은 아니므로 최종 시간까지의 self-fed 평가와 후반 오차를 별도로 수행한다.

## 학습과 고정된 단일 모델 선택

γ=.1/.2/.5 multi-scale 각각 10개, 그리고 γ=.1의 기존 manifold 데이터 raw-I/O 대조군 10개를 새로 훈련했다. 총 40개, 모델당 10,000 Adam update, batch512, lr .001→.0003→.0001, 멤버별 norm clip5. 기존 matched-budget 학습 설정을 유지하고 loss만 원문의 L1으로 바꿨다. 같은 raw 입출력·loss·구조·seed·update의 γ=.1 대조군에서 sampling 효과를 분리한다. 기존 physics-on 결과와의 차이를 sampling 효과만으로 해석하지 않는다.

각 그룹의 10개를 전체 validation 90,000 one-step 쌍에서 기존 물리 단위 selection_score로 정렬했다. 동률은 seed 순서다. 모든 선택을 저장한 뒤 test/rollout 평가를 시작했다. 모든 후보가 무효라 score가 정의되지 않으면 infinity로 기록하고 최저 seed는 실패 진단용으로만 유지한다. 이 경우 usable model이라고 부르지 않는다.

| regime | gamma | selected_seed | validation_score | usable_validation_model |
|---|---|---|---|---|
| multiscale | 0.1 | 11 | 0.26786 | True |
| multiscale | 0.2 | 110 | 6.0442 | True |
| multiscale | 0.5 | 22 | 14.954 | True |
| manifold_control | 0.1 | 55 | 0.011673 | True |

Mean10/medoid10은 유한한 raw 후보만 사용한다. 온도/질량/원소를 위반한 유한 후보도 제외하거나 고치지 않는다. Mean10은 물리 상태 공간 평균, medoid10은 기존 train-scaled T/Y 거리로 실제 후보 하나를 선택하며 그 후보의 P도 함께 사용한다. 최소 physics-correction 선택은 correction 자체가 없어 이번에는 정의하지 않는다.

## 10 ms / 20 ms a posteriori 전체 결과

완주 오류는 전체 horizon을 끝낸 궤적에서만 정의한다. 전부 중단되면 RMSE는 — 이며 0이 아니다. 점화 분류/지연 수치는 raw 온도 crossing 진단이다. 물리 허용 완주가 0이면 정상 kinetics 예측으로 해석할 수 없다. 지연은 비교 일관성을 위해 기존 T0+400 K crossing을 유지했다. Zhang의 최대 HRR 시점 정의와 구분한다. 정밀 reference는 0.2 μs로 계산했고 NN step은 2 μs로 유지했다.

### val, 10 ms

| regime | gamma | method | completed | physical_completed | coverage_% | median_end_ms | ignition_% | ignition_pairs | temperature_profile_mean | species_profile_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| multiscale | 0.1 | single_best | 2 | 0 | 13.251 | 0.205 | 83.333 | 15 | 820.27 | 0.053898 |
| multiscale | 0.1 | mean10 | 17 | 0 | 94.488 | 10 | 88.889 | 15 | 1540.3 | 0.14233 |
| multiscale | 0.1 | medoid10 | 17 | 0 | 94.492 | 10 | 94.444 | 16 | 1077.9 | 0.081974 |
| multiscale | 0.2 | single_best | 10 | 0 | 56.229 | 10 | 66.667 | 12 | 976.66 | 0.072742 |
| multiscale | 0.2 | mean10 | 12 | 0 | 67.083 | 10 | 66.667 | 12 | 963.97 | 0.057126 |
| multiscale | 0.2 | medoid10 | 7 | 0 | 41.019 | 0.79 | 66.667 | 12 | 536.84 | 0.035549 |
| multiscale | 0.5 | single_best | 1 | 0 | 6.82 | 0.004 | 27.778 | 5 | 710.82 | 0.023216 |
| multiscale | 0.5 | mean10 | 12 | 0 | 66.667 | 10 | 66.667 | 12 | 786.92 | 0.050181 |
| multiscale | 0.5 | medoid10 | 10 | 0 | 56.354 | 10 | 66.667 | 12 | 520.95 | 0.03517 |
| manifold_control | 0.1 | single_best | 0 | 0 | 4.43 | 0.433 | 88.889 | 16 | — | — |
| manifold_control | 0.1 | mean10 | 0 | 0 | 12.224 | 1.265 | 88.889 | 16 | — | — |
| manifold_control | 0.1 | medoid10 | 0 | 0 | 10.116 | 0.91 | 88.889 | 16 | — | — |

### val, 20 ms

| regime | gamma | method | completed | physical_completed | coverage_% | median_end_ms | ignition_% | ignition_pairs | temperature_profile_mean | species_profile_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| multiscale | 0.1 | single_best | 2 | 0 | 12.181 | 0.205 | 88.889 | 16 | 825.98 | 0.054334 |
| multiscale | 0.1 | mean10 | 17 | 0 | 94.466 | 20 | 83.333 | 15 | 1578.4 | 0.14322 |
| multiscale | 0.1 | medoid10 | 17 | 0 | 94.468 | 20 | 88.889 | 16 | 1112.1 | 0.088104 |
| multiscale | 0.2 | single_best | 10 | 0 | 55.892 | 20 | 66.667 | 12 | 978.12 | 0.072797 |
| multiscale | 0.2 | mean10 | 12 | 0 | 66.875 | 20 | 72.222 | 13 | 974.55 | 0.057413 |
| multiscale | 0.2 | medoid10 | 7 | 0 | 39.954 | 0.79 | 72.222 | 13 | 536.97 | 0.035233 |
| multiscale | 0.5 | single_best | 1 | 0 | 6.1878 | 0.004 | 27.778 | 5 | 716.67 | 0.02353 |
| multiscale | 0.5 | mean10 | 12 | 0 | 66.667 | 20 | 66.667 | 12 | 794.5 | 0.050515 |
| multiscale | 0.5 | medoid10 | 10 | 0 | 55.955 | 20 | 66.667 | 12 | 521.96 | 0.035304 |
| manifold_control | 0.1 | single_best | 0 | 0 | 2.215 | 0.433 | 94.444 | 17 | — | — |
| manifold_control | 0.1 | mean10 | 0 | 0 | 6.1122 | 1.265 | 94.444 | 17 | — | — |
| manifold_control | 0.1 | medoid10 | 0 | 0 | 5.0578 | 0.91 | 94.444 | 17 | — | — |

### test, 10 ms

| regime | gamma | method | completed | physical_completed | coverage_% | median_end_ms | ignition_% | ignition_pairs | temperature_profile_mean | species_profile_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| multiscale | 0.1 | single_best | 1 | 0 | 7.5189 | 0.217 | 77.778 | 14 | 886.4 | 0.054778 |
| multiscale | 0.1 | mean10 | 16 | 0 | 88.972 | 10 | 88.889 | 15 | 1522.3 | 0.13757 |
| multiscale | 0.1 | medoid10 | 17 | 0 | 94.53 | 10 | 66.667 | 12 | 1231.9 | 0.094397 |
| multiscale | 0.2 | single_best | 13 | 0 | 72.237 | 10 | 77.778 | 14 | 1098.7 | 0.071353 |
| multiscale | 0.2 | mean10 | 13 | 0 | 72.838 | 10 | 77.778 | 14 | 1288.2 | 0.065946 |
| multiscale | 0.2 | medoid10 | 7 | 0 | 40.876 | 0.614 | 83.333 | 15 | 772.05 | 0.047174 |
| multiscale | 0.5 | single_best | 1 | 0 | 6.6644 | 0 | 33.333 | 6 | 441.9 | 0.041537 |
| multiscale | 0.5 | mean10 | 12 | 0 | 66.667 | 10 | 66.667 | 12 | 1228.8 | 0.063746 |
| multiscale | 0.5 | medoid10 | 9 | 0 | 52.271 | 6.291 | 66.667 | 12 | 773.96 | 0.054408 |
| manifold_control | 0.1 | single_best | 0 | 0 | 4.3678 | 0.385 | 83.333 | 15 | — | — |
| manifold_control | 0.1 | mean10 | 0 | 0 | 13.416 | 1.347 | 83.333 | 15 | — | — |
| manifold_control | 0.1 | medoid10 | 0 | 0 | 6.06 | 0.416 | 83.333 | 15 | — | — |

### test, 20 ms

| regime | gamma | method | completed | physical_completed | coverage_% | median_end_ms | ignition_% | ignition_pairs | temperature_profile_mean | species_profile_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| multiscale | 0.1 | single_best | 1 | 0 | 6.5372 | 0.217 | 77.778 | 14 | 886.8 | 0.054725 |
| multiscale | 0.1 | mean10 | 16 | 0 | 88.931 | 20 | 88.889 | 15 | 1542.2 | 0.13829 |
| multiscale | 0.1 | medoid10 | 16 | 0 | 93.154 | 20 | 66.667 | 12 | 1216.4 | 0.095182 |
| multiscale | 0.2 | single_best | 13 | 0 | 72.229 | 20 | 77.778 | 14 | 1055 | 0.070951 |
| multiscale | 0.2 | mean10 | 13 | 0 | 72.53 | 20 | 77.778 | 14 | 1304.4 | 0.065971 |
| multiscale | 0.2 | medoid10 | 7 | 0 | 39.882 | 0.614 | 83.333 | 15 | 787.47 | 0.046626 |
| multiscale | 0.5 | single_best | 1 | 0 | 6.11 | 0 | 33.333 | 6 | 443.48 | 0.041709 |
| multiscale | 0.5 | mean10 | 12 | 0 | 66.667 | 20 | 66.667 | 12 | 1191.7 | 0.063692 |
| multiscale | 0.5 | medoid10 | 9 | 0 | 51.136 | 11.291 | 66.667 | 12 | 744.75 | 0.055303 |
| manifold_control | 0.1 | single_best | 0 | 0 | 2.1839 | 0.385 | 83.333 | 15 | — | — |
| manifold_control | 0.1 | mean10 | 0 | 0 | 6.7078 | 1.347 | 83.333 | 15 | — | — |
| manifold_control | 0.1 | medoid10 | 0 | 0 | 3.03 | 0.416 | 83.333 | 15 | — | — |

## 구간별 평가: 초기와 후기

0–2 ms, 2–10 ms, 10–20 ms 각각을 끝까지 도달한 경우에만 해당 구간 RMSE를 계산한다. 한 구간도 도달하지 못한 모델은 후기 오차가 작다고 평가하지 않는다.

| regime | gamma | method | early_completed | early_T_RMSE | late_completed | late_T_RMSE | tail_completed | tail_T_RMSE |
|---|---|---|---|---|---|---|---|---|
| manifold_control | 0.1 | mean10 | 0 | — | 0 | — | 0 | — |
| manifold_control | 0.1 | medoid10 | 0 | — | 0 | — | 0 | — |
| manifold_control | 0.1 | single_best | 0 | — | 0 | — | 0 | — |
| multiscale | 0.1 | mean10 | 16 | 1364.4 | 16 | 1551.4 | 16 | 1560.8 |
| multiscale | 0.1 | medoid10 | 17 | 1088.4 | 17 | 1249.1 | 16 | 1227 |
| multiscale | 0.1 | single_best | 1 | 852.93 | 1 | 894.57 | 1 | 887.21 |
| multiscale | 0.2 | mean10 | 13 | 1156.6 | 13 | 1313.2 | 13 | 1319.7 |
| multiscale | 0.2 | medoid10 | 7 | 651.68 | 7 | 796.47 | 7 | 802.28 |
| multiscale | 0.2 | single_best | 13 | 1238 | 13 | 1051.4 | 13 | 1001.1 |
| multiscale | 0.5 | mean10 | 12 | 1318.9 | 12 | 1197.8 | 12 | 1148.5 |
| multiscale | 0.5 | medoid10 | 10 | 913.32 | 9 | 740.07 | 9 | 685.46 |
| multiscale | 0.5 | single_best | 1 | 429.03 | 1 | 445.05 | 1 | 445.06 |

## A priori test: 전체 90,000쌍

| regime | gamma | method | samples | selection_score | rmse | temperature_rmse | invalid_row_fraction | mass_sum_mae |
|---|---|---|---|---|---|---|---|---|
| multiscale | 0.1 | single_best | 90000 | 0.32957 | 0.0040511 | 37.307 | 1.1111e-05 | 0.0082289 |
| multiscale | 0.1 | mean10 | 90000 | 0.30381 | 0.0042957 | 42.236 | 0 | 0.0072224 |
| multiscale | 0.1 | medoid10 | 90000 | 0.30456 | 0.0043206 | 42.095 | 0 | 0.0062239 |
| multiscale | 0.2 | single_best | 90000 | 6.6072 | 0.0029605 | 18.265 | 0.060256 | 0.0083518 |
| multiscale | 0.2 | mean10 | 90000 | 7.4014 | 0.0022095 | 21.796 | 0.055578 | 0.0057759 |
| multiscale | 0.2 | medoid10 | 90000 | 7.4206 | 0.0023821 | 24.474 | 0.055578 | 0.0056404 |
| multiscale | 0.5 | single_best | 90000 | 23.183 | 0.002948 | 23.865 | 0.2045 | 0.005144 |
| multiscale | 0.5 | mean10 | 90000 | 22.842 | 0.0019443 | 25.553 | 0.17403 | 0.0031799 |
| multiscale | 0.5 | medoid10 | 90000 | 22.733 | 0.0025211 | 26.008 | 0.17403 | 0.0046411 |
| manifold_control | 0.1 | single_best | 90000 | 0.011232 | 0.00046638 | 4.3353 | 0 | 0.00027925 |
| manifold_control | 0.1 | mean10 | 90000 | 0.010683 | 0.00046432 | 4.2583 | 0 | 0.00026372 |
| manifold_control | 0.1 | medoid10 | 90000 | 0.010995 | 0.00046769 | 4.2676 | 0 | 0.00024499 |

## 실패와 물리 진단

BCT inverse의 수학적 정의역 밖 출력은 NaN으로 기록하고 중단한다. signed extension이나 guard clipping을 넣지 않았다. 숫자가 유한해도 |T|>1e6 K, |Y|>1e3, |P|>1e15 Pa이면 수치 폭주로 중단한다. 200–6000 K 온도 범위, Y∈[−1e−12,1+1e−12], mass/element drift≤.01, pressure drift≤1%는 추론을 바꾸지 않는 물리 허용성 진단에만 사용한다.

| regime | gamma | method | status | trajectories |
|---|---|---|---|---|
| manifold_control | 0.1 | mean10 | numeric_magnitude_guard | 18 |
| manifold_control | 0.1 | medoid10 | numeric_magnitude_guard | 18 |
| manifold_control | 0.1 | single_best | numeric_magnitude_guard | 15 |
| manifold_control | 0.1 | single_best | undefined_BCT_inverse_or_nonfinite | 3 |
| multiscale | 0.1 | mean10 | completed | 16 |
| multiscale | 0.1 | mean10 | undefined_BCT_inverse_or_nonfinite | 2 |
| multiscale | 0.1 | medoid10 | completed | 16 |
| multiscale | 0.1 | medoid10 | undefined_BCT_inverse_or_nonfinite | 2 |
| multiscale | 0.1 | single_best | completed | 1 |
| multiscale | 0.1 | single_best | undefined_BCT_inverse_or_nonfinite | 17 |
| multiscale | 0.2 | mean10 | completed | 13 |
| multiscale | 0.2 | mean10 | undefined_BCT_inverse_or_nonfinite | 5 |
| multiscale | 0.2 | medoid10 | completed | 7 |
| multiscale | 0.2 | medoid10 | numeric_magnitude_guard | 2 |
| multiscale | 0.2 | medoid10 | undefined_BCT_inverse_or_nonfinite | 9 |
| multiscale | 0.2 | single_best | completed | 13 |
| multiscale | 0.2 | single_best | undefined_BCT_inverse_or_nonfinite | 5 |
| multiscale | 0.5 | mean10 | completed | 12 |
| multiscale | 0.5 | mean10 | undefined_BCT_inverse_or_nonfinite | 6 |
| multiscale | 0.5 | medoid10 | completed | 9 |
| multiscale | 0.5 | medoid10 | undefined_BCT_inverse_or_nonfinite | 9 |
| multiscale | 0.5 | single_best | completed | 1 |
| multiscale | 0.5 | single_best | undefined_BCT_inverse_or_nonfinite | 17 |

| regime | gamma | method | minimum_Y | maximum_Y | mass_drift | element_drift | pressure_drift |
|---|---|---|---|---|---|---|---|
| manifold_control | 0.1 | mean10 | 0 | 998.4 | 1526.6 | 1274.8 | 4.2127e-16 |
| manifold_control | 0.1 | medoid10 | 0 | 998.99 | 2034.3 | 1847.9 | 0 |
| manifold_control | 0.1 | single_best | 0 | 996.41 | 1915.6 | 1732.2 | 0 |
| multiscale | 0.1 | mean10 | 0 | 0.7554 | 0.40189 | 0.46301 | 4.2127e-16 |
| multiscale | 0.1 | medoid10 | 0 | 1.5726 | 1.2984 | 1.5177 | 0 |
| multiscale | 0.1 | single_best | 0 | 0.90782 | 0.70883 | 0.72509 | 0 |
| multiscale | 0.2 | mean10 | 0 | 947.52 | 949.13 | 841.68 | 3.6937e-16 |
| multiscale | 0.2 | medoid10 | 0 | 932.14 | 933.62 | 828.23 | 0 |
| multiscale | 0.2 | single_best | 0 | 0.7554 | 0.26671 | 0.22661 | 0 |
| multiscale | 0.5 | mean10 | 0 | 0.7554 | 0.086399 | 0.078472 | 3.6937e-16 |
| multiscale | 0.5 | medoid10 | 0 | 0.7554 | 0.14986 | 0.15516 | 0 |
| multiscale | 0.5 | single_best | 0 | 0.7554 | 0.11931 | 0.11188 | 0 |

새 학습·선택·10/20-ms 평가 실행시간: **841.2초**. 기존 dataset/weights/results는 변경하지 않았다. 각 모델/분할의 직렬화 재로딩과 첫 self-fed step 일치를 확인했다. 별도 검증 기록과 pytest 결과는 결과 디렉터리의 `verification.json` / `pytest.log`에 저장한다.

## 그림과 재현

![Training phase coverage](results/zhang_multiscale_raw/figures/training_phase_coverage.png)

![Sampling and HRR](results/zhang_multiscale_raw/figures/sampling_and_HRR.png)

### multiscale_g0.1: 18 test profiles, 20 ms

![multiscale_g0.1 temperature profiles](results/zhang_multiscale_raw/figures/multiscale_g0.1_temperature20ms.png)

### multiscale_g0.2: 18 test profiles, 20 ms

![multiscale_g0.2 temperature profiles](results/zhang_multiscale_raw/figures/multiscale_g0.2_temperature20ms.png)

### multiscale_g0.5: 18 test profiles, 20 ms

![multiscale_g0.5 temperature profiles](results/zhang_multiscale_raw/figures/multiscale_g0.5_temperature20ms.png)

### manifold_control_g0.1: 18 test profiles, 20 ms

![manifold_control_g0.1 temperature profiles](results/zhang_multiscale_raw/figures/manifold_control_g0.1_temperature20ms.png)

그림의 시간축은 로그이며 t=0만 표시에서 제외한다. 온도 표시 범위는 200–6000 K이고 raw 수치는 바꾸지 않았다. 곡선 끝의 ×와 패널 안의 시간은 실제 도달한 마지막 유한 상태다. 첫 step에서 실패하면 NN 곡선이 없고 0 ms로 표시한다.

- [IC별 원자료](results/zhang_multiscale_raw/trajectories.csv), [coverage 포함 요약](results/zhang_multiscale_raw/summary_with_coverage.csv)
- [입출력·physics-off 프로토콜](results/zhang_multiscale_raw/protocol.json), [고정 선택](results/zhang_multiscale_raw/all_selections_frozen.json)

```bash
python -m kinetics.multiscale_data --config configs/zhang_multiscale_matched_data.json
python -m kinetics.zhang_raw_study --config configs/zhang_raw_study.json
python -m kinetics.zhang_raw_report
```

재실행은 새 output 경로로 설정을 복사한다. `--resume`은 config·입력·등록 모델 hash·고정 singleton 선택을 확인한 뒤 진행한다. 완료된 실험을 더 실행하려면 새로운 output을 사용한다. 첨부 PDF는 분석에만 사용했고 저작권 있는 원문 파일을 저장소에 게시하지 않았다.
