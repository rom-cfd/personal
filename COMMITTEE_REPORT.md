# Matched-budget original: 10개 DNN의 평균과 medoid 선택 비교

## 결과

동일한 64–64–64 tanh DNN 10개, Box–Cox γ=0.1, 원본 train 데이터만 사용했다. **증강 데이터는 사용하지 않았다.**
기존 3개 모델(seed 11/22/33)을 재사용하고 seed 44/55/66/77/88/99/110을 10,000 update씩 추가 학습했다.
전체 위원회가 대표하는 훈련량은 100,000 update, 이번에 새로 실행한 훈련량은 70,000 update다. per-member budget이 같다는 뜻이며 ensemble 전체 비용이 single과 같다는 뜻이 아니다.

**이번 held-out 비교에서는 합의 중심 후보 선택이 10개 평균보다 좋지 않았다.** 같은 보정 후보 집합을 사용하는 두 방식의 비교는:

- medoid: 점화 구분 **88.9%**, τ400 중앙 상대오차 **9.1%**, T profile RMSE의 초기조건 평균 **218.7 K**.
- 평균: 점화 구분 **94.4%**, τ400 중앙 상대오차 **5.5%**, T profile RMSE 평균 **130.0 K**.
- 두 방식 모두 18개 초기조건을 **2 ms까지 물리 허용 완주 100%**였다. 보존성은 공통 보정으로 강제했으며 정확한 chemistry를 뜻하지 않는다.

medoid는 validation에서 더 좋았지만 test에서는 우세가 유지되지 않았다. test 결과를 보고 거리 설정이나 seed를 바꾸지 않았다.
초기조건 18개인 작은 비교이므로 이 결과가 모든 반응계에서 평균이 우수하다는 증거는 아니다. 다만 **대표 하나를 고르면 평균보다 강건할 것이라는 가설은 이번 비교에서 지지되지 않았다.**

단, 중요한 장점도 관측됐다. 앞서 어려웠던 긴 점화지연 **IC 44**에서 medoid는 reference **1.213 ms**에 대해 **1.177 ms**를 예측하여 오차가 **3.0%**였다. 비교한 평균 방법들은 이 조건의 점화를 2 ms 안에 놓쳤다.
반면 medoid는 **IC [5, 64]**에서 reference가 2 ms 이내 점화하지 않는데 점화를 예측했다. 즉 medoid는 실제 점화 13개를 모두 잡았지만 가짜 점화 2개가 있었고, 후보별 보정 후 평균은 실제 점화 1개를 놓쳤지만 가짜 점화는 없었다.
따라서 지연 중앙오차 9.1% 대 5.5%는 각각 **13쌍 대 12쌍**에서 계산한 조건부 지표다. 이 숫자만으로 긴 induction 구간까지 평균이 낫다고 해석하면 안 된다.

## 외부 선행연구 질의용 요약

[복사해서 전달할 상세 질의문](LITERATURE_RESEARCH_BRIEF.md)에 배경·실패 양상·전략·문헌 질문을 정리했다.
요지는 stiff chemistry의 fixed-step DNN flow map을 구성하고, 매 시점 같은 현재 상태에서 10개의 다음 상태를 예측한 다음 **실제 후보 중 가장 중심적인 하나를 선택**하는 것이다.
반응률 모델이 아니며, 연구의 초점은 one-step transformed loss보다 a posteriori 점화·전체 profile·보존성과 합의의 신뢰성이다.
현재 Python/Cantera prototype에서 1000배 가속을 확인한 것은 아니므로 문헌 질의문에도 그 구분을 명시했다.

## 데이터와 학습 통제

- 9 species: H2/H/O/O2/OH/H2O/HO2/H2O2/N2, Cantera h2o2에서 Ar 제외, 28개 반응.
- 단열 정압 0-D; 초기 T 900–1600 K, P 1–5 atm, 당량비 .5–2.
- 입력 [T,Y9,P], 다음 상태 [T_next,Y9_next]. 훈련·검증·test·rollout 모두 **dt=2e-6 s**.
- 원본 train 54개 초기조건/10,800 pair, validation 18개, test 18개. 훈련 궤적 0.4 ms, 평가는 2 ms로 길게 확장했다.
- 원본 dataset와 split, 기존 모델·기존 보고서를 덮어쓰지 않았다. source SHA256 `56457bf445a4970ca15361eb35f81cd218c01f76934a87652b8cba3b2a0ad7fe`.
- 모든 멤버는 동일 architecture·전처리·훈련 데이터·업데이트 수를 사용하며 초기화와 minibatch seed만 다르다. Adam/batch 512/gradient clip 5/동일 LR schedule을 기존 matched-budget original과 유지했다.
- TensorFlow 그래프에서 멤버 forward를 함께 호출하되, 서로 다른 모델의 가중치를 평균하거나 공유하도록 바꾸지는 않았다. compiled 결과와 개별 DNN의 물리 단위 출력을 수치 테스트했다.

## 알고리즘

1. 모든 멤버에 **동일한 현재 상태**를 입력한다. 열 개의 다른 궤적 상태를 투표시키는 방식이 아니다.
2. Box–Cox inverse 전에 질량분율 [0,1]에 해당하는 좌표 guard를 적용한다.
3. 각 후보의 Y를 양수·현재 H/O/N inventory에 투영하고, 현재 엔탈피/정압 조건의 HPY로 T를 복원한다. chemistry ODE는 풀지 않는다.
4. 물리/열역학적으로 유효하지 않은 후보는 제외한다. 전부 무효면 적분 실패다.
5. 유효한 후보들의 좌표 거리 합을 최소화하는 실제 후보를 선택한다:

   `i* = argmin_i sum_j sqrt(mean_k((z_i,k - z_j,k)^2))`.

   제곱거리 합이나 성분별 중앙값이 아니다. 선택된 T와 Y는 하나의 동일한 후보에 속한다. 동점은 고정된 seed 순서로 결정한다.
6. 선택 결과를 다음 공통 상태로 사용하고 2 μs 후를 반복한다.

기본 실험은 **ODE fallback 없음, adaptive dt 없음, hysteresis 없음, architecture 다양화 없음**이다. 합의도가 낮다는 이유로 정답 상태나 Cantera 화학 적분으로 복귀하지 않는다.

### 거리 정의

물리 좌표 distance는 T를 train std(하한 100 K)로, 종별 Y를 train q95(하한 1e-8)로 나누어 구성했다.
대안은 train q95×1e-3(하한1e-12)의 epsilon을 사용하는 soft-log 종 좌표다. 해당 좌표의 train std로 다시 스케일했다.
**이 soft-log는 후보 간 거리 정의이며, DNN의 Box–Cox 전처리를 바꾸는 것이 아니다.**
scale/epsilon은 train에서만 적합했고 두 거리 중 선택은 validation의 0-D score로만 했다.

| 방법 | 검증 score ↓ | 점화 구분 % | τ400 중앙오차 % | T profile 평균 K |
|---|---|---|---|---|
| Single original, seed33 | 0.9834 | 94.4 | 4.68 | 226.2 |
| Mean3, then repair | 1.0839 | 94.4 | 6.47 | 236.2 |
| Mean10, then repair | 0.2825 | 100.0 | 5.03 | 100.9 |
| Repair10, then mean | 0.2821 | 100.0 | 5.05 | 100.7 |
| Medoid10, scaled distance | 0.2120 | 100.0 | 3.38 | 101.2 |
| Medoid10, soft-log distance | 0.5622 | 94.4 | 4.13 | 190.1 |

선택된 medoid 거리는 **medoid10_scaled**다. `selection_frozen.json`을 저장한 뒤 test를 열었다. 선택되지 않은 soft-log 거리의 test 성능은 이번 실행에서 계산하지 않았다.

## 평균과 물리 보정 순서

비선형 보정과 평균화 순서를 분리했다.

- `mean3/10_before_repair`: decode한 제안 Y들을 먼저 평균한 뒤 한 번 보정한다. 기존 평균 앙상블의 순서다.
- `mean10_after_repair`: 개별 후보 열 개를 보정한 뒤 유효 후보 Y를 평균하고, 평균 Y에 대해 엔탈피 온도를 복원한다.
- `medoid10`: 보정된 같은 후보 열 개에서 실제 후보 하나를 선택한다.

주 비교는 **mean10_after_repair vs medoid10**다. 평균화 순서 차이를 대표 선택의 효과로 오인하지 않기 위해서다.
이미 원소 inventory가 같은 양수 후보들을 평균하면 질량·원소 보존은 유지된다. 평균이 반드시 물리적으로 불가능한 조성이라는 전제는 이 실험에 맞지 않는다. 에너지는 평균 조성으로 다시 복원한다.

## Held-out a posteriori 결과

온도와 9종 질량분율을 매번 자신의 출력으로 갱신했다. Teacher forcing은 없다.
τ400은 T0+400 K 최초 도달 시각이며 reference만 0.2 μs로 정밀 측정했다. ANN/데이터 간격은 계속 2 μs다.
최대 dT/dt 기준 결과도 CSV에 별도로 제공했다. 수 μs 점화에서는 sampled derivative의 시간 해상도를 고려해야 한다.

| 방법 | 완주 % | 물리 허용 % | 점화 구분 % | τ400 중앙오차 %* | T profile 평균 K** | T pooled K** | Y profile 평균** |
|---|---|---|---|---|---|---|---|
| Single original, seed33 | 94.4 | 94.4 | 83.3 | 5.8 | 364.6 | 519.0 | 1.934e-02 |
| Mean3, then repair | 83.3 | 83.3 | 77.8 | 7.5 | 279.7 | 446.3 | 1.259e-02 |
| Mean10, then repair | 88.9 | 88.9 | 83.3 | 5.5 | 119.2 | 313.2 | 5.909e-03 |
| Repair10, then mean | 100.0 | 100.0 | 94.4 | 5.5 | 130.0 | 304.2 | 9.731e-03 |
| Medoid10, scaled distance | 100.0 | 100.0 | 88.9 | 9.1 | 218.7 | 396.2 | 1.224e-02 |

\* τ400 오차는 reference와 ANN이 모두 점화한 쌍에서만 계산한다. 누락·가짜 점화는 구분 정확도와 혼동행렬에 별도 기록했다. 지연 중앙오차가 작아도 missed ignition을 포함하면 우수하지 않을 수 있다.
\** profile은 각 궤적의 시간적분 RMSE를 계산한 뒤 초기조건에 평균한 값이다. pooled는 모든 완주 궤적을 합쳐 squared error를 평균한 뒤 root를 취한다. 미완주 궤적의 전체 profile error는 NaN이며 실패·coverage로 따로 표시했다.
주 비교 두 방법은 모두 완주하므로 `paired_mean_medoid.csv`에는 같은 초기조건 18쌍이 포함된다.
점화 전에 적분이 실패한 경우에는 정상적인 미점화/TN으로 세지 않고 `unresolved_no_ignition`으로 별도 기록했다. 점화 후 실패는 점화 검출과 전체 profile 실패를 구분한다.

## 대표 교체와 합의-오차 관계

medoid test에서 대표 seed는 총 **5179회** 바뀌었고, 연속 스텝 중 교체 비율은 **28.8%**였다.
관측된 test의 최소 유효 후보 수는 10개였다. 따라서 이 실행에서는 medoid가 무효 후보를 피해서 좋아졌다는 설명도 성립하지 않는다.
매번 가장 중심인 후보를 골라도 대표 교체가 시간적으로 매끄러운 동역학을 보장하지는 않는다. 교체 빈도가 오차의 원인이라는 인과적 결론까지 내린 것은 아니다.

validation reference 상태에서 one-step 후보 spread와 scaled species error의 Spearman 상관은 **0.899**였다.
이것은 합의도에 정보가 있다는 관찰이지만, 장기 self-fed 상태나 미관측 조건에서의 오차 calibration은 아니다.
샘플들은 같은 궤적 안에서 강하게 시간 상관이 있으므로 출력된 IID p-value로 유의성을 주장하지 않는다.
높은 one-step 상관과 test의 medoid 열세가 동시에 관측됐다. 합의도가 정보를 주는 것과 가장 중심적인 한 후보가 가장 정확하다는 주장은 서로 다르다.

## 가능한 해석과 한계

평균은 독립적인 오차 성분을 상쇄할 수 있고, medoid는 이상치에 덜 민감하지만 평균화의 분산 감소 일부를 포기한다.
같은 architecture·데이터의 공통 편향은 다수의 합의로 제거되지 않는다. 물리 제약을 통과하는 후보도 점화 타이밍이 틀릴 수 있다.
또한 훈련 데이터는 0.4 ms까지만 관측하여 2 ms의 긴 induction 구간을 충분히 표현하지 못할 수 있다.
이번 결과만으로 architecture 다양화, hysteresis, 불확실성 기반 fallback의 효용을 주장하지 않는다. 모두 후속 통제 실험 대상이다.
새 test 결과로 거리 가중치나 switching 규칙을 재튜닝하지 않았다.

## 실행 비용과 검증

| 방법 | 18개 2ms rollout(s) | DNN+decode(s) | 보정(s) | 거리/선택 진단(s) |
|---|---|---|---|---|
| Single original, seed33 | 2.67 | 0.85 | 1.40 | 0.125 |
| Mean3, then repair | 2.48 | 0.89 | 1.31 | 0.000 |
| Mean10, then repair | 2.88 | 1.19 | 1.38 | 0.000 |
| Repair10, then mean | 15.12 | 1.45 | 13.05 | 0.257 |
| Medoid10, scaled distance | 13.69 | 1.33 | 11.72 | 0.260 |

모든 시간은 이 CPU prototype의 18개 self-fed 2 ms 적분과 진단을 측정한 값이다. 그래프 warm-up을 제외했고 file write는 제외했다. 모델 호출·보정·선택 시간은 따로 계측했다.
후보별 물리 보정이 현재 비용의 큰 부분이다. 10개 forward를 그래프로 묶어 Python 호출 overhead를 줄였지만, 10회 보정 비용은 남는다.
이 표는 Cantera 대비 speedup이 아니다. 큰 메커니즘에서의 1000배 가속을 이 작은 9종 구현에 적용할 수 없다.
전체 실험(추가 학습·검증·test·후처리 전) wall time은 243.2초다. GPU/HPC 작업은 없었다.

테스트: **48 passed, 20 warnings in 4.07s**. 실제 후보 반환·이상치·무효 후보·동점 규칙·compiled forward 일치·self-fed 상태 갱신·교체 횟수를 검사했다.
6개 method descriptor를 reload하여 예측과 선택 seed의 재현을 확인했다. 원본 데이터 checksum도 유지했다.

## 산출물과 재현

- [외부 연구 질의용 요약](LITERATURE_RESEARCH_BRIEF.md)
- [설정](configs/committee.json), [위원회/medoid 구현](kinetics/committee.py), [학습·검증·test runner](kinetics/committee_study.py), [보고서/그림](kinetics/committee_report.py)
- [최종 수치 비교](results/committee_original/summary.csv), [초기조건별 지표](results/committee_original/trajectories.csv), [혼동행렬](results/committee_original/confusion.csv), [동일조건 mean/medoid 비교](results/committee_original/paired_mean_medoid.csv)
- [종별 profile](results/committee_original/species_profiles.csv), [one-step](results/committee_original/one_step.csv), [잡음 강건성](results/committee_original/noise.csv), [시간](results/committee_original/timing.csv)
- [대표 교체 통계](results/committee_original/selection_statistics.csv), [대표 사용 빈도](results/committee_original/representative_usage.csv), [검증 spread/오차](results/committee_original/validation_confidence.csv)
- [전체 온도 profile](results/committee_original/figures/all_temperature_profiles.png), [9종 profile](results/committee_original/figures/species_profiles.png), [점화지연](results/committee_original/figures/ignition_delay.png), [대표 교체 그림](results/committee_original/figures/representative_switches.png), [profile 오차](results/committee_original/figures/profile_errors.png)

`/workspace/kinetics-venv/bin/python -m kinetics.committee_study --config configs/committee.json`으로 실행한다. 기존 output은 거부하므로 재실행할 때는 config의 output을 새 경로로 바꾼다. 이전 matched original 모델 3개와 Cantera reference는 재사용한다.
위원회는 `Committee.load('results/committee_original/methods/medoid10_scaled.json')`로 복원하며 `.predict(x)`에서 현재 [T,Y9,P] 행들을 입력받아 선택된 다음 [T,Y9] 행들을 반환한다.
