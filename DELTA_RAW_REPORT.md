# Signed physical increments, γ=.1: physics 보정 없는 비교

## 주요 결과

- single_best: 수치 완주 100.0%, 물리 유효 완주 5.6%, raw 점화 구분 83.3%, T profile 평균 RMSE 31235.6 K.
- mean100: 수치 완주 100.0%, 물리 유효 완주 5.6%, raw 점화 구분 88.9%, T profile 평균 RMSE 29094.5 K.
- medoid100: 수치 완주 100.0%, 물리 유효 완주 16.7%, raw 점화 구분 83.3%, T profile 평균 RMSE 12006.3 K.

위 raw 점화 수치와 수치 완주만으로 성공이라 판정하지 않는다. 물리 위반과 전체 profile 오차를 함께 확인해야 한다.

## 구현

사용자가 확인한 변환은 **f(v)=sign(v)|v|^.1/.1**, 역변환은 **f⁻¹(z)=sign(z)(.1|z|)^10**이다.
입력은 (T,P,Y), 출력은 **ΔT=T(t+dt)−T(t), ΔY=Y(t+dt)−Y(t)**다.
Species에는 물리 공간에서 먼저 차를 구하고 signed Box–Cox를 적용했다.
다음 상태는 (T,Y)현재 + 예측한 물리 증분이다. ΔT/1000은 선형 표준화했다.
입력 T/1000, log(P), signed Box–Cox(Y)와 출력은 각각 train만으로 표준화했다.
음수 입력 Y의 signed extension은 보정 없이 self-fed 평가하기 위한 수학적 정의이며 음수 종을 물리적으로 정당화하지 않는다.

이전 모델도 이미 **f(Y다음)−f(Y현재)**라는 transformed-state residual을 학습했다.
이번 변화는 residual 연결의 최초 도입이 아니라 **f(Y다음−Y현재)**를 target으로 사용한다는 점이다.
이 두 식은 γ≠1에서 같지 않으며 target의 오차 weighting도 달라진다.

**원소/질량 보존 투영, 엔탈피 온도 복원, clipping, sum normalization, 음수 후보 제외, ODE fallback을 모두 사용하지 않았다.**
온도도 DNN이 직접 예측한 ΔT로 진화한다. 각 finite 음수 후보는 평균·medoid 계산에 그대로 포함한다.
Mean은 inverse transform 후 물리 증분/다음 상태의 평균이고 transformed 출력의 평균이 아니다.
Medoid는 training-scaled T/Y 거리 합이 최소인 실제 raw 후보 하나를 매 step 선택한다.
Single은 validation one-step score로 선택한 **seed 242** 하나를 전 timestep 고정한다.

## 데이터와 학습 예산

기존 10-ms 원본 궤적과 동일 split, downsampled train 8,815쌍을 재사용했다. 모든 쌍·ANN step은 dt=2 μs다.
새 target/정규화이므로 기존 가중치를 재사용하지 않고 **100개 모두 새로 학습**했다.
64–64–64 tanh, 9,738 parameter, batch512, Adam, seed11,22,…,1100, 모델당10,000 update다.
기존과 같은 learning-rate schedule/global gradient clipping을 사용했다. 이는 optimizer 설정이며 출력 state의 clipping이 아니다.
Grouped 학습에서도 초기화·미니배치 RNG·Adam moment·gradient norm clipping은 멤버별 독립이다.
훈련 1181.3초, 선택/평가 216.0초였다. CPU만 사용했다.

100개 모두의 validation 90,000개 one-step 쌍으로 기존 물리 단위 `selection_score`를 계산했다.
relative error floor=1e-8, major threshold=1e-3이고 transformed MSE는 선택에 사용하지 않았다.
물리 score = mean(log1p(capped robust relative Y error)) + 10 major_Y_RMSE + 10 mass_sum_MAE + 100 invalid_row_fraction.
가중치나 score를 test/rollout 결과로 조정하지 않고 선택 seed와 metric을 rollout 전에 고정했다.
Full validation에는 평형 구간도 들어가므로 a priori 순위가 induction/rollout 최적 순위를 뜻하지 않는다.

| seed | selection_score | rmse | major_rmse | minor_rmse | invalid_row_fraction | temperature_rmse | transformed_mse |
|---|---|---|---|---|---|---|---|
| 242 | 0.0035324298384152 | 0.0001942525417622 | 0.0002613190568097 | 2.632214025918768e-06 | 0.0 | 5.00938650931009 | 0.51939723138594 |
| 880 | 0.0040398981338205 | 0.0002404082285541 | 0.0003234172209437 | 2.235415919608481e-06 | 0.0 | 4.710547224187966 | 0.4786398485772801 |
| 22 | 0.0041824075283984 | 0.0002515231755507 | 0.0003383694373595 | 2.4330299636826584e-06 | 0.0 | 7.993068250562433 | 0.4680764049597106 |
| 847 | 0.0043122542139616 | 0.0002688325992182 | 0.0003616562103787 | 2.4722619563544664e-06 | 0.0 | 13.9719916755164 | 0.5545132288401283 |
| 1012 | 0.004335977607858 | 0.0002564093744545 | 0.0003449306797818 | 4.053825220936606e-06 | 0.0 | 3.575250137581631 | 0.5251207189201029 |

## 10 ms a posteriori

같은 현재 상태를 모든 멤버에 넣고 평균/대표를 다음 입력으로 사용한다. 중간 정답은 입력하지 않았다.
Validation:

| method | 수치 완주 % | 물리 유효 완주 % | 점화 구분 % | 점화오차 중앙값 % | ignition_pairs | T profile 평균 K | T pooled K | species_profile_mean |
|---|---|---|---|---|---|---|---|---|
| single_best | 100.0 | 0.0 | 72.22 | 10.18 | 12 | 38854.6 | 42945.87 | 9.736981032882223 |
| mean100 | 100.0 | 11.11 | 77.78 | 21.05 | 12 | 29229.58 | 32016.56 | 97.25384208599564 |
| medoid100 | 100.0 | 22.22 | 83.33 | 20.51 | 13 | 11908.85 | 14295.11 | 19.25439911247064 |

Test:

| method | 수치 완주 % | 물리 유효 완주 % | 점화 구분 % | 점화오차 중앙값 % | ignition_pairs | T profile 평균 K | T pooled K | species_profile_mean |
|---|---|---|---|---|---|---|---|---|
| single_best | 100.0 | 5.56 | 83.33 | 29.93 | 14 | 31235.61 | 36575.22 | 11.38033629107814 |
| mean100 | 100.0 | 5.56 | 88.89 | 47.48 | 14 | 29094.47 | 32264.25 | 76.12082993419567 |
| medoid100 | 100.0 | 16.67 | 83.33 | 55.56 | 13 | 12006.32 | 13746.58 | 19.170734652723223 |

**수치 완주와 물리적으로 유효한 완주를 구분해야 한다.** 물리 위반이 생겨도 finite 상태는 그대로 진화했다.
진단상 Y<−1e-12, Y>1+1e-12, T outside[200,6000] K, 질량 합/원소 오차>1%면 물리 허용 범위를 벗어난 것으로 표시했다.
모든 정확한 음수 occurrence도 별도로 기록했다. 이 threshold들은 correction/filter가 아니다.
NaN/Inf 또는 |T|>1e6 K, |Y|>1e3일 때만 수치 실패로 중단했다. 마지막 finite prefix를 full-profile 정확도로 보고하지 않는다.
수치 guard를 넘은 후보는 event CSV에 남기고 trace에 성공 step으로 추가하지 않았다.
Full-profile RMSE는 수치 완주한 궤적에서만 계산하며 물리 유효성과 함께 읽어야 한다.
점화는 T0+400 K crossing, reference만0.2 μs 정밀 측정이다.
점화 구분/조건부 지연오차는 **raw temperature crossing**이며 이미 비물리적인 궤적의 crossing도 포함할 수 있다.
물리 허용 범위 안에서 감지한 점화 수를 아래 별도 열로 제공했다. 작은 raw IDT 오차만으로 성공이라 주장하지 않는다.
Crossing이 처음 물리 위반 step에서 발생하면, 보간된 crossing 시간이 그 step보다 빠르더라도 물리적으로 유효한 점화로 인증하지 않는 보수적 기준을 적용했다.
이 분류는 저장된 raw trace에서 독립적으로 재계산·검증했으며 상태·실제 IDT·raw 분류·profile error를 바꾸지 않는다. 수정 전 진단과 변경 기록도 보존했다.

## 물리 위반 진단

| method | negative_state_fraction | negative_beyond_tolerance_fraction | minimum_Y | minimum_T | maximum_T | maximum_mass_sum_error | maximum_element_error | minimum_coverage | physically_valid_ignitions |
|---|---|---|---|---|---|---|---|---|---|
| mean100 | 0.7701015352485059 | 0.7664689284365349 | -747.287347536084 | -40708.93773633292 | 2334.3471754655247 | 714.4179354168333 | 718.2517923429773 | 1.0 | 0 |
| medoid100 | 0.6851185318491857 | 0.6851185318491857 | -392.72107623347546 | -26241.34846620404 | 4962.862126564951 | 357.014156167004 | 361.2580815853128 | 1.0 | 2 |
| single_best | 0.8080272834322025 | 0.8074718389655402 | -143.99236224024233 | -83781.3954499174 | 149256.70366114972 | 130.66934472795265 | 132.1682232857499 | 1.0 | 1 |

Species profile의 y축은 **symlog**로 그려 음수 예측을 숨기지 않았다. 온도 축도 음수/큰 finite 값을 포함한다.
별도의 temperature_near_reference 그림만 표시 창을 150–3500 K로 제한하며 full-range 그림과 함께 제공한다. 예측값이나 metric을 clipping한 것이 아니다.
검은 점선은 Cantera이며, x 표시는 수치 중단 endpoint다. 시간축만 log(0.002–10 ms)이고 모든 평가 point는 유지했다.

## One-step test와 비용

아래 one-step 표만 이전 비교와 동일한 IC당31개 probe를 사용했다. 선택에는 90,000 validation쌍 전체를 썼다.

| method | rmse | major_rmse | minor_rmse | relative_log_mean | negative_row_fraction | invalid_row_fraction | mass_sum_max | elemental_max | temperature_rmse |
|---|---|---|---|---|---|---|---|---|---|
| single_best | 0.0046078317429776 | 0.0065902659911005 | 3.4644563973197714e-05 | 0.0258496091997125 | 0.028673835125448 | 0.0 | 0.3592223799314697 | 0.3226126859470948 | 5.000782534588238 |
| mean100 | 0.0018678301216787 | 0.0026713727867595 | 2.206017652392273e-05 | 0.020547589041404 | 0.007168458781362 | 0.0 | 0.0700177920808096 | 0.0645070180586286 | 3.820486591632021 |
| medoid100 | 0.0019790696577664 | 0.0028304604793505 | 2.4195668766598883e-05 | 0.0229064643780151 | 0.0053763440860215 | 0.0 | 0.0880829032512395 | 0.0777310404360574 | 5.042469568515298 |

| method | split | seconds |
|---|---|---|
| single_best | val | 4.7672 |
| mean100 | val | 23.2847 |
| medoid100 | val | 69.0238 |
| single_best | test | 4.3956 |
| mean100 | test | 23.6031 |
| medoid100 | test | 67.7303 |

미완주는 실행 step 수가 적으므로 단순 wall time 감소를 같은 10-ms 완주 속도라고 해석하지 않는다.
이 CPU prototype의 비용이며 Cantera 대비 1000배 가속을 측정한 결과가 아니다.

## 해석과 한계

Signed transform은 생성·소모 증분을 한 함수와 역함수로 나타낼 수 있고, physical delta target은 큰 현재 농도를 직접 재예측하는 부담을 줄인다.
그러나 이전 모델도 transformed residual을 학습했으므로 identity 학습을 제거했다는 이유만으로 개선을 단정할 수 없다.
f'(v)=|v|^−.9라서 **0 근처의 작은 증분과 roundoff 부호 변화는 여전히 크게 확대된다**.
특히 거의 변하지 않는 종의 매우 작은 signed label도 transformed loss에서 중요해질 수 있다.
Inverse는 작은 transformed 값의 물리 증분을 강하게 줄이지만, 표준화·종별 bias·반복 누적까지 제거하지 않는다.
물리 증분의 평균은 독립 분산을 줄일 수 있으나 sum(ΔY)=0, AΔY=0, energy consistency를 강제하지 않는다.
Medoid도 이미 biased한 후보 사이의 중앙값을 고를 뿐, 새로운 물리 조건을 추가하지 않는다.
One-step 정확도, transformed MSE, 장기 profile/물리 보존성은 구분해서 봐야 한다.

선택 모델의 validation target/출력별 진단은 다음과 같다. Temperature만 선형 ΔT/1000이고 나머지는 signed Box–Cox다.
MSE share는 새로운 출력 정규화 좌표에서의 비중이며 이전 모델의 다른 target MSE와 크기를 직접 비교하지 않는다.

| channel | physical_max_abs_delta_train | physical_std_delta_train | encoded_std_delta_train | zero_delta_fraction_train | validation_standardized_MSE | validation_MSE_share | fraction_abs_delta_le_1e_minus12_val |
|---|---|---|---|---|---|---|---|
| T | 885.5612005305875 | 46.491609413897 | 0.046491609413897 | 0.0242768009075439 | 0.0116096712652694 | 0.0022352200904672 | 0.5483555555555556 |
| H2 | 0.0305281125349283 | 0.0014648653364592 | 2.1180883619640896 | 0.0229154849688031 | 0.536245712930945 | 0.1032438527829744 | 0.8162222222222222 |
| H | 0.007780568511178 | 0.0003070752380225 | 1.985868706228331 | 0.0178105501985252 | 0.518152254152665 | 0.0997603034521471 | 0.9175444444444444 |
| O | 0.0206247356591265 | 0.000832379020281 | 2.169013419766931 | 0.0180374361883153 | 0.5520093356306182 | 0.1062788367503725 | 0.9108444444444443 |
| O2 | 0.2045291143942621 | 0.0105975014927156 | 1.7794595532629869 | 0.0197390811117413 | 0.9617671231734763 | 0.1851698594170656 | 0.8130111111111111 |
| OH | 0.0243863452025747 | 0.0010185544768561 | 2.31102470318568 | 0.019285309132161 | 0.5880769613527295 | 0.1132229680515483 | 0.9081888888888888 |
| H2O | 0.202391019892999 | 0.0102235069483736 | 1.519530964694321 | 0.0241633579126488 | 1.5049216860566683 | 0.2897438790809477 | 0.8077 |
| HO2 | 0.0005148008227305 | 2.1249389397456363e-05 | 1.6951242868775165 | 0.019171866137266 | 0.2634859020611187 | 0.0507291695333151 | 0.8332 |
| H2O2 | 0.0007269751811502 | 1.5787653850721284e-05 | 1.3842540755520711 | 0.0187180941576857 | 0.2577036672359106 | 0.0496159108411617 | 0.8399111111111112 |
| N2 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 |

실제 이 데이터의 **N2 훈련 증분은 모두 정확히 0**이며 N2 loss 기여도도 0이다. 따라서 이번 손실을 N2 roundoff 때문이라고 설명할 수 없다.
Near-zero sensitivity는 변환 자체의 성질이지만 장기 실패의 유일한 원인으로 입증한 것은 아니다.
Per-output MSE와 물리 drift의 관계는 관찰 결과이고, 별도의 원인 분리 ablation 없이 단정하지 않는다.

이번 세 arm은 같은 새 target/physics-off 조건의 공정한 비교다.
이전 Wang 결과와 비교하면 target과 physics 사용 여부가 함께 바뀌므로, 차이를 target 하나의 효과로 분리해 주장할 수 없다.
같은 공개 test IC의 추가 실험이며 독립적인 외부 검증이 아니다.

## 파일과 재현

테스트: **68 passed, 24 warnings in 8.97s**. Signed0/1e-300 역변환, 물리 delta target, serialization,
grouped training 독립성, raw 음수 전달/위반 기록, 수치 완주와 물리 유효 구분을 검사했다.
훈련 모델마다 reload prediction을 확인했고 평가 첫 self-fed step과 실제 medoid 후보 identity를 검증했다.

- [전체 온도](results/delta100_raw/evaluation/figures_log_time/all_temperature_profiles.png), [온도 표시 창 확대](results/delta100_raw/evaluation/figures_log_time/temperature_near_reference.png), [어려운 조건](results/delta100_raw/evaluation/figures_log_time/difficult_initial_conditions.png), [signed 종 profile](results/delta100_raw/evaluation/figures_log_time/species_profiles.png), [보존 위반](results/delta100_raw/evaluation/figures_log_time/conservation.png)
- [설정](configs/delta100_raw.json), [선택 순위](results/delta100_raw/evaluation/validation_member_ranking.csv), [고정 선택](results/delta100_raw/evaluation/selection_frozen.json)
- [요약 CSV](results/delta100_raw/evaluation/summary.csv), [IC별](results/delta100_raw/evaluation/trajectories.csv), [one-step](results/delta100_raw/evaluation/one_step_probes.csv), [모든 모델 hash](results/delta100_raw/members.json)

```bash
python -m kinetics.delta_training --config configs/delta100_raw.json
python -m kinetics.delta_evaluation --root results/delta100_raw
python -m kinetics.delta_target_diagnostics --root results/delta100_raw
python -m kinetics.verify_delta --root results/delta100_raw
python -m kinetics.delta_report --root results/delta100_raw
python -m pytest -q
```

재실행은 config output을 새 경로로 바꾸거나 훈련 `--resume`를 사용한다. 기존 데이터/과거 결과를 덮어쓰지 않았다.
