# 9종 수소 화학반응 ANN 전처리 비교 연구

## 핵심 결과

검증 데이터로 파라미터를 선택한 뒤 6개 변환 × 3개 시드 = 18개 모델을 실제 학습했다.
**Shifted Box–Cox는 극미량 입력 잡음에 훨씬 둔감하고 one-step 전체 질량분율 RMSE를 γ=0.1 대비 17.7% 낮췄다. 그러나 미량종 오차·음수 출력·점화지연·전체 0-D profile을 함께 보면 안정한 대체 모델이라고 결론 내릴 수 없다.**
특히 **2 ms a posteriori 적분에서 모든 방법의 원출력 및 clip/sum 처리 모두 물리 허용 완주율이 0%**다.
전처리만 바꿔 장기 적분 문제를 해결했다는 주장은 이 실험으로 뒷받침되지 않는다.
점화지연 숫자만 존재해도 질량·원소 보존이 깨진 궤적이면 신뢰 가능한 점화 예측으로 취급하지 않았다.

## 조사 결과와 실행 환경

- 저장소는 초기 커밋도 없는 빈 저장소였다. 기존 ANN/Box–Cox 코드·정규화·데이터·split·CI·평가 스크립트가 없었다. γ=0.1은 새 참조 구현이며 기존 모델 재현이라고 주장하지 않는다. 원본 데이터/구현을 덮어쓰지 않았다.
- Python 3.12.14, TensorFlow 2.20.0, Cantera 3.2.0, NumPy 2.5.4. CPU 5개, 약 33 GiB RAM, GPU 없음. TensorFlow intra/inter-op thread를 2/1로 제한했다. HPC 작업 없음.
- 최종 학습/기존 rollout 실행 127.5초, 확장 a posteriori 실행 81.2초. 패키지 설치·코드 작성·보고서 시간은 별도다. 실제 계산은 승인받은 약 30분 CPU 예산 이내다.

## 반응계, 데이터, 예측 대상

Cantera `h2o2.yaml`에서 Ar를 제외한 H2, H, O, O2, OH, H2O, HO2, H2O2, N2와 이 종들로만 이루어진 28개 반응을 사용했다. 단열·정압 0-D 이상기체 반응기다.
90개 독립 초기조건을 T0∈[900,1600] K, P∈[1,5] atm, 당량비∈[0.5,2]에서 균일 추출했다. 산화제 O2:N2=1:3.76, 연료 H2다.
Cantera `rtol=1e-10`, `atol=1e-20`. 반응 중 온도는 약 902–3024 K를 포함한다.

**최종 요청대로 reaction rate 예측은 사용하지 않는다.** 입력은 `[T(t), Y₁(t)…Y₉(t), P]`, 출력은 `[T(t+Δt), Y₁(t+Δt)…Y₉(t+Δt)]`다.
모든 학습·검증·테스트·ANN rollout에서 **Δt=2×10⁻⁶ s = 2 μs**로 고정했다.
원래 데이터 시간구간은 0–0.4 ms, 궤적당 200개 전이, 전체 18,000개 전이다.
초기조건 단위로 54/18/18개 궤적을 학습/검증/테스트에 고정 분리했다(10,800/3,600/3,600개 전이). 같은 궤적의 이웃 시점이 다른 split에 들어가지 않는다.
데이터 시드 20261029, 학습 시드 11/22/33이다. split과 초기조건은 NPZ에 저장했다.
원본 NPZ SHA256: `56457bf445a4970ca15361eb35f81cd218c01f76934a87652b8cba3b2a0ad7fe`.
Cantera의 [-1e-14,0) 음수 roundoff만 데이터 생성 시 0으로 처리하도록 했으나 이번 생성의 최소값은 0이었다. 학습/평가 예측값은 별도 진단을 제외하고 자르지 않았다.

## ANN 및 공정한 비교

11 입력 → 64 tanh → 64 tanh → 64 tanh → 10 선형 출력, 학습 파라미터 9,738개. Adam 0.001, batch 512.
모든 후보에 동일한 구조·split·시드·셔플 순서·epoch 예산을 사용했다. 마지막 선형층은 0 초기화하여 초기 예측이 현재 상태 유지와 같게 했다.
온도는 선형 스케일, 압력은 로그 스케일, Y는 비교 대상 변환을 적용했다. 평균·표준편차는 **학습 궤적에서만** 적합했다.
네트워크는 표준화된 변환 좌표의 잔차를 학습하고 현재 좌표에 더해 다음 상태를 복원한다. 따라서 실제 출력은 다음 상태이며, 물리적 signed 반응률이나 dY를 positive-only 변환에 넣지 않는다.

초기 12개 후보를 각각 15 epoch로 screening했다: γ=0.1/0.3/0.5와 softlog/asinh/shifted BC 각각 global ε=1e-8 및 종별 ε 두 가지.
종별 ε는 학습 데이터의 종별 양수 Y 중앙값 × 1e-2 또는 1e-4, 하한 1e-16이다. 예를 들어 multiplier=1e-4에서 H2O ε≈1.82e-5, H2O2 ε≈4.56e-11이었다.
검증 성능이 좋은 서로 다른 두 epsilon 변환 계열을 대상으로 1회차 ×0.1/10, 2회차 ×10^(-0.5)/10^(0.5)를 탐색했다. 총 20개 screening 학습이다.
γ 세 개와 각 epsilon 계열의 검증 최선 설정을 최종 비교에 포함했다. 각 모델 80 epoch를 모두 학습한 뒤 검증 점수가 가장 좋은 checkpoint를 복원했다.
선택 점수는 `mean(log1p(min(|error|/max(Y,1e-8),1e12))) + 10·major_RMSE + 10·mass_sum_MAE + 100·invalid_row_fraction`다.
이는 변환공간 MSE가 아닌 물리량 기반 검증 지표다. 최종 설정은 `selection_frozen.json`에 test 평가 전에 저장했다. test 기반 파라미터 조정·재학습은 하지 않았다.
초기조건을 한 번만 분할했으므로 시드 SD는 데이터 분할 불확실성을 포함하지 않는다.

## 변환 정의와 수치적 의미

- Box–Cox: `f(Y)=Y^γ/γ`, 역변환 `Y=(γf)^(1/γ)`, `f≥0`. 통상 `(Y^γ−1)/γ`에 상수를 더한 형태로, 표준화 후 수학적으로 동일하다. 0/극미량에서 상수 차감으로 정보를 잃는 것을 피했다. 음수 f는 NaN으로 표시하고 무효 예측으로 센다.
- Soft log: `f(Y)=log1p(Y/ε)`, `Y=ε·expm1(f)`.
- Asinh: `f(Y)=asinh(Y/ε)`, `Y=ε·sinh(f)`.
- Shifted BC: `f(Y)=expm1(γ·log1p(Y/ε))/γ`, `Y=ε·expm1(log1p(γf)/γ)`; 역변환 영역 `1+γf>0`. γ=0.3. 이는 `((Y+ε)^γ−ε^γ)/γ`를 ε^γ로 나눈 것이며 표준화 후 동등하다.

공용 모듈에는 signed 확장 `sign(x)·f(|x|)`와 대응 역변환도 구현했다. 이번 mass-fraction 연구에는 사용하지 않았다.
forward/inverse와 정규화는 float64, ANN 연산은 float32다. 테스트는 0, 1e-300까지의 극미량, 음수 signed 입력, 단조성, 역변환 영역, 종별 ε, 보존성, split, 저장·재로딩을 검사한다.

## One-step 물리 단위 평가

주요종/미량종은 각 reference sample에서 Y≥1e-3 / Y<1e-3로 구분했다. 상대오차 floor는 모든 방법에 동일한 1e-8이다.
**RMSE는 유한한 개별 예측값에 대한 값이다. 따라서 γ=0.3/0.5의 RMSE를 무효 행 비율과 분리해 읽으면 안 된다.** 무효 예측은 selection score에 별도 페널티를 준다.
아래 평균·SD는 고정 test 궤적에 대한 학습 시드 3개의 결과다.

| 변환 | 전체 Y RMSE ± seed SD | 주요종 RMSE | 미량종 RMSE | 무효 행 % | 음수 포함 행 % |
|---|---|---|---|---|---|
| Box–Cox γ=0.1 | 2.021e-03 ± 3.4e-04 | 2.901e-03 | 1.648e-05 | 0.00 | 0.00 |
| Box–Cox γ=0.3 | 2.728e-03 ± 1.0e-03 | 3.907e-03 | 4.113e-05 | 3.92 | 0.00 |
| Box–Cox γ=0.5 | 2.480e-03 ± 6.3e-04 | 3.491e-03 | 1.405e-04 | 15.95 | 0.00 |
| Soft log ε=1e-8 | 3.540e-03 ± 3.6e-04 | 5.080e-03 | 1.343e-05 | 0.00 | 15.95 |
| Asinh ε=3.162e-8 | 3.508e-03 ± 2.2e-04 | 5.035e-03 | 1.324e-05 | 0.00 | 16.69 |
| Shifted BC γ=0.3, ε=3.162e-8 | 1.664e-03 ± 2.7e-04 | 2.387e-03 | 4.544e-05 | 0.00 | 21.52 |

| 변환 | 합(Y) 오차 MAE | 원소분율 오차 MAE | T RMSE (K) | 변환공간 MSE |
|---|---|---|---|---|
| Box–Cox γ=0.1 | 3.393e-03 | 3.080e-03 | 12.50 | 6.829e-04 |
| Box–Cox γ=0.3 | 3.959e-03 | 3.616e-03 | 18.89 | 5.150e-03 |
| Box–Cox γ=0.5 | 2.948e-03 | 2.707e-03 | 19.11 | 1.277e-02 |
| Soft log ε=1e-8 | 7.049e-03 | 6.386e-03 | 11.42 | 1.055e-03 |
| Asinh ε=3.162e-8 | 6.910e-03 | 6.335e-03 | 10.38 | 8.236e-04 |
| Shifted BC γ=0.3, ε=3.162e-8 | 2.604e-03 | 2.413e-03 | 13.32 | 2.336e-03 |

현재 상태 유지 baseline은 전체 Y RMSE 4.159e-3, T RMSE 39.79 K, 상대오차 log 평균 0.0210이다. 학습 모델은 절대 RMSE를 줄이지만, 미량종·보존성·일부 상대지표에서는 단순 유지보다 나쁠 수 있다.
Shifted BC의 변환공간 MSE는 γ=0.1보다 크지만 물리 단위 전체/주요종 RMSE는 작다. 반대로 screening에서 asinh ε를 1e-8→1e-9로 줄이면 변환 MSE는 0.003158→0.002880으로 낮아지지만 물리 검증 점수는 0.3113→0.3579로 나빠졌다.
**변환공간 loss의 개선과 물리 예측 정확도의 개선은 서로 다른 결과다.**

## 극미량 입력 잡음 시험

test 입력의 처음 8개 종에 고정 시드 가우시안 잡음을 더하고 음수 입력을 0으로 제한한 뒤 N2로 합을 보정했다. noisy 입력과 clean 입력의 예측 차이를 비교했다.
이는 성분별 반올림/섭동 스트레스 시험이며 원소별 보존을 강제한 물리적 불확실성 모델은 아니다. 아래는 유한 prediction pair의 Y 변화 RMSE다. 무효 비율은 CSV에 별도로 기록했다.

| 변환 | 입력 잡음 SD=1e-16 | SD=1e-12 | SD=1e-8 |
|---|---|---|---|
| Box–Cox γ=0.1 | 1.089e-04 | 3.600e-04 | 2.123e-03 |
| Box–Cox γ=0.3 | 2.584e-07 | 7.225e-06 | 2.873e-04 |
| Box–Cox γ=0.5 | 8.029e-10 | 1.504e-07 | 8.174e-05 |
| Soft log ε=1e-8 | 1.687e-09 | 2.050e-07 | 9.916e-04 |
| Asinh ε=3.162e-8 | 1.355e-09 | 1.044e-07 | 8.501e-04 |
| Shifted BC γ=0.3, ε=3.162e-8 | 1.893e-10 | 1.404e-08 | 1.097e-04 |

입력 잡음 SD=1e-16에서 shifted BC의 출력 변화는 γ=0.1보다 약 5.75e+05배 작았다.
γ=0.1의 `df/dY=Y^-0.9`는 0 근방에서 발산한다. 극미량 radical 입력이 변환 좌표에서 크게 이동하고 DNN의 다른 종 출력까지 흔들 수 있다.
softlog/asinh/shifted BC의 원점 기울기는 1/ε로 유한하다. 그러나 bounded forward sensitivity가 보존성·역변환 오차·장기 안정성을 보장하지는 않는다.
γ=0.3/0.5는 원점 발산이 약해져 잡음 민감도는 줄었지만, 이 DNN에서 작은 종의 표현과 출력-domain 위반이 악화됐다.

## 실제 a posteriori 0-D 점화 및 profile 평가

단순 held-out one-step accuracy로 결론 내리지 않았다. **각 held-out 초기조건에서 시작해 ANN이 예측한 T,Y를 다음 입력으로 반복 사용했다. 중간 reference 상태를 입력하는 teacher forcing은 없다.**
P는 해당 반응기의 초기 압력으로 고정했다. reference는 같은 초기조건·메커니즘의 독립 Cantera 단열 정압 반응기다.
18개 초기조건 × 3개 시드 = 방법당 54개 궤적을 0.4 ms와 2 ms에서 평가했다. 2 ms는 학습 궤적 길이의 5배인 장기 확장 시험이다.
ANN 적분 간격은 계속 2 μs이며, Cantera 내부 adaptive 적분은 위의 엄격한 허용오차를 썼다.

점화지연은 (1) T0+400 K 최초 도달 시각(구간 선형 보간), (2) 400 K 상승이 확인되고 구간 내부에 있는 sampled 최대 dT/dt 시각을 모두 기록했다.
점화시각 reference 해상도 진단만 0.2 μs로 추가 계산했다. 이는 학습/test 데이터나 ANN Δt 변경이 아니다. 아래 점화 오차는 이 미세 reference를 사용한다.
2 ms 이내 reference 점화는 13/18개다. 나머지는 **2 ms 내 미점화/right-censored**이며 영구 미점화라고 단정하지 않는다.
2 μs ANN은 수 μs의 빠른 점화에 대해 시간 해상도 한계가 있으므로 작은 지연 차이를 과해석하면 안 된다.

`raw`는 원출력 반복이며, `clip/sum`은 매번 음수를 0으로 자르고 합이 1이 되게 정규화하는 별도 진단이다. 이 처리는 원소·에너지 보존을 보장하지 않는다.
NaN/Inf, T∉[200,6000] K, 양의 총질량 없음, raw에서 음수 종 발생 시 적분을 중단한다.
수치 완주는 이 중단 조건을 피했다는 의미일 뿐이다. Y가 매우 커지거나 원소/총질량이 틀어진 채 완주할 수 있다.
따라서 별도로 모든 시점에서 `|sum(Y)−1|≤0.01`, 최대 원소질량분율 drift≤0.01, Y≤1.01를 만족해야 **물리 허용 완주**로 센다. 이는 느슨한 진단 임계값이며 정확한 보존을 뜻하지 않는다.
점화 전에 이 기준을 위반하면 해당 점화값을 물리 허용 점화 쌍으로 세지 않는다.

profile 오차는 시간 적분 L2 RMSE, 즉 `sqrt(integral(error² dt)/horizon)`다. 종별 전체 profile RMSE와 reference L2에 대한 상대오차도 CSV에 제공한다.
**미완주 궤적의 전체 profile 오차는 NaN**이며, 별도로 coverage와 prefix 오차를 기록했다. 실패한 예측을 0 오차로 대체하지 않았다.
아래 `*` profile 평균은 **완주한 궤적에 한정**되므로 완주율과 같이 읽어야 하며, 방법간 단독 순위 기준으로 사용하면 생존자 편향이 생긴다.
점화 오차도 시각이 존재하는 쌍의 평균이라 실패/미점화를 포함하지 않는다. 마지막 열로 물리 허용 쌍 수를 명시했다. 각 행 전체 대상은 54개다.

### 0.4 ms: 학습 궤적과 같은 시간구간

| 변환 | 처리 | 수치 완주 % | 물리 허용 완주 % | 전체 T RMSE K* | 전체 Y RMSE* | τ400오차 % (쌍 수) | τdT/dt오차 % (쌍 수) | 물리 허용τ400쌍 수 |
|---|---|---|---|---|---|---|---|---|
| Box–Cox γ=0.1 | raw | 51.9 | 0.0 | 493.4 | 2.589e+02 | 34.6 (29) | 40.2 (29) | 1 |
| Box–Cox γ=0.1 | clip/sum | 29.6 | 0.0 | 451.3 | 3.106e-02 | 41.9 (32) | 40.7 (30) | 3 |
| Box–Cox γ=0.3 | raw | 44.4 | 0.0 | 839.3 | 5.759e-02 | 67.6 (34) | 67.9 (34) | 0 |
| Box–Cox γ=0.3 | clip/sum | 20.4 | 0.0 | 441.1 | 5.016e-02 | 98.5 (30) | 82.0 (30) | 1 |
| Box–Cox γ=0.5 | raw | 0.0 | 0.0 | 미완주 | 미완주 | 75.3 (20) | 81.6 (14) | 0 |
| Box–Cox γ=0.5 | clip/sum | 0.0 | 0.0 | 미완주 | 미완주 | 83.7 (20) | 86.0 (20) | 0 |
| Soft log ε=1e-8 | raw | 74.1 | 0.0 | 712.2 | 2.404e+34 | 38.0 (36) | 41.9 (36) | 5 |
| Soft log ε=1e-8 | clip/sum | 20.4 | 0.0 | 951.5 | 5.685e-02 | 43.6 (27) | 56.0 (27) | 5 |
| Asinh ε=3.162e-8 | raw | 75.9 | 0.0 | 830.7 | 3.260e+16 | 35.7 (36) | 37.2 (36) | 3 |
| Asinh ε=3.162e-8 | clip/sum | 16.7 | 0.0 | 573.2 | 4.260e-02 | 37.9 (27) | 73.8 (27) | 2 |
| Shifted BC γ=0.3, ε=3.162e-8 | raw | 20.4 | 0.0 | 366.7 | 3.669e-02 | 38.8 (24) | 36.7 (23) | 0 |
| Shifted BC γ=0.3, ε=3.162e-8 | clip/sum | 29.6 | 5.6 | 430.4 | 3.028e-02 | 84.4 (16) | 65.2 (16) | 0 |

### 2 ms: 장기 확장

| 변환 | 처리 | 수치 완주 % | 물리 허용 완주 % | 전체 T RMSE K* | 전체 Y RMSE* | τ400오차 % (쌍 수) | τdT/dt오차 % (쌍 수) | 물리 허용τ400쌍 수 |
|---|---|---|---|---|---|---|---|---|
| Box–Cox γ=0.1 | raw | 24.1 | 0.0 | 605.0 | 6.906e+00 | 34.6 (29) | 40.2 (29) | 1 |
| Box–Cox γ=0.1 | clip/sum | 9.3 | 0.0 | 508.7 | 3.346e-02 | 43.5 (33) | 42.5 (31) | 4 |
| Box–Cox γ=0.3 | raw | 20.4 | 0.0 | 885.4 | 4.942e-02 | 70.0 (37) | 70.2 (37) | 0 |
| Box–Cox γ=0.3 | clip/sum | 14.8 | 0.0 | 444.1 | 6.066e-02 | 98.4 (33) | 83.4 (33) | 1 |
| Box–Cox γ=0.5 | raw | 0.0 | 0.0 | 미완주 | 미완주 | 76.3 (21) | 82.7 (15) | 0 |
| Box–Cox γ=0.5 | clip/sum | 0.0 | 0.0 | 미완주 | 미완주 | 84.9 (22) | 87.0 (22) | 0 |
| Soft log ε=1e-8 | raw | 33.3 | 0.0 | 716.1 | 1.293e+14 | 42.4 (39) | 46.0 (39) | 5 |
| Soft log ε=1e-8 | clip/sum | 3.7 | 0.0 | 93.7 | 9.595e-03 | 47.2 (29) | 58.6 (29) | 5 |
| Asinh ε=3.162e-8 | raw | 11.1 | 0.0 | 884.8 | 1.235e+26 | 40.2 (39) | 41.6 (39) | 3 |
| Asinh ε=3.162e-8 | clip/sum | 3.7 | 0.0 | 525.5 | 3.032e-02 | 40.0 (28) | 742.9 (28) | 2 |
| Shifted BC γ=0.3, ε=3.162e-8 | raw | 3.7 | 0.0 | 647.4 | 6.451e-02 | 38.8 (24) | 36.7 (23) | 0 |
| Shifted BC γ=0.3, ε=3.162e-8 | clip/sum | 7.4 | 0.0 | 441.0 | 4.365e-02 | 80.8 (18) | 63.8 (18) | 0 |

one-step RMSE가 가장 낮은 shifted BC도 2 ms 점화지연·전체 profile과 물리 허용 완주를 통과하지 못했다.
raw softlog/asinh의 일부 수치 완주에서는 종 질량분율이 비현실적으로 커졌다. finite/positive라는 사실만으로 안정하다고 볼 수 없는 사례다.
일부 completed-only profile 평균이 작아 보여도 대부분의 궤적이 실패하거나 원소 보존을 어기므로 우수한 적분기라는 뜻이 아니다.
모든 초기조건의 seed 11 온도 profile을 그림으로 제공하며, 종별 그림은 사전에 고정한 첫 test 초기조건을 사용했다. 성능이 좋은 사례를 골라 보여주지 않았다.

## 각 변환의 성공·실패 원인

1. **γ=0.1 Box–Cox:** 이번 one-step 비교에서는 무효·음수 없이 미량종 표현과 상대오차의 균형이 가장 좋았다. 다만 극미량 잡음 증폭이 크고, 질량·원소 보존 제약이 없는 DNN의 작은 편향이 반복 입력에서 누적된다. 높은 one-step 정확도가 점화·profile 정확도로 이어지지 않았다.
2. **γ=0.3/0.5:** 작은 Y에 대한 forward 민감도를 완화하지만 trace 종을 압축된 작은 좌표에 놓는다. 유한 학습·float32 출력 오차가 domain 경계를 넘으면 다음 상태를 복원할 수 없다. γ=0.5는 최종 test 행의 약 16%가 무효이며 장기 수치 완주도 없다.
3. **Soft log:** ε 아래에서 선형이라 극미량 잡음은 억제한다. 그러나 큰 Y에서 지수형 역변환이 출력 좌표 오차를 증폭하며 주요종·질량합 오차가 커졌다. 작은 음수 좌표는 음수 Y로 복원된다. 이번 음수 크기는 대체로 1e-9 수준이지만 raw chemistry state로 사용하기에는 부적합하다.
4. **Asinh:** 원점에서 매끄럽고 signed 값에도 자연스럽다. trace/온도 one-step 오차에는 장점이 있지만 주요종 오차 및 반복 drift가 크다. signed 수학적 정의가 질량분율의 음수를 물리적으로 정당화하지 않는다.
5. **Shifted BC:** ε 아래는 유한 기울기, 그 위는 power-law tail이어서 로그 계열보다 주요종을 잘 복원하고 γ=0.1보다 잡음에 강했다. 그러나 smoothing으로 미량종 정보가 약해지고 음수 출력이 생긴다(이번 최소값 평균 약 −2.65e-8). clamp/normalize는 원소/에너지 보존을 고치지 않으므로 점화와 profile의 장기 정확도는 부족하다.
6. **종별 ε:** 종별 dynamic range를 반영하지만 양수 중앙값은 점화 전의 0/극미량 구간보다 점화 후 농도를 대표하기 쉽다. 예를 들어 H2O에 큰 ε를 주면 초기 생성 정보를 평탄화한다. 이번 두 multiplier는 검증 점수에서 global ε보다 나빴다. 종별 ε 자체가 항상 나쁘다는 일반 결론은 아니다.

## 효율과 검증

| 변환 | 80 epoch 학습(s) | batch 추론(μs/sample) | 단일 상태 추론(μs) |
|---|---|---|---|
| Box–Cox γ=0.1 | 3.80 | 2.03 | 2928 |
| Box–Cox γ=0.3 | 3.97 | 1.99 | 3012 |
| Box–Cox γ=0.5 | 4.03 | 2.03 | 2990 |
| Soft log ε=1e-8 | 3.59 | 1.99 | 2977 |
| Asinh ε=3.162e-8 | 3.83 | 1.98 | 3014 |
| Shifted BC γ=0.3, ε=3.162e-8 | 3.74 | 2.06 | 3025 |

batch 추론은 3,600개 상태를 한 번에 처리한 warm-up 후 5회 중앙값이며, 단일 상태 추론은 Python/Keras 호출 overhead를 포함한다. batch throughput을 순차 chemistry 적분 속도로 해석하면 안 된다. Cantera 대비 speedup은 주장하지 않는다.
최종 모델 18개의 저장·재로딩 후 원래 test 예측이 엄격한 수치 허용오차 내에서 모두 재현됐다. 파일명 소수점으로 인한 초기 저장 충돌을 수정한 뒤 동일 설정을 재실행했으며, model persistence 회귀 테스트를 추가했다.
테스트 40개 통과. Keras/NumPy 연동의 deprecation warning 12개가 있지만 실패는 없다. 데이터 checksum, 재로딩 및 self-fed rollout 테스트를 포함한다.

## 한계와 다음 연구

이 결과는 작은 0-D 수소 benchmark, 한 데이터 분할, 작은 DNN 및 제한된 학습 예산에서 얻었다. 기존 사용자의 ANN을 개선했다는 증거가 아니고 CFD/다른 연료/압력 범위로 일반화하지 않는다.
선별과 추가 두 round는 물리 단위 **one-step 검증**으로 했으며, a posteriori 결과는 이후 독립 평가에 사용했다. 따라서 long-term 최적 파라미터를 찾았다고 주장하지 않는다.
다음 단계에서는 test를 고정한 채 **검증 초기조건의 점화·전체 profile·보존 실패를 선택 기준**에 포함하고, 원소 보존 projection/보존 구조 출력, enthalpy에 따른 온도 복원, rollout loss 및 trace noise augmentation을 별도 통제 실험으로 검토해야 한다.
이는 전처리 효과와 섞지 않도록 공통 제약 적용 여부를 모든 후보에 동일하게 비교해야 한다. 이번에는 세 번째 최적화 round나 test 결과를 이용한 재튜닝을 하지 않았다.
현재로서는 shifted BC를 **잡음 강건성을 더 연구할 후보**로 권장할 수 있지만, 장기 chemistry integrator로 채택할 수 있는 승자는 없다.

## 산출물과 재현

- [실행 방법](README.md), [버전 고정](requirements.lock.txt), [실험 설정](configs/study.json)
- [변환 모듈](kinetics/transforms.py), [데이터 생성](kinetics/data.py), [학습/평가](kinetics/study.py), [후처리](kinetics/analysis.py), [a posteriori 0-D](kinetics/posteriori.py), [reference 해상도](kinetics/reference_resolution.py), [보고서 생성](kinetics/report.py)
- [최종 one-step CSV](results/verified/test_comparison.csv), [seed 평균/SD](results/verified/summary.csv), [검증 탐색](results/verified/validation_screen.csv), [history](results/verified/history.csv)
- [종별 오차](results/verified/species_errors.csv), [농도 bin 분포](results/verified/concentration_errors.csv), [잡음](results/verified/noise_robustness.csv), [추론 시간](results/verified/inference_benchmark.csv)
- [0-D 전체 궤적 지표](results/verified/posteriori/trajectory_metrics.csv), [점화 비교](results/verified/posteriori/ignition_comparison.csv), [0-D 종별 profile](results/verified/posteriori/species_profile_errors.csv), [0-D 요약](results/verified/posteriori/summary.csv)
- [원소/질량 보존 survival](results/verified/figures/conservative_survival.png), [잡음 그래프](results/verified/figures/noise_robustness.png), [종별 one-step](results/verified/figures/species_errors.png), [loss vs physical](results/verified/figures/loss_vs_physical.png)
- [0-D raw 온도: 모든 초기조건](results/verified/posteriori/all_temperature_profiles_repair0.png), [0-D repaired 온도](results/verified/posteriori/all_temperature_profiles_repair1.png), [9종 전체 profile](results/verified/posteriori/all_species_profiles_repair1.png), [점화지연 산점도](results/verified/posteriori/ignition_delay.png)

**유효 최종 결과 경로는 `results/verified/`**다. `results/study/`는 중단한 pilot이고 `results/final/`은 모델 저장 검증 전의 대체된 실행이다. 재현·해석에는 verified 결과만 사용한다. 원본 및 중간 기록은 보존했다.
