# 시간 입력 / 여러 스텝 DeepONet 예비 실험

동일한 54/18/18 초기조건 분할, gamma=0.2, 기존 8,815개 시작점을 사용했다. 원본 데이터는 변경하지 않았다.
세 방법을 seed 11/22/33으로 각각 처음부터 학습했다. 각 모델은 활성 학습 계산 150초이며, 기존 10,000회 모델의 연속 학습이 아니다.
soft constraint와 학습 중 물리 보정은 없다. 단일 스텝/다중 스텝 모델은 같은 P-only 구조(9,719 parameters), 시간 입력 모델은 9,783 parameters다.
시간 입력 모델은 현재 T/Y/P를 branch, 앞으로 진행할 시간 간격을 trunk에 넣는다. 2/4/8/16 us 비율은 1/2,1/6,1/6,1/6이다.
다중 스텝은 활성 시간의 각 1/4을 K=1/4/16/64에 배정하고 예측 상태를 되먹여 전체 경로에 역전파한다. 후반 K의 업데이트 수가 적을 수 있다.
평가에서는 모든 방법이 2 us씩 5,000번 자기 예측을 되먹인다. seed별 결과와 물리 상태를 평균한 3-member ensemble을 구분했다.
raw는 clipping/projection/enthalpy correction/invalid filtering이 없다. repaired는 기존 inverse-domain clipping, 원소 보존 투영, 엔탈피 온도 재구성, 유효 후보 선택 후 평균 및 재보정을 사용한다.
완주는 유한/비음수 조성 및 200~6000 K 조건이다. raw에서 완주해도 원소/질량/엔탈피 보존이나 정확도를 보장하지 않는다.
NRMSE = RMSE(Cantera, NN) / RMS(Cantera), 무차원 비율. 불완주에는 전체 10 ms NRMSE를 부여하지 않았다.
worst9와 joint9는 T 및 8개 반응성 화학종을 함께 평가한다. 일정한 N2는 별도 NRMSE 표에 포함했다. 성공 기준 0.1은 예비 진단 기준이다.
방법 선택은 테스트를 읽기 전에 검증 raw 완주 수, joint9 통과 수, worst9 중앙값 순으로 동결했다. gamma를 탐색하거나 전체 화학종 최적 gamma라고 결론 내리지 않았다.

## Test ensemble (mean of 3)

| Method | Mode | Completed | joint9 <= 0.1 | Median worst9 among completed |
|---|---|---:|---:|---:|
| interval | raw | 0/18 | 0/18 | nan |
| interval | repaired | 18/18 | 0/18 | 9.6144 |
| multistep | raw | 14/18 | 0/18 | 0.889004 |
| multistep | repaired | 18/18 | 0/18 | 0.960691 |
| single_step | raw | 7/18 | 0/18 | 1.24938 |
| single_step | repaired | 18/18 | 3/18 | 1.13111 |

## Training

| Method | Seed | Updates | Active seconds | End |
|---|---:|---:|---:|---|
| single_step | 11 | 94719 | 150.001 | time_cap |
| interval | 11 | 89948 | 150.001 | time_cap |
| multistep | 11 | 29152 | 150.036 | time_cap |
| single_step | 22 | 95561 | 150.000 | time_cap |
| interval | 22 | 90259 | 150.000 | time_cap |
| multistep | 22 | 30192 | 150.013 | time_cap |
| single_step | 33 | 94638 | 150.001 | time_cap |
| interval | 33 | 90658 | 150.001 | time_cap |
| multistep | 33 | 29962 | 150.019 | time_cap |

## Files

- test_summary.csv: seed별 / mean3, raw / repaired 완주 및 joint9 성능
- NRMSE_by_IC.csv: 각 테스트 IC/변수의 mean3 전체 구간 NRMSE
- common_completed_NRMSE.csv: 세 방법이 공통 완주한 IC에서만 비교한 변수별 NRMSE
- profiles_test_{raw,repaired}.pdf: 18개 테스트 각각 3x10 전체 구간 그림
- profiles_test_{raw,repaired}_ignition_zoom.pdf: 같은 IC의 점화 근처 확대
- diagnostic_IC39_*: 과거에 확인했던 IC39 고정 진단 그림. 이번 결과에서 고른 최고 사례가 아니다.
- ../evaluations/{val,test}/one_step_NRMSE.csv: Cantera 현재 상태를 입력한 단일 스텝 진단. finite-only 수치는 불완전 표본 진단으로 전체 정확도와 구분.

공통 완주 집합이 없으면 비교값을 비워 두었다. 낮은 온도 NRMSE만으로 전체 반응계가 정확하다고 평가하지 않는다.
