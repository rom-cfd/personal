# 시간 입력 / 여러 스텝 DeepONet 예비 실험

동일한 54/18/18 초기조건 분할, gamma=0.2, 기존 8,815개 시작점을 사용했다. 원본 데이터는 변경하지 않았다.
세 방법 모두 직전 예비 실험의 가중치와 Adam 상태를 정확히 복원해 모델당 최대 150초 추가 학습했다. 누적 활성 학습 예산은 모델당 약 300초이다. 고정 training probe 정체 시 LR을 줄였으며, LR 하한 1e-6에서 6회 더 개선이 없을 때만 평탄 종료로 판정했다.
soft constraint와 학습 중 물리 보정은 없다. 단일 스텝/다중 스텝 모델은 같은 P-only 구조(9,719 parameters), 시간 입력 모델은 9,783 parameters다.
시간 입력 모델은 현재 T/Y/P를 branch, 앞으로 진행할 시간 간격을 trunk에 넣는다. 2/4/8/16 us 비율은 1/2,1/6,1/6,1/6이다.
다중 스텝은 K=64를 유지해 추가 학습했다. K=1부터 재시작하지 않았고, 입력은 자기 예측을 되먹이며 전체 경로에 역전파했다.
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
| interval | repaired | 18/18 | 0/18 | 12.2983 |
| multistep | raw | 13/18 | 0/18 | 0.874558 |
| multistep | repaired | 18/18 | 0/18 | 0.964196 |
| single_step | raw | 9/18 | 1/18 | 1.24764 |
| single_step | repaired | 18/18 | 2/18 | 1.03836 |

## Training

| Method | Seed | Updates | Active seconds | End |
|---|---:|---:|---:|---|
| single_step | 11 | 199086 | 150.001 | time_cap |
| interval | 11 | 169255 | 120.052 | plateau |
| multistep | 11 | 31301 | 150.026 | time_cap |
| single_step | 22 | 199724 | 150.002 | time_cap |
| interval | 22 | 148282 | 90.045 | plateau |
| multistep | 22 | 32329 | 150.074 | time_cap |
| single_step | 33 | 199418 | 150.002 | time_cap |
| interval | 33 | 173204 | 125.059 | plateau |
| multistep | 33 | 32128 | 150.084 | time_cap |

## Files

- test_summary.csv: seed별 / mean3, raw / repaired 완주 및 joint9 성능
- NRMSE_by_IC.csv: 각 테스트 IC/변수의 mean3 전체 구간 NRMSE
- common_completed_NRMSE.csv: 세 방법이 공통 완주한 IC에서만 비교한 변수별 NRMSE
- profiles_test_{raw,repaired}.pdf: 18개 테스트 각각 3x10 전체 구간 그림
- profiles_test_{raw,repaired}_ignition_zoom.pdf: 같은 IC의 점화 근처 확대
- diagnostic_IC39_*: 과거에 확인했던 IC39 고정 진단 그림. 이번 결과에서 고른 최고 사례가 아니다.
- ../evaluations/{val,test}/one_step_NRMSE.csv: Cantera 현재 상태를 입력한 단일 스텝 진단. finite-only 수치는 불완전 표본 진단으로 전체 정확도와 구분.

공통 완주 집합이 없으면 비교값을 비워 두었다. 낮은 온도 NRMSE만으로 전체 반응계가 정확하다고 평가하지 않는다.
