이번 순차 장기 학습·평가를 완료했다. 다만 **설정한 평탄 기준에 도달한 모델은 0/20개**이며, 모두 모델별 배정 시간상한에서 종료했다. 따라서 아래는 수렴이 확인된 모델의 비교가 아니라, 승인된 6시간 창 안에서 얻은 종료 모델의 비교다. 판정을 통과시키려고 기준이나 LR을 바꾸지 않았다.

학습은 2026-11-02 17:48:41 KST에 시작했고, 본 파이프라인은 23:34:45 KST에 끝났다. 학습 wall time은 19,204.43초(5시간 20분 4초), 평가·최종 train/validation loss 계산은 1,297.10초(21분 37초), 보고서·그림은 262.75초(4분 23초)였다. 합계는 20,764.28초(5시간 46분 4초)이며 후속 읽기 전용 검증·온도 개요 그림도 같은 6시간 상한 안에서 마쳤다. 학습·평가·그림·후속 검증은 순차 실행했다.

| 계열 | 모델 수 | 모델당 파라미터 | 종료 update 범위 | 전체 멤버 update 합 | 이번 멤버 실행시간 합 |
|---|---:|---:|---:|---:|---:|
| P-only | 10 | 9,719 | 338k–438k | 4,092,000 | 9,608.55초 |
| T/P | 10 | 9,687 | 263k–359k | 3,116,000 | 9,572.39초 |
| ANN 대조군 | 기존 첫 10개 재사용 | 9,738 | 기존 10k 유지 | 기존 100,000 | 신규 학습 0초 |

P-only는 이번에 만든 10k 체크포인트의 Adam·RNG에서 정확히 이어갔다. 표의 P-only update 합에는 기존 100,000이 포함되므로 이번 추가 update는 3,992,000이다. T/P는 과거 파일에 Adam/RNG가 없어 초기화부터 새로 재현하고 상태를 유지하며 연장했다. 새 T/P의 10k 체크포인트 10개는 기존 10k 모델과 **모든 가중치 배열이 정확히 일치**했다. 이것은 새 재현 검증이며, 소실된 과거 Adam 파일이나 장기 가중치가 복구됐다는 뜻은 아니다.

두 계열 모두 seed=11,22,…,110, batch512 복원추출, Adam, global norm clip5다. update1–6000은 LR=.001, 6001–9000은 .0003, 이후는 .0001을 유지했다. 순서는 P-only 10개 → T/P 10개 → validation 평가 → test 평가 → 그림이었다. TensorFlow intra/inter=2/1, OMP/OpenBLAS/MKL=2, oneDNN=0, CPU만 사용했다. 코드의 직렬 TensorFlow loop는 멤버 간 병렬 학습을 하지 않으며, scalar update 및 재개와 가중치·Adam·다음 minibatch가 정확히 같은지 테스트했다.

문제와 데이터는 기존 0-D 단열·정압 폐쇄 수소–공기 반응계, 이상기체, 지정 9종·28반응이다. 종 순서는 H2,H,O,O2,OH,H2O,HO2,H2O2,N2다. 90개 궤적×5,001점, train/validation/test=54/18/18, train에서 선별된 8,815쌍, 모든 정답 간격 2 μs, 전체 horizon 10 ms를 유지했다. 증강은 없고 공통 train-only Box–Cox gamma=.1·표준화·변환 좌표 잔차 타깃을 사용했다. 현재 T,Y9,P에서 다음 T,Y9를 예측하는 유한 상태 DeepONet-style 전이 모델이다.

평탄 판정은 전체 train MSE를 1k마다 관측하고, 최소 50k 이후 5k마다 연속한 두 20k 창의 q25/median/q75 상대변화 절댓값이 1%/1%/3% 이내인지 검사한다. 각 창에는 실제 20개 관측값이 필요하며 연속 세 번의 통과로 10k 지속을 요구했다. test 성능으로 종료점이나 설정을 선택하지 않았다. 1M update 안전 상한은 도달하지 않았고, 모델별 시간상한이 먼저 적용됐다. 평가·보고서에 40분을 유보했으며 상한 종료를 평탄 종료로 바꾸어 표기하지 않았다.

| 계열 | 최종 전체 train residual MSE 멤버 평균 | 전체 validation residual MSE 멤버 평균 |
|---|---:|---:|
| ANN10 | 1.31007e-5 | 3.12726e-5 |
| P-only10 | 3.93335e-6 | 1.52319e-5 |
| T/P10 | 2.02466e-5 | 4.22234e-5 |

train은 선택된 8,815쌍, validation은 전체 90,000쌍의 평균이다. 서로 표본 가중치가 다른 두 split의 절대값을 일반화 격차처럼 해석하지 않는다. mini-batch loss와 위 전체 데이터 MSE도 구분한다. P-only의 validation one-step loss가 낮아도 장기 자기 예측이 더 좋아지지는 않았다.

기존에 이미 관찰한 test 18개 IC에서 수행한 repaired 10 ms 결과다. 온도 RMSE는 시간 사다리꼴 적분으로 계산한 IC별 RMSE의 산술평균이다.

| 방법 | 10 ms 완주 | 공통 완주 16개 IC 온도 RMSE 평균 K | 각 방법의 완주 IC만 사용한 평균 K |
|---|---:|---:|---:|
| ANN10 | 18/18 | **192.9068** | 179.1297 (18개) |
| P-only10 | 18/18 | 403.6951 | 360.8736 (18개) |
| T/P10 | 16/18 | 646.9501 | 646.9501 (16개) |
| ANN+P-only20 | 18/18 | 261.0384 | 234.6912 (18개) |
| ANN+T/P20 | 18/18 | 454.1620 | 464.4924 (18개) |

마지막 열은 완주 집합이 서로 다르므로 직접적인 공정 비교에는 공통 16개 열을 사용한다. 공통 IC는 `[4,5,7,32,39,44,45,57,64,68,69,73,74,76,78,88]`이다. 이번 종료 모델들에서는 ANN10이 공통 집합의 온도 RMSE가 가장 낮았다. Pooled RMSE는 별도 CSV 열이며 위 IC별 평균과 다르다.

T/P10의 실패는 IC18의 step277(0.554 ms), IC53의 step244(0.488 ms)에서 발생했다. 두 경우 유효 후보가 0/10이었다. 저장 trace의 마지막 정상 시점은 각각 0.552/0.486 ms이며 실패 이후 값을 채우지 않았다. 두 mixed20은 각 팀 자신의 현재 상태에서 같은 ANN10과 해당 DeepONet10을 평가해 후보를 평균했다. 별도 궤적의 사후 평균이나 가중치 평균이 아니다.

test dense 90,000쌍의 repaired one-step 온도 RMSE는 ANN10=1.1990 K, P-only10=0.8466 K, T/P10=1.3434 K, ANN+P-only20=0.8716 K, ANN+T/P20=1.0901 K였다. 이 평가에서는 모든 팀의 invalid row/candidate 비율이 0이었다. 실제 현재 상태를 입력하는 one-step 평가와 자기 상태를 반복 입력하는 rollout 성능이 다르다는 결과다. 모델이나 종료점은 이 수치로 다시 선택하지 않았다.

| raw 방법 | 10 ms 완주 | 기존 물리 허용 조건까지 만족한 완주 |
|---|---:|---:|
| ANN10 | 2/18 | 0/18 |
| P-only10 | 6/18 | 1/18 |
| T/P10 | 12/18 | 0/18 |
| ANN+P-only20 | 9/18 | 0/18 |
| ANN+T/P20 | 6/18 | 0/18 |

raw와 repaired는 별도 자기 예측이다. repaired의 질량·원소·엔탈피 보존은 기존 보정으로 강제한 것이며 학습된 보존성이 아니다. 엔트로피는 후보 수용 기준이 아닌 진단으로 유지했다. repaired에서도 전체 가용 trace의 최소 S−S0는 P-only 약 −136.8, ANN+P-only 약 −23.1, T/P 약 −2460.3, ANN+T/P 약 −648.5 J/(kg K)로 음수였다. 따라서 완주만으로 올바른 화학 경로나 열역학적 일관성을 입증하지 않는다. 미완주를 포함한 진단은 가용 prefix에 대한 값임을 CSV의 full_trace 표기로 구분했다.

순차 평가 20개 조합(5팀×raw/repaired×val/test)을 모두 저장했다. NN 추론, feature encoding, decode, 후보 보정, 평균 보정, shared30 모델 로딩, warm-up, 전체 rollout, dense one-step, 물리 진단 시간을 분리했다. `neural_decode`는 합계이고 features/neural/decode는 그 하위 항목이므로 둘을 다시 합산하지 않는다. 실패한 rollout은 호출 수와 평가 행 수가 적을 수 있으므로 작업량 CSV를 함께 읽는다. 과거 ANN grouped training 시간을 나눠 새 순차 학습과 직접적인 공정 속도비로 주장하지 않았다.

저장된 20개 endpoint·Adam iteration과 1,249개 평탄 검사를 재계산해 확인했다. 기존 결과 1,607개 파일의 SHA256은 모두 불변이고 기존 추적 파일의 Git 차이도 없다. 구현 단계의 전체 pytest는 96 passed/84 warnings, 추가 member-runner 통합 테스트는 2 passed였다. 경고는 기존 Keras/NumPy __array__ 관련이었다. 실험 코드는 그 뒤 바꾸지 않았고, 후속 검증/개요 그림은 별도 scripts 파일에서 읽기 전용으로 수행했다.

주요 산출물은 다음과 같다.

- [영문 전체 보고서와 모든 지표](reports/report_1793629823192730538/REPORT.md)
- [공통 완주 IC 비교 CSV](reports/report_1793629823192730538/common_completed_IC_comparison.csv)
- [모델별 update·종료 사유·시간](reports/report_1793629823192730538/training_summary.csv)
- [전체 관측 loss와 LR](reports/report_1793629823192730538/training_loss_and_lr.png)
- [P-only 10개 전체 train loss 곡선](reports/report_1793629823192730538/p_only_full_train_per_member.png)
- [T/P 10개 전체 train loss 곡선](reports/report_1793629823192730538/tp_full_train_per_member.png)
- [test 18개 IC의 온도 개요](overview/run_001/test_repaired_all18_temperature.png)
- [평탄 판정 1,249개 재계산 CSV](audits/audit_001/plateau_checks.csv)
- [반복 재개를 포함한 시간 합산](audits/audit_001/training_time_all_invocations.csv)
- [평가 작업량·시간·완주·점화 혼동행렬](audits/audit_001/evaluation_workload_and_time.csv)
- [T/P 10k 가중치 재현 검증](audits/audit_001/tp_10k_weight_replay.csv)

각 test IC의 T+9종 패널은 raw/repaired 각각 18개로 총 36 PNG+36 PDF이며 같은 prefix 데이터의 프로파일 CSV 36개도 저장했다. 긴 loss tail은 모든 멤버가 공통으로 관측한 지점의 요약 밴드와 별도로 모델별 전체 곡선에 표시했다. min–max 밴드는 신뢰구간이 아니다. 최종 모델·Adam·RNG·history·monitor·hash는 `final_members.json`이 가리키는 새 체크포인트 디렉터리에 있다.

재현 절차는 저장소의 `DEEPONET_LONG_WORKFLOW.md`, 설정은 `configs/deeponet_long_study.json`이다. 후속 파일 검증은 `scripts/audit_deeponet_long_results.py`, 온도 개요는 `scripts/plot_deeponet_long_overview.py`를 사용한다. 새 출력 경로를 지정해야 하며 이미 끝난 6시간 예산을 resume으로 초기화하지 않는다.

현재 브랜치는 bw → origin/bw, Git HEAD는 `3affe7545a3946075a7f80547d70cb6147c15c06`이다. 새 코드·모델·체크포인트·결과는 현재 클라우드 디스크의 untracked 파일이며 commit/push와 게시 snapshot 변경은 하지 않았다. 이번 실행은 과거 소실된 장기 학습 결과의 복구가 아니며, 현재 종료 모델의 재검증 가능한 새 연구 실행이다.
