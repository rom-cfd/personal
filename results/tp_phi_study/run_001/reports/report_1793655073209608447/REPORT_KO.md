T/P/초기 phi trunk 실험을 종료했다. 아래 결과는 사전에 고정한 10k 및 장기 종료점의 비교다.
장기 모델 10개 중 평탄 도달 0개, 시간상한 종료 10개. 시간상한 종료는 수렴으로 표시하지 않았다.
장기 종료 update 범위 467,000–825,000.
실험 파이프라인 경과 20165.90초 / 승인 상한 21,600초. 구현·사전 테스트 및 사전 백업 시간은 학습 파이프라인 시간과 별도다.

기존 코드·모델·Adam/RNG·결과 11,159개 파일의 사전 백업 커밋은 798a673f863c0aa6f59ab6b69d684815743145a6이며 학습 전 GitHub bw 반영을 확인했다.
새 실험은 기존 9종·28반응, 54/18/18 분할, 8,815 train 쌍, 2us target, 10ms horizon을 유지했다. 증강은 없다.
Branch=현재 Y9, trunk=현재 T·log(P)·초기 당량비 phi. Phi는 데이터 initials에서 초기조건별 상수로 읽으며 미래 정답을 사용하지 않는다.
폐쇄계의 원소비를 정확히 보존하면 phi는 조성에도 포함된 정보다. 이 실험은 보존된 혼합물 정보를 trunk에 명시하는 효과를 검토한다.
파라미터 9,719개. 공통 T/Y/logP 전처리는 기존 ANN과 정확히 같고, phi 표준화는 선별 train 입력에서만 fit했다.
ANN은 기존 첫10개(seed11,22,…,110), 10k 모델을 재사용했다. 새 10개는 같은 seed·batch512·복원추출·Adam·clip5 및 절대 update LR 경계 6000/9000을 사용했다.
모든 모델을 10k까지 순차 학습해 저장한 뒤 Adam·RNG·history를 복원해 장기 학습했다. 재개는 LR/전체 6시간 시계를 초기화하지 않는다.
장기 평탄 정책은 전체 train MSE를1k마다 관측하고 최소50k부터5k마다 두20k창의 q25/median/q75 상대변화1%/1%/3%를 연속3회(10k지속) 요구한다.
모델별 순차 실행시간을 기록했고, 전체 예산 중40분은 두 종료점 평가·그림에 유보했다. 장기 모델들에는 남은 시간을 나누며 조기 평탄으로 남은 시간은 뒤 모델에 배분된다.
TF intra/inter=2/1, OMP/OpenBLAS/MKL=2, oneDNN=0, CPU만 사용했다. 학습·평가·그림은 순차다.

평가 완료 20/20 arms: ANN10, phi10k10, ANN+phi10k20, philong10, ANN+philong20 × raw/repaired × val/test.
Mixed 팀은 자기 현재 상태에서 20개 후보를 계산해 평균했다. Raw는 모든 후보가 유한해야 하며 repaired는 기존 동일 보정과 유효 후보 제외 규칙을 적용했다.
초기 phi는 raw에서 원소량이 드리프트해도 고정된다. 물리 보정은 현재 예측 상태를 기준으로 동작하며 미래 참값을 사용하지 않는다.
One-step은 참 현재 상태를 입력하고 rollout은 자기 예측을 재입력한다. 두 지표를 동일하게 해석하지 않는다.
Test18 IC는 이전 연구에서도 관찰된 집합이다. Test를 종료·모델 선택이나 하이퍼파라미터 변경에 사용하지 않았다.
미완주는 가용 prefix만 기록하고 전체 RMSE에서 제외했다. 공정 비교는 common_completed_IC_comparison.csv의 동일 IC 교집합을 사용한다.
보존은 물리 보정으로 강제한 성질이며 학습된 보존성이 아니다. 엔트로피는 진단만 하고 후보 필터에는 사용하지 않았다.
과거 ANN 병렬 그룹 학습 시간과 이번 순차 시간으로 공정한 속도비를 주장하지 않는다. NN/feature/decode/repair/rollout/one-step/로딩/warm-up 시간을 분리했다.
기존 결과 11987개 SHA256 불변. 20개 checkpoint 해시·iteration과 평탄 검사 1279개를 재검증했다.

|방법|test repaired10ms 완주|완주 IC 온도 RMSE 평균 K|
|---|---:|---:|
|ANN10|18/18|179.129665|
|ANN+T/P/phi20 (10k)|18/18|464.803650|
|ANN+T/P/phi20 (long)|18/18|403.492421|
|T/P/phi10 (10k)|17/18|483.552013|
|T/P/phi10 (long)|18/18|633.801948|

위 평균은 각 방법의 완주 집합 기준이다. 공통 완주 IC 수와 평균은 별도 CSV에서 확인한다.
[공통 IC 비교](common_completed_IC_comparison.csv) · [학습 기록](training_summary.csv) · [모든 지표](summary.csv) · [one-step](one_step_summary.csv) · [loss](training_loss_and_lr.png) · [멤버별 loss](full_train_per_member.png) · [test18 온도](test_repaired_all18_temperature.png)
각 test IC의 raw/repaired T+9종 PNG/PDF 및 profile CSV를 같은 폴더에 저장했다.
이번 새 실험 파일은 아직 commit/push하지 않았다. 사전 백업 커밋과 구분한다.
