Soft constraint mu10,10000 update 비교를 완료했다.
ANN / P-only DeepONet / T/P DeepONet / T/P/phi DeepONet 각각10개, 총40개를 초기화부터 순차 학습했다.
각 모델10000 update, 합계400000 update. 장기 학습이나 평탄 종료를 수행한 실험이 아니다.
기존 첫10개 ANN과 각 DeepONet10k를 baseline으로 재사용했다. 새 mixed팀은 soft ANN10과 soft DeepONet10을 함께 사용한다.

Loss = 변환 좌표 잔차 MSE +10×mean(sum(abs(A@Y_pred − A@Y_true))). A는 H/O/N 원소 질량분율 행렬이다.
첨부 step2의 물리 단위 원소 L1 아이디어와 mu10을 현재 다음 상태 예측에 적용했다. Methane53종·정규화·출력 반응률 문제를 가져오지는 않았다.
데이터·분할·8815 train쌍·고정2us target·BoxCox gamma.1·train-only 표준화·입력분할·네트워크 폭·batch512·Adam·LR경계6000/9000·clip5를 유지했다.
물리 penalty는 표준화 잔차에 현재 상태를 더한 뒤 역표준화/역BoxCox로 복원한 다음 상태 Y9에 계산한다. T는 원소항에 넣지 않는다.
Loss 계산에는 signed inverse 확장을 사용해 음수 변환 예측에서도 유한 gradient를 유지했다. 유효 도메인에서는 기존 역변환과 같다. 실제 추론은 기존 엄격한 역변환/실패 규칙을 유지한다.
새 전처리 floor, 농도 클리핑, positivity/energy/entropy penalty, 증강, ODE fallback 또는 teacher forcing을 추가하지 않았다.
Train/validation loss 및 물리 원소 L1/역변환 도메인 위반율을 baseline과 soft 각각40개에서 비교했다. 종료점 고정 후 test18 IC만 rollout/one-step 평가했다.
Test는 이전에 관찰된 집합이며 모델선택·mu조정·종료에 사용하지 않았다.

그래프는7행×10변수, 검정Cantera/파랑baseline10k/주황soft10k다. 첫ANN행도이번에는baseline/soft를 포함한3개 선이다.
각 그림 바로 다음 페이지에 같은 IC의 NRMSE 표를 배치했다. repaired/raw 각각36페이지다.
NRMSE는 사용자 util.py의 sqrt(mean((Cantera−NN)^2))/sqrt(mean(Cantera^2)), 무차원 비율, 시간점 동일 가중치다.
미완주는 full-horizon NRMSE를 비웠다. 전체 요약은14팀 모두가 완주한 동일 IC 집합의 중앙값이며, 평균/표준편차도CSV에 저장했다.
Raw 공통 완주가0이면 공정한 전체 요약을 만들 수 없어 N/A로 표시한다. 개별 완주 IC와 실패 시점은 per_IC CSV에 남긴다.
Repaired 원소/엔탈피 보존은 후처리로 강제된다. Soft의 학습 효과는 별도 비보정 member 진단 및 raw와 함께 판단해야 한다.

기존 결과 SHA256 불변 22186개. 최종 모델/Adam/RNG 체크포인트 40개 검증.
Repaired 공통완주 IC: [5, 7, 39, 44, 53, 64, 68, 69, 73]
Raw 공통완주 IC: []

[Repaired 그림+NRMSE](repaired/profiles_and_NRMSE.pdf) · [Repaired 요약](repaired/NRMSE_summary.pdf)
[Raw 그림+NRMSE](raw/profiles_and_NRMSE.pdf) · [Raw 요약](raw/NRMSE_summary.pdf)
[학습 loss 성분](training_loss_components.pdf) · [Baseline/soft member 진단](mean_member_loss_components.csv)
이번 새 코드·모델·결과는 현재 클라우드 저장 상태이며 별도commit/push는 하지 않았다.
