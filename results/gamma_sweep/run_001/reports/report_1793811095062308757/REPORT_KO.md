Box-Cox gamma=0.05,0.1,0.2,0.3,0.5,1.0 비교. soft constraint 없이 정규화된 증분 MSE만 학습했다.
기존gamma.1의40개 모델/시험궤적을 재사용하고 나머지5개gamma의200개 모델을10000update 또는모델당150초상한으로 순차 학습했다.
10000회완료200/200, 모델시간상한0/200. 모델별실제update와종료사유를training_summary.csv에기록했다.
ANN10/P-only10/ANN+P20/TP10/ANN+TP20/TPphi10/ANN+TPphi20의7개 팀. 혼합팀의 두 계열은 같은gamma를 쓴다.
데이터8815쌍,2us target,10ms horizon,구조/seed/Adam/배치512/clip5/LR스케줄/추론보정은 고정했다.
각gamma에서 학습current+next 상태의 공통스케일, 학습current 입력스케일을 다시 맞췄다. validation/test로 스케일을 fit하지 않았다.
새gamma들의 초기가중치와미니배치RNG를 seed/계열별로 짝지었다. gamma.1의 과거 ANN grouped time은 신규 serial time과 별개다.
gamma마다 변환/목표척도가 달라 raw training MSE의 크기로 gamma 성능순위를 매기면 안 된다. 학습그래프는 초기MSE대비 비율이다.
모든 endpoint를 고정하고 validation을 먼저 평가했다. 대표gamma는 repaired 완주수 우선, 동률은공통IC 온도NRMSE 중앙값으로 선택했다.
validation_selection.json을 저장한 뒤test를 평가했다. 이미 과거에 관찰한test의 탐색적비교이며 독립최종검증을 주장하지 않는다.
NRMSE=RMSE(Cantera,NN)/RMS(Cantera), user util.py 기본값,동일시간점가중치,무차원비율. 미완주에는 전체점수를 부여하지 않는다.
각IC의7x10그림 바로 뒤에 같은구성의NRMSE표를 넣었다. 검정Cantera+6개gamma곡선이다.
전체요약은42개구성 공통완주IC, 팀별요약은해당팀6개gamma 공통완주IC를 쓴다. 빈집합은N/A이며 팀마다IC집합이 다를수있다.
test 각모드의 validation_selected_vs_reference.csv는 validation에서 고른gamma와.1의 공통완주IC에서10개변수를 비교한다.
신규200개실제Adam iteration과기록update일치/순수MSE/초기화40그룹 및기존파일48303개의SHA보존을검증했다.

[Test repaired 그림+표](test/repaired/profiles_and_NRMSE.pdf)
[Test raw 그림+표](test/raw/profiles_and_NRMSE.pdf)
[팀별gamma NRMSE](test/repaired/within_team_NRMSE.pdf)
[Validation 그림+표](val/repaired/profiles_and_NRMSE.pdf)

