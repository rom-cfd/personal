원소 soft constraint 가중치 lambda=1,100,1000의 matched10k 비교를 완료했다.
ANN/P-only/T-P/T-P-phi 각10개, 가중치별40개, 총120개를 순차로 새로10000회 학습했다.
lambda0/10은 기존10000회 모델과 평가를 재사용했다. 기존lambda10 장기 모델은 회차가 달라 이 비교에 넣지 않았다.
각seed의 초기화, minibatch stream, 데이터8815쌍, 2us target, 전처리, architecture, Adam/LR/clip 및 추론보정은 동일하다.
전체손실=MSE+lambda*원소L1. lambda가 다르므로 전체손실 수치만으로 가중치 간 성능을 비교하면 안 된다.
120개 종료점을 모두 고정한 뒤 기존18개 test IC에서 raw/repaired를 평가했다. 테스트는 종료점/가중치 선택에 쓰지 않았다.
이미 여러 번 확인한 테스트의 탐색적 가중치 비교이며, 최종 선택을 독립적으로 검증한 결과는 아니다.
그래프는7행×10변수, 검정Cantera와5개lambda 곡선이다. 각IC 그림 바로 뒤에 같은NRMSE표를 넣었다.
NRMSE는util.py기본값 RMSE(Cantera,NN)/RMS(Cantera), 동일 시간점 가중치, 무차원 비율이다.
미완주는전체NRMSE없음. 전체요약은35개구성 공통완주IC 중앙값이며 빈집합은N/A다.
별도 within-team 표는 각팀 내5개가중치가 모두완주한IC만 사용한다. 팀마다IC가 다르므로 architecture 간 순위로 쓰면 안 된다.
Repaired 공통IC: [44, 53]
Raw 공통IC: []
기존결과 33350개 파일을 보존했고, 새120개 실제Adam iteration과전처리를 검증했다.
추론에서 강제한 보존성을 학습된 보존성으로 해석하지 않는다. 개별member 보존진단과raw완주율을 함께 확인한다.

[Repaired 그림+표](repaired/profiles_and_NRMSE.pdf) · [요약](repaired/NRMSE_summary.pdf)
[Raw 그림+표](raw/profiles_and_NRMSE.pdf) · [팀별가중치비교](repaired/within_team_NRMSE.pdf)
[손실성분](training_loss_components.pdf)
