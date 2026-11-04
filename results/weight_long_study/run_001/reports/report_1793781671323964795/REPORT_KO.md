lambda1,100,1000의120개 모델을 기존10000회 체크포인트에서 이어 학습했다.
공동 평탄 0개, 배정 시간상한 종료 120개. 전체상한6시간 중1시간을 평가/그림에 유보했다.
학습/평가/그림은 순차, CPU intra2/inter1 및 OMP/BLAS2/oneDNN0을 유지했다.
가중치·데이터·전처리·구조·Adam/RNG·절대LR(.0001이후유지)은 그대로며, update상한을 추가하지 않았다.
공동 평탄: 최소50k, 관측1k/검사5k, 연속20k창의q25/median/q75상대변화1%/1%/3%,3회 연속10k지속을 전체/MSE/원소L1 모두 통과해야 한다.
미도달 시간상한을 수렴으로 표시하지 않았다. 모델별 종료update와 실제시간은각 training_summary.csv에 있다.
가중치별7x10 그림은Cantera/무제약10k/동일lambda soft10k/soft장기 네 곡선이다. 각IC 그림 뒤에 같은NRMSE표를 넣었다.
NRMSE=RMSE(Cantera,NN)/RMS(Cantera), util.py기본값·동일샘플가중치·무차원비율. 미완주에는전체점수를 부여하지 않는다.
각요약은명시된공통완주IC 중앙값이다. 별도 within-team 표는각팀내공통IC만쓰며 팀마다집합이다를수있어 architecture순위로쓰면안된다.
cross_weight 표는이번 세lambda의short/long42구성을비교한다. 장기endpoint update가다르므로 완전수렴 후동일회차lambda순위로해석하면안된다.
기존lambda10장기는더큰모델당시간과다른update를사용했으므로이번주요lambda비교에합치지않았다. 이전보고서는원위치에보존했다.
이테스트IC들은이미이전실험에서관찰했다. 본비교는탐색적민감도/추가학습평가다. 테스트로학습종료점이나lambda를조정하지않았다.
기존결과파일39020개SHA불변,120개endpoint와공동평탄검사241개를검증했다.

lambda1: update41000–81000, 공동평탄0/40.
[lambda1 repaired 그림+표](lambda_1/repaired/profiles_and_NRMSE.pdf)
[lambda1 raw 그림+표](lambda_1/raw/profiles_and_NRMSE.pdf)
lambda100: update35000–84000, 공동평탄0/40.
[lambda100 repaired 그림+표](lambda_100/repaired/profiles_and_NRMSE.pdf)
[lambda100 raw 그림+표](lambda_100/raw/profiles_and_NRMSE.pdf)
lambda1000: update29000–62000, 공동평탄0/40.
[lambda1000 repaired 그림+표](lambda_1000/repaired/profiles_and_NRMSE.pdf)
[lambda1000 raw 그림+표](lambda_1000/raw/profiles_and_NRMSE.pdf)
