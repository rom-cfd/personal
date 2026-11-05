6개gamma,4개계열,10개seed의240개모델을 pure MSE로 장기학습했다. 물리soft constraint는없다.
훈련MSE 평탄0/240, 시간상한240/240. 모델별새활성계산150초이며gamma.1 ANN/T-P20개의필수10k재생성시간도포함했다.
나머지220개는기존Adam/RNG를정확히이어받았다. gamma.1 ANN/T-P20개는역사적가중치의exact continuation이아닌재생성10k와짝지은continuation이다.
gamma01_historical_vs_paired_short.csv에역사적gamma.1과새paired-short의차이를명시했다. 이전결과는보존했다.
평탄기준은최소50k, 관측1k/검사5k, 연속20k창q25/median/q75상대변화1%/1%/3%,10k지속이다. 추가update상한과LR리셋은없다.
개별150초를bootstrap/continuation방문간누적하며대기/평가시간은제외한다. 재개는새150초를부여하지않는다. 실제시간/update/종료사유를수록했다.
각IC마다short7x10+NRMSE표,long7x10+NRMSE표를연속배치했다. 두stage의변수열y축범위는같다. 각그림은Cantera+6개gamma곡선이다.
NRMSE=RMSE(Cantera,NN)/RMS(Cantera), user util.py기본값,동일시간점가중치,무차원비율. 미완주는전체점수없음,빈공통집합은N/A다.
전체/팀내요약은short+long이공통완주한IC이며,same_gamma_short_long_summary.csv는각gamma/팀의paired공통IC를별도로쓴다.
long gamma 선택은validation repaired완주수우선,동률공통IC온도NRMSE로고정한뒤test를평가했다. 선택gamma의동일gamma short/long test비교도제공한다.
기존에관찰한test의탐색적추가학습평가다. 모든종에대한단일최적gamma 또는수렴을검증했다고표현하지않는다.
240개실제Adam update,평탄검사5145개와기존파일58917개의SHA보존을검증했다.

[Test repaired short/long 그림+표](test/repaired/profiles_and_NRMSE.pdf)
[Test raw short/long 그림+표](test/raw/profiles_and_NRMSE.pdf)
[팀별gamma short/long NRMSE](test/repaired/within_team_NRMSE.pdf)

