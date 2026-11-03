Soft constraint mu10 장기 학습과 비교를 완료했다.
40개 모델: 전체손실/MSE/원소L1 공동 평탄 0개, 배정 시간상한 종료 40개.
종료 update 범위 132,000–212,000.
모든 모델은 기존soft10k Adam/iteration/RNG에서 이어갔다. mu10·전처리·입력분할·architecture·batch512·clip5·절대LR(.0001이후유지)는 바꾸지 않았다.
학습 순서는 ANN10→P-only10→T/P10→T/P/phi10이며, 평가·그림도 순차였다. 전체6시간 중40분을 평가/보고서에 유보했다.
평탄은 전체train관측1k, 검사5k, 최소50k, 연속20k창의q25/median/q75상대변화1%/1%/3%,3회연속10k지속이다. 전체손실·MSE·원소L1 모두 동시에 통과해야 한다.
테스트는 종료점 선택이나 가중치 조정에 사용하지 않았다. 미도달 상한 종료는 수렴으로 표시하지 않았다.
기존loss-only signed BoxCox 역변환 확장과 엄격한 실제추론 도메인, raw/repaired규칙을 유지했다. 새로운penalty나ODEfallback은 없다.

그림7행×10변수의4개선: 검정Cantera, 회색기존무제약10k, 파랑soft10k, 주황soft장기. mixed팀은 각단계의ANN과DeepONet을 모두 사용한다.
각그림 바로뒤에 같은IC의NRMSE표를 배치했다. raw/repaired각각36페이지, 전체18testIC를 보존했다.
NRMSE=RMSE(Cantera,NN)/RMS(Cantera), 첨부util.py기본값과동일한무차원비율이며 모든시간점에같은가중치를준다.
미완주에는전체NRMSE를부여하지않는다. 전체요약은21개모델구성이모두완주한동일IC집합의중앙값이다. 빈교집합이면N/A로표시하고완주율을함께보인다.
Repaired 공통 IC: [5, 7, 39, 44, 53, 64, 69]
Raw 공통 IC: []
보존은repaired후처리에서강제된다. 학습된보존성은별도member진단과raw결과로확인해야한다.
기존결과파일 24158개 SHA256불변.40개endpoint와공동평탄검사 938개를재검산했다.

[Repaired 그림+NRMSE](repaired/profiles_and_NRMSE.pdf) · [요약표](repaired/NRMSE_summary.pdf)
[Raw 그림+NRMSE](raw/profiles_and_NRMSE.pdf) · [요약표](raw/NRMSE_summary.pdf)
[전체loss성분](training_loss_components.pdf) · [개별member진단](member_loss_components.csv)
새결과는현재클라우드저장상태이며이번장기실험에대한commit/push는하지않았다.
