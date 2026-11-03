test 18개 초기조건, IC별 한 페이지의7행×10열 비교 그림입니다.
열: T,H2,H,O,O2,OH,H2O,HO2,H2O2,N2. T는 K, 종 변수는 질량분율 Y입니다.
행1 ANN10 / 행2 P-only / 행3 ANN+P-only / 행4 T/P / 행5 ANN+T/P / 행6 T/P/phi / 행7 ANN+T/P/phi.
검정 점선=Cantera, 파랑=10,000 update, 주황=장기 학습. 첫 행은 Cantera+ANN의2개 선입니다.
ANN은 모든 혼합에서 기존10k를 유지하고 DeepONet만 단기/장기를 비교합니다.
장기 모델당 update 범위: {'P-only': [338000, 438000], 'T/P': [263000, 359000], 'T/P/phi': [467000, 825000]}. 모두 평탄 미도달 시간상한 종료입니다.
모두 repaired 자기 예측이며, 기존 동일 물리 보정 및 후보 평균 규칙을 사용했습니다.
동일 IC 내 같은 열은 동일한 선형 y축을 씁니다. 열 사이 단위/범위는 다릅니다.
중간 실패 trace는 마지막 정상점에서 종료하며 x로 표시합니다. 미래 값을 채우지 않았습니다.
N2가 거의 일정할 때 약±2% 여백을 주어 기계정밀도 변화의 과도한 확대를 막았습니다.
기존 source/model/result는 수정하지 않았고, 누락된4개10k 팀의 rollout만 순차 추론했습니다. 새 학습은 없습니다.
PDF는18페이지 벡터 그림, png 폴더는IC별5580×3636 고해상도 이미지입니다.
재현: scripts/plot_kinetics_7x10_comparison.py --phase plot --output 새경로
