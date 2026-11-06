# 일반 ANN 단일 스텝 / 다중 스텝 비교

모델은 입력11 → tanh64 → tanh64 → tanh64 → 출력10의 일반 MLP(9,738 parameters)이다. branch/trunk 또는 시간 입력은 없다.
기존 8,815개 학습 시작점, train/val/test=54/18/18 IC, gamma=0.2, 학습 자료에만 맞춘 기존 정규화를 사용했다. 원본 데이터를 바꾸지 않았다.
단기: 각 방식 seed 11/22/33을 처음부터 각각 활성 학습150초. 단일 스텝은 K1, 다중 스텝은 시간의 각1/4을 K1/4/16/64에 배정했다.
장기: 단기 결과 게시를 원격 bw에서 확인한 후 정확한 가중치와 Adam 상태를 이어 모델당 최대150초 추가한다. 다중 스텝은 K64 유지. LR 하한1e-6에서 고정 training probe가 6회 더 개선되지 않을 때만 평탄 종료이며, 나머지는 시간상한 종료다.
soft constraint 및 학습 중 물리 보정은 없다. loss는 변환·정규화 상태 오차의 평균이고, 다중 스텝은 자기 예측을 되먹이며 모든 중간 시점에 역전파한다.
모든 검증/테스트 예측은2us씩5,000번 자기 예측을 되먹여10ms까지 진행한다. seed별 결과와 물리 상태를 평균한 mean3를 구분한다.
raw에는 clipping/원소투영/엔탈피 온도보정/유효 후보 필터링이 없다. repaired는 이 후처리 시스템을 사용하며 soft constraint와 구분한다.
NRMSE=RMSE(Cantera,NN)/RMS(Cantera), 무차원 비율. 불완주에는 전체 구간 점수를 주지 않는다. joint9는 T+8반응종 모두NRMSE<=0.1이고 N2는 별도 표에 포함한다.
완주는 유한값, 비음수 조성, 200~6000K 조건이며 정확도나 보존법칙 만족을 뜻하지 않는다. 초기0.1ms 점수도 원본 평가 표에 포함했다.
방법 선택은 검증 raw 완주수, joint9 통과수, worst9 중앙값 순으로 테스트를 읽기 전에 동결한다. gamma 최적화는 수행하지 않았다.
PDF는18개 테스트 각각2x10칸이고 장기 PDF에는 Cantera/단기/장기 3개 선을 함께 그렸다. 개별 그림 축 범위는 표시된 시간 창 안의 값으로 잡았다.

| Method | Mode | Completed | joint9<=0.1 | Median worst9 among completed |
|---|---|---:|---:|---:|
| multistep | raw | 1/18 | 0/18 | 62.6881 |
| multistep | repaired | 13/18 | 0/18 | 28.6281 |
| single_step | raw | 0/18 | 0/18 | nan |
| single_step | repaired | 18/18 | 0/18 | 6.43099 |
