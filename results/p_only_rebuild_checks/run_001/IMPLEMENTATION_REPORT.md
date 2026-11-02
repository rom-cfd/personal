P-only 재구현과 검증을 2026-11-02 KST에 완료했다. 작업 브랜치는 bw이며 HEAD는 `3affe7545a3946075a7f80547d70cb6147c15c06`, upstream은 origin/bw다. 새 코드는 아직 untracked이며 commit/push하지 않았다.

| 항목 | 실제 확인값 |
|---|---|
| 구현 | kinetics/p_only_deeponet.py의 PressureDeepONet |
| branch | 현재 T·Y9 →64–64 tanh→70 coefficients→10×7 |
| trunk | log(P)만→32 tanh→7 basis |
| 출력 | 성분별 내적+bias, 공통 변환 좌표 잔차10개, dt=2 μs |
| 파라미터 | 9,719개 |
| 전처리 | 기존 ANN/T/P와 학습 features·targets 및 train-only scaler 정확히 일치 |
| 초기 모델 | results/p_only_rebuild/run_001/initial_untrained_seed11.keras와 .json |
| 연구 학습량 | 0 update, 초기 모델은 학습 전 상태 |
| 실제 데이터 준비 | 동일 8,815 train pairs, 90×5001 trajectory, 54/18/18 분할 검증 |
| 전체 테스트 | 87 passed, 66 warnings in 46.64s; 신규 테스트16개 포함 |
| 경고 | Keras/NumPy __array__(copy=...) DeprecationWarning |
| 재개 검증 | 새 Python 프로세스 복원 후 가중치·Adam 배열·다음 minibatch가 연속 실행과 array_equal |
| 원본 보존 | 기존 결과843개와 앞선 감사 산출물25개 모두 SHA256 불변, 기존 추적 파일 Git 차이0 |

학습 실행기는 별도 모듈 kinetics/p_only_training.py와 kinetics/p_only_study.py에 추가했다. 기본 prepare는 학습하지 않는다. 준비 후 --resume 검증도 실행해 같은 provenance에서 기존 산출물을 덮어쓰지 않고 재사용함을 확인했다.

학습 시에는 독립 seed 10개, 각10k update, batch512 복원추출, Adam/global-norm clip5, 절대 update 6000/9000 LR 경계를 유지한다. 모델·전처리·Adam slots/iteration·멤버별 NumPy RNG·TF Generator·update·LR·mini-batch/full-train history·경과시간·비활성 평탄 monitor·config/data/source/runtime hash를 저장한다. 체크포인트는 새 디렉터리에 기록하고 완료 hash marker를 마지막에 남긴다. 잘못된 데이터/config/해시/optimizer iteration은 거부한다. 최초 학습의 절대 만료시각을 고정하므로 재개가 시간 예산을 초기화하지 않는다. 이 보수적 시간 창에는 중단·대기 시간도 포함한다.

연구 학습을 시작하기 전 상태의 보고서다. 사용자 인계문에 새 학습의 예산 합의 조건이 있어 P-only 10개×10k의 최대10분 학습 여부를 확인했다. 이 보고서의 테스트 학습은 synthetic 데이터의 소수 update이며, 연구 데이터 모델을 학습한 결과가 아니다. 장기 평탄 정책·전체 validation/test 평가·그래프 비교는 이번 구현 범위에 포함하지 않았다.

상세 명령과 사용법은 저장소의 P_ONLY_REBUILD_WORKFLOW.md, 설정은 configs/p_only_rebuild.json, 검증 수치·소스 해시는 이 디렉터리의 verification.json, 전체 테스트 로그는 full_pytest.log에 있다. 기존 T/P·ANN 코드는 수정하지 않았다.
