2026-11-02 16시대 KST에 실제 OpenAI 클라우드 실행 환경에서 인계 점검을 완료했다. 이번 작업은 환경·저장 체크포인트·초기조건 첫 스텝 검증과 저장 이력 그림 재생성까지다. 연구 모델을 학습하거나 10 ms 전체 성능 평가를 실행하지 않았다. 후속 연구 코드의 재구현 범위와 계산 예산은 아래의 제안이며 아직 실행되지 않았다.

| 항목 | 이번에 직접 확인한 상태 |
|---|---|
| 저장소 | `/workspace/personal`, origin `https://github.com/rom-cfd/personal.git` |
| 시작 체크아웃 | `work`, `e89c656eeeac0e93bafb4322b9482c7dfec70168`, 미커밋 변경 없음 |
| 최종 체크아웃 | 기존 로컬 `bw`, upstream `origin/bw` |
| 로컬·원격 bw 전체 SHA | `3affe7545a3946075a7f80547d70cb6147c15c06`, ahead/behind 0/0 |
| 원격 확인 | HTTPS 프록시를 통한 `git ls-remote --heads origin bw`와 bw 한정 fetch 성공 |
| 보존 | `work` 브랜치 보존, reset/추가 clone/worktree/commit/push/merge/삭제 없음 |
| 환경 설정 | 환경 초안·게시 snapshot 변경 없음. GitHub 상태와 게시 snapshot은 별개 |
| Python | `/workspace/kinetics-venv/bin/python`, Python 3.12.14 |
| 라이브러리 | Cantera 3.2.0, TensorFlow CPU 2.20.0, Keras 3.15.1, NumPy 2.5.4, SciPy 1.16.3, pandas 2.3.3, Matplotlib 3.10.7, pytest 8.4.2 |
| 의존성 | bw의 requirements.lock.txt 모든 버전과 설치 버전 일치. 추가 설치 없음 |
| CPU | Intel Xeon Platinum 8573C, 보이는 논리 CPU 5개, cgroup cpu.max=400000/100000로 유효 CPU quota 4개 |
| 메모리·디스크 | cgroup 메모리 상한 16 GiB, swap 없음, /workspace 사용 가능 디스크 약 30 GiB |
| 실행 스레드 | TF intra=2, inter=1; OMP/OpenBLAS/MKL=2; GPU 없음; CUDA_VISIBLE_DEVICES=-1 |
| oneDNN | 본 baseline은 프로세스 시작 전 TF_ENABLE_ONEDNN_OPTS=0. 별도 진단에서만 1 사용 |

첫 원격 조회는 기본 명령 sandbox의 네트워크 격리로 프록시에 연결하지 못했다. 같은 HTTPS 프록시를 유지하고 명령의 네트워크 접근을 허용한 뒤 성공했다. 인증·시스템 네트워크·영속 환경 설정을 변경하지 않았다. 원래 fetch refspec에는 main만 있어 bw 추적 설정이 한 번 실패했으며, bw 한정 refspec을 추가한 후 정상 설정했다. README/AGENTS의 kisung 지침보다 이번 사용자의 명시적 bw 선택을 적용했다.

`pip_check.log`: **No broken requirements found.** 캐시 디렉터리 쓰기 권한 경고로 pip 캐시는 비활성화됐지만 패키지 오류는 없다. `pytest.log`: **71 passed, 39 warnings in 39.67s**. 이는 이번 실행값이다. 경고는 Keras의 NumPy `__array__(copy=...)` 관련 DeprecationWarning이다. 이전 설치 헬퍼를 실행하지 않았다.

다음 세 파일은 인계 기준 SHA256과 정확히 일치했다.

| 파일 | SHA256 |
|---|---|
| training_pairs.npz | `44db17b16e167ea389631dacb8015528f76cbd5c2072e9fb919701e54d50fa3e` |
| trajectories.npz | `8d41f19f2145e12f9e89ee3eaa88fb0b1b93a84dcaaee65dcaa2e8b264077563` |
| requirements.lock.txt | `5f05f78490963bb0b00e20866a50d7bf69ec684119ccadba0fca30a2967b45c9` |

`baseline_verified/baseline.json`과 `verified_models.csv`에 데이터·모델·전처리 검증 근거를 저장했다. ANN 100개 및 T/P형 DeepONet 10개의 .keras 해시를 manifest와 대조했다. 모든 멤버는 해당 8,815쌍 데이터 해시와 10,000 update를 기록하고 있다. 전처리 JSON은 현재 Git 체크포인트의 바이트와 대조했고, ANN100/T/P10의 실제 변환·표준화 값이 같다. 비교용 첫 ANN10의 전처리는 train pair만으로 재적합한 값과 배열 단위로 정확히 일치했다. ANN의 spec에만 있는 표기용 `name`은 변환 조건이 아니므로 kind/gamma/species 옵션과 epsilon·scaler를 비교했다. 초기 감사 스크립트가 이 표기 차이까지 동일하게 요구해 중단한 로그는 `baseline.log`에 보존했고, 수정 후 검증은 `baseline_verified/`에 별도로 저장했다.

저장된 T/P 실행의 config·fingerprint 전체가 현재 재검증값과 정확히 일치한다. 실제로 불러온 ANN10의 파라미터 수는 각각 9,738, T/P10은 각각 9,687이다. 110개 .keras의 weights archive 모두 optimizer 그룹이 없다. 이전 Adam slots/iteration·RNG·평탄 monitor는 복원되지 않았다. 저장 가중치 사용은 가능하지만 정확한 optimizer continuation은 불가능하다.

데이터는 `(90, 5001, 10)`, train/val/test=54/18/18, dt=2e-6 s, 10 ms horizon이며 모든 pair의 현재·정답·P를 원본 parent `(IC,j)`와 `(IC,j+1)`에 정확히 대조했다. 실제 메커니즘은 h2o2.yaml에서 N2를 포함한 지정 9종과 해당 종만 사용하는 28반응을 추출한다. 생성 코드는 IdealGasConstPressureReactor로 0-D 단열·정압 폐쇄 배치 반응기를 계산한다. 입력은 현재 T,Y9,P, 출력은 2 μs 뒤 T,Y9다.

Box–Cox는 코드상 `Y**0.1 / 0.1`이고 역변환은 `(0.1*z)**10`이다. 일반 Box–Cox 식과 상수 차이가 있으나 표준화에서 상쇄된다. epsilon=1e-8은 저장돼 있지만 이 boxcox 분기에서는 Y에 더하거나 floor로 쓰지 않는다. T는 T/1000, P는 log(P)이고 output scaler는 train 입력·정답의 변환 상태를 합쳐 적합한다. 표준화된 현재 상태에 잔차를 더한다. 이 수식과 fit 범위를 후속 구현에서 유지해야 한다.

P-only/continuation/followup/time-profile 관련 지정 7개 경로는 모두 없었다. /workspace의 프로젝트·공유·onboarding 경로 이름을 조사했으나 추가 복구 후보도 없었다(`recovery_path_search.json`). virtualenv, Git 내부, 자격증명/설정 디렉터리, 첨부물, 이번 감사 출력은 검색에서 제외했다. 접속 불가한 이전 실행 공간의 보존 여부는 여기서 확인할 수 없다. 과거 seed11=244k/seed22=159k라는 기록으로 가중치나 optimizer 상태가 복구됐다고 볼 수 없다.

이전 medoid100 첫 스텝 불일치 원인을 별도 프로세스로 조사했다. 데이터·가중치·스레드·코드를 고정하고 TF_ENABLE_ONEDNN_OPTS만 변경했다. medoid는 이 과거 검증 오류의 진단에만 사용했다.

| oneDNN 설정 | 18 IC 최대 T 차이 | 최대 Y 차이 | 저장 선택 seed | 원래 검사 결과 |
|---|---:|---:|---|---|
| 0, TensorFlow import 이전 설정 | 0 K | 0 | 전부 동일 | T와 종 모두 통과 |
| 1, TensorFlow import 이전 설정 | 3.3531373446749058e-6 K | 2.8239814653385764e-9 | 전부 동일 | T와 종 모두 실패 |

oneDNN=1에서 T 실패는 배열 행 17, IC88이며 이전 설치 로그의 값과 정확히 같은 차이가 재현됐다. 이전 헬퍼는 TensorFlow를 먼저 import한 뒤 kinetics를 불러왔고, study.py의 oneDNN 기본 비활성화가 그 시점에는 늦었다. oneDNN=0에서는 더 엄격한 저장소 verifier의 rtol=1e-9/atol=1e-10 검사까지 값의 차이 없이 통과했다. 헬퍼의 T rtol=1e-9/atol=1e-6, 종 rtol=1e-8/atol=1e-10도 각각 평가했으며 어느 허용오차도 수정하지 않았다. 이것은 첫 스텝의 실행 설정 차이 규명이지 전체 10 ms 궤적 재현 완료가 아니다.

ANN10/T/P10/mixed20의 raw·repaired를 test 초기조건 18개의 동일 입력에서 각각 호출했다. 모든 팀에서 첫 출력 18행과 모든 후보가 유한했다(10/10 또는 20/20). ANN10 repaired는 기존 mean10 trace의 첫 출력과 T·Y9 모두 정확히 일치한다. T/P와 mixed의 저장 비교 trace는 없으므로 이 두 팀에 대해 과거 성능 재현을 주장하지 않는다. 결과는 `baseline_verified/initial_condition_first_steps.npz`에 있다. Dense 90,000 pair one-step 오차나 5,000단계 완주율은 이번에 계산하지 않았다.

물리 유효 기준은 원본을 유지한다. mixed_ensemble.py의 repaired 순서는 멤버 잔차 → 변환 좌표 Y를 [0, forward(1)]로 guard → inverse decode → VectorThermo의 비음수/원소 보정 → 현재 상태의 엔탈피·정압을 만족하는 온도 복원 → 유한한 후보 행만 같은 가중치로 평균 → 평균 재보정이다. project_batch는 음수 판정 -1e-14, 원소 잔차 <1e-10을 사용하고 불안정한 경우 기존 scalar projection을 호출한다. NASA 온도 root는 200–6000 K 범위와 엔탈피 잔차 1e-5 J/kg 조건을 사용하며 원본 scalar HPY thermodynamics fallback을 유지한다. 이는 화학 ODE fallback이 아니다. 각 후보는 10개 성분 모두 유한할 때만 유효하고, 한 성분이라도 실패하면 행 전체를 제외한다. valid_count와 valid_ann/valid_deeponet를 기록하므로 제외 후 계열 비중이 달라지는 것이 드러난다. 전부 무효면 NaN이다. 최종 rollout은 유한성, T 200–6000 K, Y 비음수, 양의 조성합을 요구하고 실패한 스텝을 trace에 붙이지 않는다. raw는 guard·repair 없이 모든 멤버가 유한해야 한다. 엔트로피는 진단이며 후보 수용 기준이 아니다.

저장 ANN/T/P의 학습 이력으로 `history_figures/training_convergence.png`, `.pdf`, `training_statistics.csv`를 만들었다. 두 계열 모두 관측된 공통 update 500,1000,…,10000의 20시점만 비교했다. 실선 geometric mean, 파선 max, 점선 min, min–max 음영이며 신뢰구간이 아니다. 표시 loss는 mini-batch residual MSE다. 전체 train loss, validation loss 또는 rollout 성능으로 해석하지 않는다. LR은 원래 고정 6000/9000 경계를 사용하며 ReduceLROnPlateau가 아니다. 기존 T/P full-train MSE 보고서는 그대로 보존했다.

이번 baseline의 공통 ANN10/T/P10 모델 로딩은 1.713초, 감사 본체는 9.502초였다. 감사 본체 시간은 함수 진입 전 라이브러리 import를 제외한다. 각 팀별 loading/JIT warm-up과 첫 호출 이후 neural+decode/candidate repair/mean repair 시간은 baseline.json에 각각 있다. 기존 코드가 neural과 decode를 합산하므로 두 값을 이번 결과에서 분리했다고 주장하지 않는다. 첫 스텝 한 번의 시간은 정식 추론 속도 벤치마크가 아니다. 과거 ANN grouped 학습시간을 나눠 새 순차 학습의 속도비로 사용하지 않았다.

후속 재구현은 다음 범위로 제안한다. 기존 kinetics/*.py와 원본 config/result는 변경하지 않고 새 모듈·config·namespace에서 구현한다.

1. P-only 모델: branch=current T,Y9 → 64–64 → 10×7, trunk=log(P) →32→7, 성분별 product+bias. 파라미터 수는 산식상 9,719이나 실제 모델 생성·gradient·partition 테스트는 후속 구현에서 확인해야 한다. 공통 전처리를 재사용하며 T/P 원본 동작을 보존한다.
2. 재개 가능한 순차 trainer: 가중치, Adam slots/iteration, NumPy minibatch RNG 및 사용한 프레임워크 RNG, update, LR, history, plateau monitor, 누적 사용시간, 설정·데이터·코드 hash를 하나의 등록된 체크포인트로 저장한다. 부분 저장은 완료 marker로 구분하고 기존 산출물을 덮어쓰지 않는다. 끊김 없는 실행과 저장·새 프로세스 복원의 다음 minibatch/Adam iteration/weight를 비교한다. 코드/data/config 불일치도 거부한다.
3. LR 경계는 절대 update 6000/9000/10000으로 명문화하고 이후 1e-4를 유지한다. 기존 train_member에 updates=50000을 넘기면 경계가 30000/45000으로 이동하므로 이를 연장 학습기로 그대로 쓰면 안 된다. 저장 T/P10k는 optimizer가 없어 weights-only warm start로 명시하거나 새 trainer에서 0→10k→50k를 다시 학습하는 재현을 선택해야 한다. 권장 재현은 상태를 보존하는 새 0→50k 실행이며 별도 예산 대상이다.
4. 기존 mean 동작을 재사용하는 five-team 평가: ANN10, T/P10, P-only10, ANN+T/P20, ANN+P-only20. 팀별 자기 상태, 후보 행 제외, 계열별 유효 수, 실패 즉시 종료를 검사한다. raw/repaired 및 val/test를 분리하고 기존 ignition reference/보존·엔트로피 정의를 유지한다. loading/JIT, NN, decode, candidate repair, mean repair, one-step와 rollout 총시간을 별도로 계측한다.
5. 그림: 전체 18 test IC 각각 T+9종, Cantera 포함 6곡선, 실패선은 prefix에서 종료. PNG/PDF, CSV, 재현 .py, 완료율·공통 완주 IC 비교, pooled와 IC별 평균 집계식을 함께 저장한다. 장기 전후는 별도로 비교한다. test는 이미 관찰한 평가 집합이며 종료/설계 선택에 사용하지 않는다.

제안 계산 예산은 단일 실행 wall time 상한이며 CPU-hour 수치가 아니다. 모든 학습·평가·벤치마크·그림은 순차로 실행한다. 아래 수치는 예약하거나 실행한 예산이 아니며 사용자와 합의할 후보다.

| 순서 | 새 계산·산출물 | 제안 상한 | 근거·제약 |
|---|---|---:|---|
| B | 새 모델/trainer/평가 코드와 synthetic gradient·독립 RNG·실제 재개 테스트 | 테스트 3분 | 연구 데이터 장기 학습 없음. 구현 인력 시간과 계산 상한은 별개 |
| C0 | 1멤버 2k update 이하 + train/validation의 짧은 속도 측정 | 2분 | 장기 시간 예상의 보정용. test로 설정 선택하지 않음. 이것도 신규 연구 학습이라 사전 예산 합의 대상 |
| C1 | P-only 10×10k 학습, 매 1k 전체 train loss와 checkpoint | 학습 10분 | 과거 저장 T/P10×10k는 약210.57초였으나 새 P-only/CPU/저장 비용은 다름 |
| C2 | 5팀×2모드×2분할의 full rollout 및 dense one-step, 모든 IC 그림 | 평가 25분 + 그림 5분 | 각 분할 18×5000 전이, dense 90k pair. 총 20 team/mode/split 평가. 범위를 줄이거나 미완료면 그 사실을 기록 |
| D | 필요한 경우 새 T/P 0→50k 재현, 10k 체크포인트 포함 | 학습 25분 + 새 비교·그림 최대30분 | 총 500k member-updates. ANN은 재학습하지 않음. 중간91k–144k/최종151k–170k 전 구간 재학습은 일괄 승인 대상으로 넣지 않음 |
| E | P-only10k의 새 완전 상태에서 장기 학습, 평탄 조사 | 학습80분 + 체크포인트/기록10분, 평가·그림 별도30분 | 모델당 총250k update ceiling. 완료 보장이 아닌 시간 상한안. pilot 결과로 예산 재합의 권장 |

추천 다음 계산 단위는 B의 후속 코드 구현·재개 검증을 먼저 마친 뒤 C0으로 실제 속도를 확인하고, C1+C2에 최대40분을 따로 합의하는 것이다. 과거 T/P 학습시간으로 단순 환산하면 250k×10은 약86분의 학습량이고 진단/저장 오버헤드는 추가되므로, E의 학습80분으로 모든 멤버가 update ceiling 또는 평탄에 도달한다고 보장할 수 없다. 모델별 남은 시간 배분과 전체 누적 예산을 체크포인트에 저장하고 재개가 예산을 새로 시작하지 않도록 한다. 상한에 걸린 모델은 평탄 종료로 표기하지 않는다. 모든 멤버가 평가 가능한 종료 상태를 저장했는지 확인한 후 별도 평가한다. 아직 미완료인 멤버의 상태를 최종 장기 결과라고 합쳐 보고하지 않는다.

P-only 평탄 정책은 과거 설계를 코드로 명문화하는 안이다. 전체 train loss를 매1000 update 기록하고, 최소 총50k 이후 매5000 update 검사한다. 검사 update u에서 `(u-40000,u-20000]`와 `(u-20000,u]`의 두 20k 창을 비교하며 각 창은 20개의 실제 관측값을 요구한다. q25/median/q75 각각의 상대 변화 절댓값이 1%/1%/3% 이하여야 하고 u,u+5k,u+10k의 연속 세 검사에서 만족하면 평탄으로 종료한다. 손실이 0인 특이 경우의 상대변화 정의도 사전에 정해 테스트한다. 두 20k 창은 미니배치 변동 영향을 줄이고, 50k 최소량은 초기 급감 이후를 보며, 10k 유지 조건은 일시적 우연을 줄이기 위한 선택이다. 이 임계값은 수렴의 수학적 증명이 아니므로 전체 관측 곡선·창별 분위수·상대 변화·종료 사유를 함께 저장한다. 10k 이후 LR=1e-4 유지, test 값으로 정책·종료점·상한을 조정하지 않는다. 시간상한/update상한/평탄/비유한 실패를 구분한다. 이 정책과 예산은 후속 실행 전에 합의할 제안이다.

재현 명령은 저장소 루트에서 다음처럼 실행한다. 기존 산출물 보존을 위해 각 `--output`에는 존재하지 않는 새 경로를 지정한다.

```bash
cd /workspace/personal
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=-1
export MPLCONFIGDIR=/tmp/kinetics-mpl XDG_CACHE_HOME=/tmp/kinetics-cache
/workspace/kinetics-venv/bin/python -m pip check
/workspace/kinetics-venv/bin/python -m pytest -q
/workspace/kinetics-venv/bin/python scripts/audit_ann_deeponet_handoff.py --mode audit --output results/handoff_audit_20261102/run_002/baseline
/workspace/kinetics-venv/bin/python scripts/audit_ann_deeponet_handoff.py --mode medoid --output results/handoff_audit_20261102/run_002/medoid_off
TF_ENABLE_ONEDNN_OPTS=1 /workspace/kinetics-venv/bin/python scripts/audit_ann_deeponet_handoff.py --mode medoid --output results/handoff_audit_20261102/run_002/medoid_on
/workspace/kinetics-venv/bin/python scripts/audit_ann_deeponet_handoff.py --mode plots --output results/handoff_audit_20261102/run_002/history_figures
```

실행 후 기존 results의 843개 파일을 실행 전 SHA256과 다시 대조해 변경·누락 0개를 확인했다(`original_results_preserved.json`). 기존 추적 파일의 Git 차이도 0이다. 새 `scripts/audit_ann_deeponet_handoff.py`와 `results/handoff_audit_20261102/`는 현재 공간의 untracked 파일이며 Git commit·GitHub push·게시 snapshot에는 포함돼 있지 않다. 후속 연구 코드·P-only 모델·장기 학습·전체 10 ms 비교는 아직 구현하거나 실행하지 않았으므로, 소실된 결과의 복구나 재현 완료로 취급하지 않는다.
