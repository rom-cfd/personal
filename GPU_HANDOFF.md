# GPU에서 1000 epoch 이어학습

**전달 상태:** GPU 실행 코드와 10 epoch 비교 결과는 공개했지만, 학습 배열과
Adam 체크포인트를 담은 분할 패키지는 아직 업로드하지 않았다. 로컬에서
복원·재개 검증은 완료됐다. 아래 설치를 준비할 수 있으며, `restore`와
이어학습 명령은 전달 패키지를 업로드한 뒤 사용할 수 있다.

CPU 학습을 멈췄다. 저장 지점은 **10.676197836 epoch**, Adam update
**110,520회**다. 이 중 기존 10 epoch 이후 추가 update는 7,000회다.
목표는 **총 1000 epoch**, batch **512**, gamma=.1, seed11,
11–1600–800–400–11 GELU다. 처음부터 다시 학습하지 않는다.

## 실행

Linux 또는 WSL2의 NVIDIA GPU/드라이버와 Python 3.11 또는 3.12가 있는
환경에서 새 가상환경을 사용한다.
기존 `requirements.lock.txt`는 CPU TensorFlow를 포함하므로 GPU 설치에는
아래 파일을 사용한다. 설치되는 CUDA 사용자 라이브러리는 pip가 선택한다.

```bash
git clone --branch pre-process https://github.com/rom-cfd/personal.git
cd personal
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.gpu.txt
python -m kinetics.gpu_handoff restore
python -m kinetics.gpu_resume --check
python -m kinetics.gpu_resume
```

`--check`는 실제 TensorFlow GPU 인식과 전달 파일을 검사한다. GPU가 없으면
CPU 학습을 시작하지 않고 오류를 낸다. GPU 메모리는 필요한 만큼 확장하도록
설정한다. 학습 중단 후에는 저장된 체크포인트로 같은 명령을 다시 실행한다.

```bash
python -m kinetics.gpu_resume
```

최초 동작을 짧게 확인하려면 `python -m kinetics.gpu_resume --run-updates 10`을
사용한다. 10번 update 후 저장하고 멈추며, 다음 실행은 그 지점에서 이어간다.
이어학습 후 `gpu_handoff restore`를 다시 실행할 필요는 없다. 복원 명령은
원래 전달 상태와 다른 파일을 덮어쓰지 않도록 의도적으로 거부한다.

## 전달 데이터와 재개 검증

`handoff/zhang_1000_gpu/`의 분할 ZIP에 **530만 쌍 전체의 표준화된 입력·타깃
배열**, 원본 10 epoch와 현재 모델의 Adam 체크포인트, validation 데이터,
전처리 통계와 설정을 포함했다. 각 조각은 40 MiB 이하이며 Python 표준
라이브러리로 복원한다. 조각·압축파일·개별 파일의 SHA256을 모두 확인한다.
데이터 재생성과 전처리 재학습은 필요 없다. 복원된 배열과 상태를 위해
압축 조각 외에 약 0.6 GB 이상의 여유 공간이 필요하고, 복원 중 임시 압축
파일을 위한 공간도 필요하다. 가상환경과 CUDA 라이브러리 공간은 별도다.

원본 float64 화학 데이터 배열 및 2,450개 shard 전체는 포함하지 않는다.
그 데이터의 manifest와 hash는 저장소에 있으며, 훈련에 사용하는 float32
배열과 정규화 통계는 원본에서 이미 계산된 정확한 파일을 전달한다.
평가용 train metric도 원본 전체에서 계산된 값을 manifest hash와 함께
저장했으므로 GPU 환경에서 원본 float64 배열을 재생성할 필요가 없다.

CPU에서 원본 모델 가중치가 정확히 일치하고 Adam iteration이 복원되는지
확인했다. 중단 3회 update + 재개 3회 update와 연속 6회 update의 모델,
Adam, scheduler 텐서 31개도 정확히 일치했다(검증용 batch1024).
최종 전달은 사용자 선택대로 batch512다. 분할 패키지를 별도 폴더에 복원한
뒤 원본 float64 배열 없이 update 110,520 → 110,521로 이어가는 것도
확인했다. 전체 테스트는 98개가 통과했다. GPU 하드웨어에서의 실행 자체는
현재 GPU가 없는 CPU 환경에서 검증할 수 없었다. GPU 연산의 부동소수점
결과가 CPU와 비트 단위로 같다는 보장은 하지 않는다.

## 학습률과 저장

- 기존 Adam 상태를 유지하고, LR은 현재 값 **1e-4**에서 이어간다.
- 매 epoch validation **90,000쌍의 표준화 MAE**를 계산한다.
- 유의한 개선(1e-5)이 10 epoch 없으면 LR ×0.5, cooldown 2 epoch,
  최저 LR **1e-7**을 적용한다. 조기 종료 없이 총 1000 epoch가 목표다.
- 500 update마다 및 epoch 종료마다 모델·Adam·scheduler·진행 위치를 함께 저장한다.
- `seed11/best/`에는 validation MAE가 가장 낮은 가중치를 별도로 저장한다.
  test 결과로 모델이나 LR을 고르지 않는다.
- epoch 20/50/100/200/500/1000의 가중치도 보존한다.

설정은 [configs/zhang_5p3m_gelu_1000.json](configs/zhang_5p3m_gelu_1000.json),
진행 상태는 `results/zhang_5p3m_gelu_1000/g0.1/seed11/progress.json`,
epoch별 검증 기록은 같은 폴더의 `validation_history.jsonl`에 기록한다.
재개 시 config/source binding을 검사하므로 이어학습 중 원본 설정이나
`large_zhang_continue.py`를 수정하지 않는다. 별도 조건 실험은 새 출력 경로를 쓴다.

## 엔탈피 온도 복원과 평가

학습 입출력과 loss는 기존과 같다. 해석 시 **NN의 온도 출력을 버리고**,
이전 상태의 엔탈피와 갱신된 조성으로 다음 온도를 복원한다. 물성 계산에는
Cantera의 정규화 조성 정의를 사용하지만 저장·다음 입력의 ANN 종 조성은
변경하지 않는다. 원소 projection, 조성 clipping, 출력 질량합 재정규화는
추가하지 않았다. 따라서 엔탈피 복원이 종·원소 보존까지 보장하지 않는다.

동일한 10 epoch 모델의 비교는
[엔탈피 온도 복원 보고서](ZHANG_5P3M_ENTHALPY_EPOCH10_REPORT.md)에 있다.
20 ms 수치 완주는 15/18, 물리 허용 완주는 1/18이다. 완주한 15개만의
온도 RMSE 평균은 429.2 K이며, raw 온도 모델의 18개 평균과 직접 비교해
향상이라고 해석하면 안 된다.

1000 epoch 학습 후 최종 가중치와 validation 최상 가중치를 각각 평가한다.

```bash
python -m kinetics.large_zhang_evaluate --root results/zhang_5p3m_gelu_1000/g0.1 --expected-epochs 1000 --temperature-mode enthalpy
python -m kinetics.large_zhang_report --root results/zhang_5p3m_gelu_1000/g0.1/evaluation_enthalpy_seeds11 --report ZHANG_EPOCH1000_FINAL_REPORT.md
python -m kinetics.large_zhang_evaluate --root results/zhang_5p3m_gelu_1000/g0.1 --expected-epochs 1000 --temperature-mode enthalpy --model-version best
python -m kinetics.large_zhang_report --root results/zhang_5p3m_gelu_1000/g0.1/evaluation_enthalpy_best_seeds11 --report ZHANG_EPOCH1000_BEST_REPORT.md
```

진행 중 snapshot 모델의 평가를 위해 완료 기록을 임의로 1000 epoch로
바꾸지 않는다. 현재 공개된 결과는 10 epoch 비교이며, 1000 epoch 결과는
아직 없다.
