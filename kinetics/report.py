"""Generate the research report from completed, immutable study outputs."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/kinetics-cache')
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'boxcox_g0.1':'Box–Cox γ=0.1','boxcox_g0.3':'Box–Cox γ=0.3','boxcox_g0.5':'Box–Cox γ=0.5',
        'softlog_global':'Soft log ε=1e-8','r2_asinh_global_3.16228':'Asinh ε=3.162e-8',
        'r2_shifted_boxcox_global_3.16228':'Shifted BC γ=0.3, ε=3.162e-8'}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',default='results/verified');parser.add_argument('--report',default='RESEARCH_REPORT.md');args=parser.parse_args()
    out=Path(args.study);post=out/'posteriori'
    comp=pd.read_csv(out/'test_comparison.csv');means=comp.groupby('name').mean(numeric_only=True)
    std=comp.groupby('name').std(numeric_only=True)
    names=list(comp.name.unique())
    labels=lambda n:LABELS.get(n,n)
    tab=table(['변환','전체 Y RMSE ± seed SD','주요종 RMSE','미량종 RMSE','무효 행 %','음수 포함 행 %'],
        [[labels(n),f'{means.loc[n,"rmse"]:.3e} ± {std.loc[n,"rmse"]:.1e}',f'{means.loc[n,"major_rmse"]:.3e}',f'{means.loc[n,"minor_rmse"]:.3e}',f'{100*means.loc[n,"invalid_row_fraction"]:.2f}',f'{100*means.loc[n,"negative_row_fraction"]:.2f}'] for n in names])
    physical=table(['변환','합(Y) 오차 MAE','원소분율 오차 MAE','T RMSE (K)','변환공간 MSE'],
        [[labels(n),f'{means.loc[n,"mass_sum_mae"]:.3e}',f'{means.loc[n,"elemental_mae"]:.3e}',f'{means.loc[n,"temperature_rmse"]:.2f}',f'{means.loc[n,"transformed_mse"]:.3e}'] for n in names])
    noise=pd.read_csv(out/'noise_robustness.csv').groupby(['name','noise_std']).mean(numeric_only=True)
    ntable=table(['변환','입력 잡음 SD=1e-16','SD=1e-12','SD=1e-8'],
        [[labels(n)]+[f'{noise.loc[(n,a),"prediction_change_rmse"]:.3e}' for a in (1e-16,1e-12,1e-8)] for n in names])
    timing=pd.read_csv(out/'inference_benchmark.csv').groupby('name').mean(numeric_only=True)
    ttable=table(['변환','80 epoch 학습(s)','batch 추론(μs/sample)','단일 상태 추론(μs)'],
        [[labels(n),f'{means.loc[n,"train_seconds"]:.2f}',f'{timing.loc[n,"batch_microseconds_per_sample"]:.2f}',f'{timing.loc[n,"single_state_microseconds"]:.0f}'] for n in names])
    raw=pd.read_csv(post/'trajectory_metrics.csv')
    refined=pd.read_csv(post/'reference_ignition_refined.csv')
    joined=raw.merge(refined[['trajectory','tau400','tau_dTdt']],on='trajectory',how='left',suffixes=('','_fine'))
    for definition in ('tau400','tau_dTdt'):
        # Fine reference remains right-censored at each evaluation horizon.
        ref=joined[definition].where(joined[definition]<=joined.horizon)
        pred=joined['predicted_'+definition]
        joined['refined_'+definition]=ref
        joined[definition+'_relative_error_refined']=np.abs(pred-ref)/ref
        joined['physical_'+definition+'_relative_error_refined']=joined[definition+'_relative_error_refined'].where(
            np.isfinite(pred)&(joined.first_physical_violation_seconds.isna()|(pred<joined.first_physical_violation_seconds)))
    joined.to_csv(post/'ignition_comparison.csv',index=False)
    def posteriori_table(horizon):
        entries=[]
        for n in names:
            for repair in (False,True):
                g=joined[(joined.name==n)&(joined.repair==repair)&(joined.horizon==horizon)]
                err=g.tau400_relative_error_refined;peak=g.tau_dTdt_relative_error_refined
                entries.append([labels(n),'clip/sum' if repair else 'raw',f'{100*g.complete.mean():.1f}',f'{100*g.physically_admissible_complete.mean():.1f}',
                  f'{g.full_temperature_rmse.mean():.1f}' if g.complete.any() else '未完走',
                  f'{g.full_species_rmse.mean():.3e}' if g.complete.any() else '未完走',
                  f'{100*err.mean():.1f} ({err.count()})' if err.count() else 'N/A',
                  f'{100*peak.mean():.1f} ({peak.count()})' if peak.count() else 'N/A',
                  str(g.physical_tau400_relative_error_refined.count())])
        return table(['変換','処理','数値完走 %','物理許容完走 %','全区間 T RMSE K*','全区間 Y RMSE*','τ400誤差 % (対数)','τdT/dt誤差 % (対数)','物理許容τ400対数'],entries)
    # Headers are rendered in Korean; keep formulas and counts compact.
    short=posteriori_table(.0004).replace('変換','변환').replace('処理','처리').replace('数値完走','수치 완주').replace('物理許容完走','물리 허용 완주').replace('全区間','전체').replace('誤差','오차').replace('対数','쌍 수').replace('物理許容','물리 허용').replace('未完走','미완주')
    long=posteriori_table(.002).replace('変換','변환').replace('処理','처리').replace('数値完走','수치 완주').replace('物理許容完走','물리 허용 완주').replace('全区間','전체').replace('誤差','오차').replace('対数','쌍 수').replace('物理許容','물리 허용').replace('未完走','미완주')
    manifest=json.loads((out/'manifest.json').read_text())
    pmanifest=json.loads((post/'manifest.json').read_text())
    rmanifest=json.loads((post/'reference_resolution.json').read_text())
    baseline=means.loc['boxcox_g0.1'];shift=means.loc['r2_shifted_boxcox_global_3.16228']
    improvement=100*(1-shift.rmse/baseline.rmse)
    sensitivity=noise.loc[('boxcox_g0.1',1e-16),'prediction_change_rmse']/noise.loc[('r2_shifted_boxcox_global_3.16228',1e-16),'prediction_change_rmse']
    finite_reference=refined.tau400.notna().sum()
    # Additional figures make noise and all-species profiles directly inspectable.
    plt.figure(figsize=(9,5))
    for n in names:
        g=noise.loc[n];plt.plot(g.index,g.prediction_change_rmse,'o-',label=labels(n))
    plt.xscale('log');plt.yscale('log');plt.xlabel('Input mass-fraction noise SD');plt.ylabel('Prediction-change RMSE (finite pairs)');plt.legend(fontsize=7)
    plt.tight_layout();plt.savefig(out/'figures/noise_robustness.png',dpi=180);plt.close()
    reference=np.load(post/'cantera_reference.npz');ids=reference['ids'];truth=reference['states'];dt=float(reference['dt'])
    # Predetermined first held-out IC, seed 11. All nine species shown; failures end curves.
    for repair in (False,True):
        figure,axes=plt.subplots(3,3,figsize=(13,10));times=np.arange(truth.shape[1])*dt*1e3
        for j,ax in enumerate(axes.flat):
            ax.plot(times,np.maximum(truth[0,:,j+1],1e-20),'k--',lw=2,label='Cantera')
            for n in names:
                trace=np.load(post/f'{n}_seed11_repair{int(repair)}.npz')[f'trajectory_{int(ids[0])}']
                ax.plot(times[:len(trace)],np.maximum(trace[:,j+1],1e-20),lw=.8,label=labels(n))
            ax.set(yscale='log',ylim=(1e-20,2),xlabel='Time (ms)',ylabel='Mass fraction',title=str(np.load(out/'dataset.npz')['species'][j]))
        axes.flat[-1].legend(fontsize=6);figure.suptitle(f'Self-fed 0-D species profiles, IC {int(ids[0])}, seed 11; repair={repair}; positive log display only')
        figure.tight_layout();figure.savefig(post/f'all_species_profiles_repair{int(repair)}.png',dpi=160);plt.close(figure)
    text=f'''# 9종 수소 화학반응 ANN 전처리 비교 연구

## 핵심 결과

검증 데이터로 파라미터를 선택한 뒤 6개 변환 × 3개 시드 = 18개 모델을 실제 학습했다.
**Shifted Box–Cox는 극미량 입력 잡음에 훨씬 둔감하고 one-step 전체 질량분율 RMSE를 γ=0.1 대비 {improvement:.1f}% 낮췄다. 그러나 미량종 오차·음수 출력·점화지연·전체 0-D profile을 함께 보면 안정한 대체 모델이라고 결론 내릴 수 없다.**
특히 **2 ms a posteriori 적분에서 모든 방법의 원출력 및 clip/sum 처리 모두 물리 허용 완주율이 0%**다.
전처리만 바꿔 장기 적분 문제를 해결했다는 주장은 이 실험으로 뒷받침되지 않는다.
점화지연 숫자만 존재해도 질량·원소 보존이 깨진 궤적이면 신뢰 가능한 점화 예측으로 취급하지 않았다.

## 조사 결과와 실행 환경

- 저장소는 초기 커밋도 없는 빈 저장소였다. 기존 ANN/Box–Cox 코드·정규화·데이터·split·CI·평가 스크립트가 없었다. γ=0.1은 새 참조 구현이며 기존 모델 재현이라고 주장하지 않는다. 원본 데이터/구현을 덮어쓰지 않았다.
- Python {manifest['python']}, TensorFlow {manifest['tensorflow']}, Cantera 3.2.0, NumPy {manifest['numpy']}. CPU 5개, 약 33 GiB RAM, GPU 없음. TensorFlow intra/inter-op thread를 2/1로 제한했다. HPC 작업 없음.
- 최종 학습/기존 rollout 실행 {manifest['wall_seconds']:.1f}초, 확장 a posteriori 실행 {pmanifest['wall_seconds']:.1f}초. 패키지 설치·코드 작성·보고서 시간은 별도다. 실제 계산은 승인받은 약 30분 CPU 예산 이내다.

## 반응계, 데이터, 예측 대상

Cantera `h2o2.yaml`에서 Ar를 제외한 H2, H, O, O2, OH, H2O, HO2, H2O2, N2와 이 종들로만 이루어진 28개 반응을 사용했다. 단열·정압 0-D 이상기체 반응기다.
90개 독립 초기조건을 T0∈[900,1600] K, P∈[1,5] atm, 당량비∈[0.5,2]에서 균일 추출했다. 산화제 O2:N2=1:3.76, 연료 H2다.
Cantera `rtol=1e-10`, `atol=1e-20`. 반응 중 온도는 약 902–3024 K를 포함한다.

**최종 요청대로 reaction rate 예측은 사용하지 않는다.** 입력은 `[T(t), Y₁(t)…Y₉(t), P]`, 출력은 `[T(t+Δt), Y₁(t+Δt)…Y₉(t+Δt)]`다.
모든 학습·검증·테스트·ANN rollout에서 **Δt=2×10⁻⁶ s = 2 μs**로 고정했다.
원래 데이터 시간구간은 0–0.4 ms, 궤적당 200개 전이, 전체 18,000개 전이다.
초기조건 단위로 54/18/18개 궤적을 학습/검증/테스트에 고정 분리했다(10,800/3,600/3,600개 전이). 같은 궤적의 이웃 시점이 다른 split에 들어가지 않는다.
데이터 시드 20261029, 학습 시드 11/22/33이다. split과 초기조건은 NPZ에 저장했다.
원본 NPZ SHA256: `{manifest['dataset_sha256']}`.
Cantera의 [-1e-14,0) 음수 roundoff만 데이터 생성 시 0으로 처리하도록 했으나 이번 생성의 최소값은 0이었다. 학습/평가 예측값은 별도 진단을 제외하고 자르지 않았다.

## ANN 및 공정한 비교

11 입력 → 64 tanh → 64 tanh → 64 tanh → 10 선형 출력, 학습 파라미터 9,738개. Adam 0.001, batch 512.
모든 후보에 동일한 구조·split·시드·셔플 순서·epoch 예산을 사용했다. 마지막 선형층은 0 초기화하여 초기 예측이 현재 상태 유지와 같게 했다.
온도는 선형 스케일, 압력은 로그 스케일, Y는 비교 대상 변환을 적용했다. 평균·표준편차는 **학습 궤적에서만** 적합했다.
네트워크는 표준화된 변환 좌표의 잔차를 학습하고 현재 좌표에 더해 다음 상태를 복원한다. 따라서 실제 출력은 다음 상태이며, 물리적 signed 반응률이나 dY를 positive-only 변환에 넣지 않는다.

초기 12개 후보를 각각 15 epoch로 screening했다: γ=0.1/0.3/0.5와 softlog/asinh/shifted BC 각각 global ε=1e-8 및 종별 ε 두 가지.
종별 ε는 학습 데이터의 종별 양수 Y 중앙값 × 1e-2 또는 1e-4, 하한 1e-16이다. 예를 들어 multiplier=1e-4에서 H2O ε≈1.82e-5, H2O2 ε≈4.56e-11이었다.
검증 성능이 좋은 서로 다른 두 epsilon 변환 계열을 대상으로 1회차 ×0.1/10, 2회차 ×10^(-0.5)/10^(0.5)를 탐색했다. 총 20개 screening 학습이다.
γ 세 개와 각 epsilon 계열의 검증 최선 설정을 최종 비교에 포함했다. 각 모델 80 epoch를 모두 학습한 뒤 검증 점수가 가장 좋은 checkpoint를 복원했다.
선택 점수는 `mean(log1p(min(|error|/max(Y,1e-8),1e12))) + 10·major_RMSE + 10·mass_sum_MAE + 100·invalid_row_fraction`다.
이는 변환공간 MSE가 아닌 물리량 기반 검증 지표다. 최종 설정은 `selection_frozen.json`에 test 평가 전에 저장했다. test 기반 파라미터 조정·재학습은 하지 않았다.
초기조건을 한 번만 분할했으므로 시드 SD는 데이터 분할 불확실성을 포함하지 않는다.

## 변환 정의와 수치적 의미

- Box–Cox: `f(Y)=Y^γ/γ`, 역변환 `Y=(γf)^(1/γ)`, `f≥0`. 통상 `(Y^γ−1)/γ`에 상수를 더한 형태로, 표준화 후 수학적으로 동일하다. 0/극미량에서 상수 차감으로 정보를 잃는 것을 피했다. 음수 f는 NaN으로 표시하고 무효 예측으로 센다.
- Soft log: `f(Y)=log1p(Y/ε)`, `Y=ε·expm1(f)`.
- Asinh: `f(Y)=asinh(Y/ε)`, `Y=ε·sinh(f)`.
- Shifted BC: `f(Y)=expm1(γ·log1p(Y/ε))/γ`, `Y=ε·expm1(log1p(γf)/γ)`; 역변환 영역 `1+γf>0`. γ=0.3. 이는 `((Y+ε)^γ−ε^γ)/γ`를 ε^γ로 나눈 것이며 표준화 후 동등하다.

공용 모듈에는 signed 확장 `sign(x)·f(|x|)`와 대응 역변환도 구현했다. 이번 mass-fraction 연구에는 사용하지 않았다.
forward/inverse와 정규화는 float64, ANN 연산은 float32다. 테스트는 0, 1e-300까지의 극미량, 음수 signed 입력, 단조성, 역변환 영역, 종별 ε, 보존성, split, 저장·재로딩을 검사한다.

## One-step 물리 단위 평가

주요종/미량종은 각 reference sample에서 Y≥1e-3 / Y<1e-3로 구분했다. 상대오차 floor는 모든 방법에 동일한 1e-8이다.
**RMSE는 유한한 개별 예측값에 대한 값이다. 따라서 γ=0.3/0.5의 RMSE를 무효 행 비율과 분리해 읽으면 안 된다.** 무효 예측은 selection score에 별도 페널티를 준다.
아래 평균·SD는 고정 test 궤적에 대한 학습 시드 3개의 결과다.

{tab}

{physical}

현재 상태 유지 baseline은 전체 Y RMSE 4.159e-3, T RMSE 39.79 K, 상대오차 log 평균 0.0210이다. 학습 모델은 절대 RMSE를 줄이지만, 미량종·보존성·일부 상대지표에서는 단순 유지보다 나쁠 수 있다.
Shifted BC의 변환공간 MSE는 γ=0.1보다 크지만 물리 단위 전체/주요종 RMSE는 작다. 반대로 screening에서 asinh ε를 1e-8→1e-9로 줄이면 변환 MSE는 0.003158→0.002880으로 낮아지지만 물리 검증 점수는 0.3113→0.3579로 나빠졌다.
**변환공간 loss의 개선과 물리 예측 정확도의 개선은 서로 다른 결과다.**

## 극미량 입력 잡음 시험

test 입력의 처음 8개 종에 고정 시드 가우시안 잡음을 더하고 음수 입력을 0으로 제한한 뒤 N2로 합을 보정했다. noisy 입력과 clean 입력의 예측 차이를 비교했다.
이는 성분별 반올림/섭동 스트레스 시험이며 원소별 보존을 강제한 물리적 불확실성 모델은 아니다. 아래는 유한 prediction pair의 Y 변화 RMSE다. 무효 비율은 CSV에 별도로 기록했다.

{ntable}

입력 잡음 SD=1e-16에서 shifted BC의 출력 변화는 γ=0.1보다 약 {sensitivity:.2e}배 작았다.
γ=0.1의 `df/dY=Y^-0.9`는 0 근방에서 발산한다. 극미량 radical 입력이 변환 좌표에서 크게 이동하고 DNN의 다른 종 출력까지 흔들 수 있다.
softlog/asinh/shifted BC의 원점 기울기는 1/ε로 유한하다. 그러나 bounded forward sensitivity가 보존성·역변환 오차·장기 안정성을 보장하지는 않는다.
γ=0.3/0.5는 원점 발산이 약해져 잡음 민감도는 줄었지만, 이 DNN에서 작은 종의 표현과 출력-domain 위반이 악화됐다.

## 실제 a posteriori 0-D 점화 및 profile 평가

단순 held-out one-step accuracy로 결론 내리지 않았다. **각 held-out 초기조건에서 시작해 ANN이 예측한 T,Y를 다음 입력으로 반복 사용했다. 중간 reference 상태를 입력하는 teacher forcing은 없다.**
P는 해당 반응기의 초기 압력으로 고정했다. reference는 같은 초기조건·메커니즘의 독립 Cantera 단열 정압 반응기다.
18개 초기조건 × 3개 시드 = 방법당 54개 궤적을 0.4 ms와 2 ms에서 평가했다. 2 ms는 학습 궤적 길이의 5배인 장기 확장 시험이다.
ANN 적분 간격은 계속 2 μs이며, Cantera 내부 adaptive 적분은 위의 엄격한 허용오차를 썼다.

점화지연은 (1) T0+400 K 최초 도달 시각(구간 선형 보간), (2) 400 K 상승이 확인되고 구간 내부에 있는 sampled 최대 dT/dt 시각을 모두 기록했다.
점화시각 reference 해상도 진단만 0.2 μs로 추가 계산했다. 이는 학습/test 데이터나 ANN Δt 변경이 아니다. 아래 점화 오차는 이 미세 reference를 사용한다.
2 ms 이내 reference 점화는 {finite_reference}/18개다. 나머지는 **2 ms 내 미점화/right-censored**이며 영구 미점화라고 단정하지 않는다.
2 μs ANN은 수 μs의 빠른 점화에 대해 시간 해상도 한계가 있으므로 작은 지연 차이를 과해석하면 안 된다.

`raw`는 원출력 반복이며, `clip/sum`은 매번 음수를 0으로 자르고 합이 1이 되게 정규화하는 별도 진단이다. 이 처리는 원소·에너지 보존을 보장하지 않는다.
NaN/Inf, T∉[200,6000] K, 양의 총질량 없음, raw에서 음수 종 발생 시 적분을 중단한다.
수치 완주는 이 중단 조건을 피했다는 의미일 뿐이다. Y가 매우 커지거나 원소/총질량이 틀어진 채 완주할 수 있다.
따라서 별도로 모든 시점에서 `|sum(Y)−1|≤0.01`, 최대 원소질량분율 drift≤0.01, Y≤1.01를 만족해야 **물리 허용 완주**로 센다. 이는 느슨한 진단 임계값이며 정확한 보존을 뜻하지 않는다.
점화 전에 이 기준을 위반하면 해당 점화값을 물리 허용 점화 쌍으로 세지 않는다.

profile 오차는 시간 적분 L2 RMSE, 즉 `sqrt(integral(error² dt)/horizon)`다. 종별 전체 profile RMSE와 reference L2에 대한 상대오차도 CSV에 제공한다.
**미완주 궤적의 전체 profile 오차는 NaN**이며, 별도로 coverage와 prefix 오차를 기록했다. 실패한 예측을 0 오차로 대체하지 않았다.
아래 `*` profile 평균은 **완주한 궤적에 한정**되므로 완주율과 같이 읽어야 하며, 방법간 단독 순위 기준으로 사용하면 생존자 편향이 생긴다.
점화 오차도 시각이 존재하는 쌍의 평균이라 실패/미점화를 포함하지 않는다. 마지막 열로 물리 허용 쌍 수를 명시했다. 각 행 전체 대상은 54개다.

### 0.4 ms: 학습 궤적과 같은 시간구간

{short}

### 2 ms: 장기 확장

{long}

one-step RMSE가 가장 낮은 shifted BC도 2 ms 점화지연·전체 profile과 물리 허용 완주를 통과하지 못했다.
raw softlog/asinh의 일부 수치 완주에서는 종 질량분율이 비현실적으로 커졌다. finite/positive라는 사실만으로 안정하다고 볼 수 없는 사례다.
일부 completed-only profile 평균이 작아 보여도 대부분의 궤적이 실패하거나 원소 보존을 어기므로 우수한 적분기라는 뜻이 아니다.
모든 초기조건의 seed 11 온도 profile을 그림으로 제공하며, 종별 그림은 사전에 고정한 첫 test 초기조건을 사용했다. 성능이 좋은 사례를 골라 보여주지 않았다.

## 각 변환의 성공·실패 원인

1. **γ=0.1 Box–Cox:** 이번 one-step 비교에서는 무효·음수 없이 미량종 표현과 상대오차의 균형이 가장 좋았다. 다만 극미량 잡음 증폭이 크고, 질량·원소 보존 제약이 없는 DNN의 작은 편향이 반복 입력에서 누적된다. 높은 one-step 정확도가 점화·profile 정확도로 이어지지 않았다.
2. **γ=0.3/0.5:** 작은 Y에 대한 forward 민감도를 완화하지만 trace 종을 압축된 작은 좌표에 놓는다. 유한 학습·float32 출력 오차가 domain 경계를 넘으면 다음 상태를 복원할 수 없다. γ=0.5는 최종 test 행의 약 16%가 무효이며 장기 수치 완주도 없다.
3. **Soft log:** ε 아래에서 선형이라 극미량 잡음은 억제한다. 그러나 큰 Y에서 지수형 역변환이 출력 좌표 오차를 증폭하며 주요종·질량합 오차가 커졌다. 작은 음수 좌표는 음수 Y로 복원된다. 이번 음수 크기는 대체로 1e-9 수준이지만 raw chemistry state로 사용하기에는 부적합하다.
4. **Asinh:** 원점에서 매끄럽고 signed 값에도 자연스럽다. trace/온도 one-step 오차에는 장점이 있지만 주요종 오차 및 반복 drift가 크다. signed 수학적 정의가 질량분율의 음수를 물리적으로 정당화하지 않는다.
5. **Shifted BC:** ε 아래는 유한 기울기, 그 위는 power-law tail이어서 로그 계열보다 주요종을 잘 복원하고 γ=0.1보다 잡음에 강했다. 그러나 smoothing으로 미량종 정보가 약해지고 음수 출력이 생긴다(이번 최소값 평균 약 −2.65e-8). clamp/normalize는 원소/에너지 보존을 고치지 않으므로 점화와 profile의 장기 정확도는 부족하다.
6. **종별 ε:** 종별 dynamic range를 반영하지만 양수 중앙값은 점화 전의 0/극미량 구간보다 점화 후 농도를 대표하기 쉽다. 예를 들어 H2O에 큰 ε를 주면 초기 생성 정보를 평탄화한다. 이번 두 multiplier는 검증 점수에서 global ε보다 나빴다. 종별 ε 자체가 항상 나쁘다는 일반 결론은 아니다.

## 효율과 검증

{ttable}

batch 추론은 3,600개 상태를 한 번에 처리한 warm-up 후 5회 중앙값이며, 단일 상태 추론은 Python/Keras 호출 overhead를 포함한다. batch throughput을 순차 chemistry 적분 속도로 해석하면 안 된다. Cantera 대비 speedup은 주장하지 않는다.
최종 모델 18개의 저장·재로딩 후 원래 test 예측이 엄격한 수치 허용오차 내에서 모두 재현됐다. 파일명 소수점으로 인한 초기 저장 충돌을 수정한 뒤 동일 설정을 재실행했으며, model persistence 회귀 테스트를 추가했다.
테스트 40개 통과. Keras/NumPy 연동의 deprecation warning 12개가 있지만 실패는 없다. 데이터 checksum, 재로딩 및 self-fed rollout 테스트를 포함한다.

## 한계와 다음 연구

이 결과는 작은 0-D 수소 benchmark, 한 데이터 분할, 작은 DNN 및 제한된 학습 예산에서 얻었다. 기존 사용자의 ANN을 개선했다는 증거가 아니고 CFD/다른 연료/압력 범위로 일반화하지 않는다.
선별과 추가 두 round는 물리 단위 **one-step 검증**으로 했으며, a posteriori 결과는 이후 독립 평가에 사용했다. 따라서 long-term 최적 파라미터를 찾았다고 주장하지 않는다.
다음 단계에서는 test를 고정한 채 **검증 초기조건의 점화·전체 profile·보존 실패를 선택 기준**에 포함하고, 원소 보존 projection/보존 구조 출력, enthalpy에 따른 온도 복원, rollout loss 및 trace noise augmentation을 별도 통제 실험으로 검토해야 한다.
이는 전처리 효과와 섞지 않도록 공통 제약 적용 여부를 모든 후보에 동일하게 비교해야 한다. 이번에는 세 번째 최적화 round나 test 결과를 이용한 재튜닝을 하지 않았다.
현재로서는 shifted BC를 **잡음 강건성을 더 연구할 후보**로 권장할 수 있지만, 장기 chemistry integrator로 채택할 수 있는 승자는 없다.

## 산출물과 재현

- [실행 방법](README.md), [버전 고정](requirements.lock.txt), [실험 설정](configs/study.json)
- [변환 모듈](kinetics/transforms.py), [데이터 생성](kinetics/data.py), [학습/평가](kinetics/study.py), [후처리](kinetics/analysis.py), [a posteriori 0-D](kinetics/posteriori.py), [reference 해상도](kinetics/reference_resolution.py), [보고서 생성](kinetics/report.py)
- [최종 one-step CSV](results/verified/test_comparison.csv), [seed 평균/SD](results/verified/summary.csv), [검증 탐색](results/verified/validation_screen.csv), [history](results/verified/history.csv)
- [종별 오차](results/verified/species_errors.csv), [농도 bin 분포](results/verified/concentration_errors.csv), [잡음](results/verified/noise_robustness.csv), [추론 시간](results/verified/inference_benchmark.csv)
- [0-D 전체 궤적 지표](results/verified/posteriori/trajectory_metrics.csv), [점화 비교](results/verified/posteriori/ignition_comparison.csv), [0-D 종별 profile](results/verified/posteriori/species_profile_errors.csv), [0-D 요약](results/verified/posteriori/summary.csv)
- [원소/질량 보존 survival](results/verified/figures/conservative_survival.png), [잡음 그래프](results/verified/figures/noise_robustness.png), [종별 one-step](results/verified/figures/species_errors.png), [loss vs physical](results/verified/figures/loss_vs_physical.png)
- [0-D raw 온도: 모든 초기조건](results/verified/posteriori/all_temperature_profiles_repair0.png), [0-D repaired 온도](results/verified/posteriori/all_temperature_profiles_repair1.png), [9종 전체 profile](results/verified/posteriori/all_species_profiles_repair1.png), [점화지연 산점도](results/verified/posteriori/ignition_delay.png)

**유효 최종 결과 경로는 `results/verified/`**다. `results/study/`는 중단한 pilot이고 `results/final/`은 모델 저장 검증 전의 대체된 실행이다. 재현·해석에는 verified 결과만 사용한다. 원본 및 중간 기록은 보존했다.
'''
    Path(args.report).write_text(text.replace('results/verified/',out.as_posix()+'/'))
    # Final source provenance includes evaluation/report code added after training.
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('kinetics').glob('*.py'))}
    (out/'source_hashes_final.json').write_text(json.dumps(hashes,indent=2))
    print('Research report and a posteriori ignition comparison written.')


if __name__=='__main__':main()
