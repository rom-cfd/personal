"""Report physical-increment learning without hard physics corrections."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/kinetics-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/kinetics-cache')
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .wang_report import table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/delta100_raw'))
    parser.add_argument('--report',type=Path,default=Path('DELTA_RAW_REPORT.md'))
    args=parser.parse_args();root=args.root;out=root/'evaluation'
    manifest=json.loads((out/'manifest.json').read_text());assert manifest['status']=='completed'
    cfg=json.loads((root/'config.json').read_text())
    frozen=json.loads((out/'selection_frozen.json').read_text());seed=frozen['selected_seed']
    with np.load(cfg['source']) as data:
        states,split,dt,names=data['states'],data['split'],float(data['dt']),data['species']
        matrix=data['element_matrix']
    ids=np.flatnonzero(split=='test');times=np.arange(states.shape[1])*dt*1e3
    figures=out/'figures_log_time';figures.mkdir(exist_ok=True)
    colors=dict(single_best='tab:red',mean100='tab:blue',medoid100='tab:orange')
    labels=dict(single_best=f'Single seed {seed}',mean100='Mean 100',medoid100='Medoid 100')
    traces={}
    for name in colors:
        with np.load(out/'traces'/f'{name}_test.npz') as archive:
            traces[name]={int(i):archive[f'trajectory_{int(i)}'] for i in ids}
    def draw(ax,i,col):
        ax.plot(times[1:],states[i,1:,col],'k--',lw=1.8,label='Cantera',zorder=5)
        for name,color in colors.items():
            trace=traces[name][i]
            ax.plot(times[1:len(trace)],trace[1:,col],color=color,lw=1,label=labels[name])
            if 1<len(trace)<len(times):
                ax.plot(times[len(trace)-1],trace[-1,col],marker='x',ms=7,color=color)
        ax.set(xscale='log',xlim=(times[1],times[-1]),xlabel='Time (ms, log scale)',ylabel='T (K)' if col==0 else 'Y (signed)')
        ax.grid(True,alpha=.2)
        if col:
            ax.set_yscale('symlog',linthresh=1e-12)
        else:
            # Autoscale includes all finite negative/large raw temperatures.
            ax.axhspan(200,6000,color='gray',alpha=.06)
    def save(fig,name):
        fig.tight_layout(rect=(0,.025,1,.96))
        fig.text(.5,.008,'No physics correction or clipping. t=0 omitted on log axis; x marks numerical termination.',ha='center')
        fig.savefig(figures/name,dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,6,figsize=(22,11),sharex=True)
    for i,ax in zip(ids,axes.flat):
        draw(ax,int(i),0);ax.set_title(f'IC {i}')
    axes.flat[-1].legend(fontsize=7)
    fig.suptitle('Signed physical increments: raw 10 ms autoregressive temperature profiles')
    save(fig,'all_temperature_profiles.png')
    fig,axes=plt.subplots(3,6,figsize=(22,11),sharex=True)
    for i,ax in zip(ids,axes.flat):
        draw(ax,int(i),0);ax.set_title(f'IC {i}');ax.set_ylim(150,3500)
    axes.flat[-1].legend(fontsize=7)
    fig.suptitle('Temperature VIEW WINDOW 150--3500 K; see full-range figure for excursions')
    save(fig,'temperature_near_reference.png')
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True)
    for i,ax in zip([4,44,64,78],axes.flat):
        draw(ax,i,0);ax.set_title(f'IC {i}')
    axes.flat[-1].legend(fontsize=8);fig.suptitle('Raw signed-delta prediction: no state repair, including nonphysical states')
    save(fig,'difficult_initial_conditions.png')
    fig,axes=plt.subplots(3,3,figsize=(14,11),sharex=True);i=int(ids[0])
    for j,ax in enumerate(axes.flat):
        draw(ax,i,j+1);ax.set_title(str(names[j]))
    axes.flat[-1].legend(fontsize=7)
    fig.suptitle(f'IC {i}: signed species profiles (symlog Y, linear window +/-1e-12)')
    save(fig,'species_profiles.png')
    fig,axes=plt.subplots(1,3,figsize=(16,5));i=int(ids[0])
    for name,color in colors.items():
        trace=traces[name][i];y=trace[:,1:];t=times[:len(trace)]
        values=[y.sum(axis=1)-1,np.max(np.abs((y-states[i,0,1:])@matrix.T),axis=1),y.min(axis=1)]
        for ax,v in zip(axes,values):ax.plot(t[1:],v[1:],color=color,label=labels[name])
    for ax,title,threshold in zip(axes,['sum(Y)-1','max element mass-fraction error','minimum Y'],[1e-8,1e-8,1e-12]):
        ax.set(xscale='log',yscale='symlog',xlim=(times[1],times[-1]),xlabel='Time (ms)',title=title)
        ax.set_yscale('symlog',linthresh=threshold);ax.grid(True,alpha=.2);ax.legend(fontsize=7)
    fig.suptitle(f'IC {i}: constraint violations are recorded without correction')
    save(fig,'conservation.png')
    summary=pd.read_csv(out/'summary.csv');detail=pd.read_csv(out/'trajectories.csv');probe=pd.read_csv(out/'one_step_probes.csv')
    def result(partition):
        d=summary[summary.split==partition].copy()
        for label,column in [('수치 완주 %','completion_fraction'),('물리 유효 완주 %','physical_fraction'),('점화 구분 %','ignition_classification'),('점화오차 중앙값 %','median_ignition_error_refined')]:
            d[label]=(100*d[column]).round(2)
        d['T profile 평균 K']=d.temperature_profile_mean.round(2)
        d['T pooled K']=d.temperature_profile_pooled.round(2)
        return table(d[['method','수치 완주 %','물리 유효 완주 %','점화 구분 %','점화오차 중앙값 %','ignition_pairs','T profile 평균 K','T pooled K','species_profile_mean']])
    d=detail[detail.split=='test'].groupby('method').agg(
        negative_state_fraction=('negative_state_fraction','mean'),
        negative_beyond_tolerance_fraction=('negative_beyond_tolerance_fraction','mean'),
        minimum_Y=('minimum_mass_fraction','min'),minimum_T=('minimum_temperature','min'),maximum_T=('maximum_temperature','max'),
        maximum_mass_sum_error=('mass_sum_max','max'),maximum_element_error=('elemental_max','max'),
        minimum_coverage=('coverage','min'),physically_valid_ignitions=('ignition_physically_admissible','sum')).reset_index()
    ranking=pd.read_csv(out/'validation_member_ranking.csv')
    target_diagnostics=pd.read_csv(out/'target_diagnostics.csv')
    timing=pd.read_csv(out/'timing.csv')
    test_path=root/'tests.log';tests=test_path.read_text().splitlines()[-1] if test_path.exists() else 'See pytest log'
    findings=[]
    for row in summary[summary.split=='test'].itertuples():
        findings.append(f'- {row.method}: 수치 완주 {100*row.completion_fraction:.1f}%, 물리 유효 완주 {100*row.physical_fraction:.1f}%, '
                        f'raw 점화 구분 {100*row.ignition_classification:.1f}%, T profile 평균 RMSE {row.temperature_profile_mean:.1f} K.')
    text=f'''# Signed physical increments, γ=.1: physics 보정 없는 비교

## 주요 결과

{chr(10).join(findings)}

위 raw 점화 수치와 수치 완주만으로 성공이라 판정하지 않는다. 물리 위반과 전체 profile 오차를 함께 확인해야 한다.

## 구현

사용자가 확인한 변환은 **f(v)=sign(v)|v|^.1/.1**, 역변환은 **f⁻¹(z)=sign(z)(.1|z|)^10**이다.
입력은 (T,P,Y), 출력은 **ΔT=T(t+dt)−T(t), ΔY=Y(t+dt)−Y(t)**다.
Species에는 물리 공간에서 먼저 차를 구하고 signed Box–Cox를 적용했다.
다음 상태는 (T,Y)현재 + 예측한 물리 증분이다. ΔT/1000은 선형 표준화했다.
입력 T/1000, log(P), signed Box–Cox(Y)와 출력은 각각 train만으로 표준화했다.
음수 입력 Y의 signed extension은 보정 없이 self-fed 평가하기 위한 수학적 정의이며 음수 종을 물리적으로 정당화하지 않는다.

이전 모델도 이미 **f(Y다음)−f(Y현재)**라는 transformed-state residual을 학습했다.
이번 변화는 residual 연결의 최초 도입이 아니라 **f(Y다음−Y현재)**를 target으로 사용한다는 점이다.
이 두 식은 γ≠1에서 같지 않으며 target의 오차 weighting도 달라진다.

**원소/질량 보존 투영, 엔탈피 온도 복원, clipping, sum normalization, 음수 후보 제외, ODE fallback을 모두 사용하지 않았다.**
온도도 DNN이 직접 예측한 ΔT로 진화한다. 각 finite 음수 후보는 평균·medoid 계산에 그대로 포함한다.
Mean은 inverse transform 후 물리 증분/다음 상태의 평균이고 transformed 출력의 평균이 아니다.
Medoid는 training-scaled T/Y 거리 합이 최소인 실제 raw 후보 하나를 매 step 선택한다.
Single은 validation one-step score로 선택한 **seed {seed}** 하나를 전 timestep 고정한다.

## 데이터와 학습 예산

기존 10-ms 원본 궤적과 동일 split, downsampled train 8,815쌍을 재사용했다. 모든 쌍·ANN step은 dt=2 μs다.
새 target/정규화이므로 기존 가중치를 재사용하지 않고 **100개 모두 새로 학습**했다.
64–64–64 tanh, 9,738 parameter, batch512, Adam, seed11,22,…,1100, 모델당10,000 update다.
기존과 같은 learning-rate schedule/global gradient clipping을 사용했다. 이는 optimizer 설정이며 출력 state의 clipping이 아니다.
Grouped 학습에서도 초기화·미니배치 RNG·Adam moment·gradient norm clipping은 멤버별 독립이다.
훈련 {manifest['training_seconds']:.1f}초, 선택/평가 {manifest['evaluation_seconds']:.1f}초였다. CPU만 사용했다.

100개 모두의 validation 90,000개 one-step 쌍으로 기존 물리 단위 `selection_score`를 계산했다.
relative error floor=1e-8, major threshold=1e-3이고 transformed MSE는 선택에 사용하지 않았다.
물리 score = mean(log1p(capped robust relative Y error)) + 10 major_Y_RMSE + 10 mass_sum_MAE + 100 invalid_row_fraction.
가중치나 score를 test/rollout 결과로 조정하지 않고 선택 seed와 metric을 rollout 전에 고정했다.
Full validation에는 평형 구간도 들어가므로 a priori 순위가 induction/rollout 최적 순위를 뜻하지 않는다.

{table(ranking[['seed','selection_score','rmse','major_rmse','minor_rmse','invalid_row_fraction','temperature_rmse','transformed_mse']].head(5))}

## 10 ms a posteriori

같은 현재 상태를 모든 멤버에 넣고 평균/대표를 다음 입력으로 사용한다. 중간 정답은 입력하지 않았다.
Validation:

{result('val')}

Test:

{result('test')}

**수치 완주와 물리적으로 유효한 완주를 구분해야 한다.** 물리 위반이 생겨도 finite 상태는 그대로 진화했다.
진단상 Y<−1e-12, Y>1+1e-12, T outside[200,6000] K, 질량 합/원소 오차>1%면 물리 허용 범위를 벗어난 것으로 표시했다.
모든 정확한 음수 occurrence도 별도로 기록했다. 이 threshold들은 correction/filter가 아니다.
NaN/Inf 또는 |T|>1e6 K, |Y|>1e3일 때만 수치 실패로 중단했다. 마지막 finite prefix를 full-profile 정확도로 보고하지 않는다.
수치 guard를 넘은 후보는 event CSV에 남기고 trace에 성공 step으로 추가하지 않았다.
Full-profile RMSE는 수치 완주한 궤적에서만 계산하며 물리 유효성과 함께 읽어야 한다.
점화는 T0+400 K crossing, reference만0.2 μs 정밀 측정이다.
점화 구분/조건부 지연오차는 **raw temperature crossing**이며 이미 비물리적인 궤적의 crossing도 포함할 수 있다.
물리 허용 범위 안에서 감지한 점화 수를 아래 별도 열로 제공했다. 작은 raw IDT 오차만으로 성공이라 주장하지 않는다.
Crossing이 처음 물리 위반 step에서 발생하면, 보간된 crossing 시간이 그 step보다 빠르더라도 물리적으로 유효한 점화로 인증하지 않는 보수적 기준을 적용했다.
이 분류는 저장된 raw trace에서 독립적으로 재계산·검증했으며 상태·실제 IDT·raw 분류·profile error를 바꾸지 않는다. 수정 전 진단과 변경 기록도 보존했다.

## 물리 위반 진단

{table(d)}

Species profile의 y축은 **symlog**로 그려 음수 예측을 숨기지 않았다. 온도 축도 음수/큰 finite 값을 포함한다.
별도의 temperature_near_reference 그림만 표시 창을 150–3500 K로 제한하며 full-range 그림과 함께 제공한다. 예측값이나 metric을 clipping한 것이 아니다.
검은 점선은 Cantera이며, x 표시는 수치 중단 endpoint다. 시간축만 log(0.002–10 ms)이고 모든 평가 point는 유지했다.

## One-step test와 비용

아래 one-step 표만 이전 비교와 동일한 IC당31개 probe를 사용했다. 선택에는 90,000 validation쌍 전체를 썼다.

{table(probe[probe.split=='test'][['method','rmse','major_rmse','minor_rmse','relative_log_mean','negative_row_fraction','invalid_row_fraction','mass_sum_max','elemental_max','temperature_rmse']])}

{table(timing.round(4))}

미완주는 실행 step 수가 적으므로 단순 wall time 감소를 같은 10-ms 완주 속도라고 해석하지 않는다.
이 CPU prototype의 비용이며 Cantera 대비 1000배 가속을 측정한 결과가 아니다.

## 해석과 한계

Signed transform은 생성·소모 증분을 한 함수와 역함수로 나타낼 수 있고, physical delta target은 큰 현재 농도를 직접 재예측하는 부담을 줄인다.
그러나 이전 모델도 transformed residual을 학습했으므로 identity 학습을 제거했다는 이유만으로 개선을 단정할 수 없다.
f'(v)=|v|^−.9라서 **0 근처의 작은 증분과 roundoff 부호 변화는 여전히 크게 확대된다**.
특히 거의 변하지 않는 종의 매우 작은 signed label도 transformed loss에서 중요해질 수 있다.
Inverse는 작은 transformed 값의 물리 증분을 강하게 줄이지만, 표준화·종별 bias·반복 누적까지 제거하지 않는다.
물리 증분의 평균은 독립 분산을 줄일 수 있으나 sum(ΔY)=0, AΔY=0, energy consistency를 강제하지 않는다.
Medoid도 이미 biased한 후보 사이의 중앙값을 고를 뿐, 새로운 물리 조건을 추가하지 않는다.
One-step 정확도, transformed MSE, 장기 profile/물리 보존성은 구분해서 봐야 한다.

선택 모델의 validation target/출력별 진단은 다음과 같다. Temperature만 선형 ΔT/1000이고 나머지는 signed Box–Cox다.
MSE share는 새로운 출력 정규화 좌표에서의 비중이며 이전 모델의 다른 target MSE와 크기를 직접 비교하지 않는다.

{table(target_diagnostics)}

실제 이 데이터의 **N2 훈련 증분은 모두 정확히 0**이며 N2 loss 기여도도 0이다. 따라서 이번 손실을 N2 roundoff 때문이라고 설명할 수 없다.
Near-zero sensitivity는 변환 자체의 성질이지만 장기 실패의 유일한 원인으로 입증한 것은 아니다.
Per-output MSE와 물리 drift의 관계는 관찰 결과이고, 별도의 원인 분리 ablation 없이 단정하지 않는다.

이번 세 arm은 같은 새 target/physics-off 조건의 공정한 비교다.
이전 Wang 결과와 비교하면 target과 physics 사용 여부가 함께 바뀌므로, 차이를 target 하나의 효과로 분리해 주장할 수 없다.
같은 공개 test IC의 추가 실험이며 독립적인 외부 검증이 아니다.

## 파일과 재현

테스트: **{tests}**. Signed0/1e-300 역변환, 물리 delta target, serialization,
grouped training 독립성, raw 음수 전달/위반 기록, 수치 완주와 물리 유효 구분을 검사했다.
훈련 모델마다 reload prediction을 확인했고 평가 첫 self-fed step과 실제 medoid 후보 identity를 검증했다.

- [전체 온도](results/delta100_raw/evaluation/figures_log_time/all_temperature_profiles.png), [온도 표시 창 확대](results/delta100_raw/evaluation/figures_log_time/temperature_near_reference.png), [어려운 조건](results/delta100_raw/evaluation/figures_log_time/difficult_initial_conditions.png), [signed 종 profile](results/delta100_raw/evaluation/figures_log_time/species_profiles.png), [보존 위반](results/delta100_raw/evaluation/figures_log_time/conservation.png)
- [설정](configs/delta100_raw.json), [선택 순위](results/delta100_raw/evaluation/validation_member_ranking.csv), [고정 선택](results/delta100_raw/evaluation/selection_frozen.json)
- [요약 CSV](results/delta100_raw/evaluation/summary.csv), [IC별](results/delta100_raw/evaluation/trajectories.csv), [one-step](results/delta100_raw/evaluation/one_step_probes.csv), [모든 모델 hash](results/delta100_raw/members.json)

```bash
python -m kinetics.delta_training --config configs/delta100_raw.json
python -m kinetics.delta_evaluation --root results/delta100_raw
python -m kinetics.delta_target_diagnostics --root results/delta100_raw
python -m kinetics.verify_delta --root results/delta100_raw
python -m kinetics.delta_report --root results/delta100_raw
python -m pytest -q
```

재실행은 config output을 새 경로로 바꾸거나 훈련 `--resume`를 사용한다. 기존 데이터/과거 결과를 덮어쓰지 않았다.
'''
    args.report.write_text(text.replace('results/delta100_raw',str(root)))
    print(f'{args.report} written')


if __name__=='__main__':main()
