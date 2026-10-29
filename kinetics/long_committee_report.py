"""Full 10 ms committee results, regime errors and original-data sampling audit."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/kinetics-cache')
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .long_data import equilibrium_mask
from .data import mechanism


def markdown_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    root=Path('results/committee100_long10ms');manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='completed':raise RuntimeError('Full evaluation must complete before writing final report')
    fig=root/'figures';fig.mkdir(exist_ok=True)
    cfg=json.loads((root/'config.json').read_text());data=np.load(cfg['source']);states=data['states'];pressures=data['pressures'];eq=data['equilibrium'];split=data['split'];dt=float(data['dt'])
    summary=pd.read_csv(root/'summary.csv');details=pd.read_csv(root/'trajectories.csv');timing=pd.read_csv(root/'timing.csv')
    selected=pd.read_csv('results/long10ms_data/sampling_summary.csv');methods=['mean10','medoid10','mean100','medoid100']
    confusion=[];regimes=[];selection=[];usage=[]
    for name in methods:
        for partition in ('val','test'):
            d=details[(details.method==name)&(details.split==partition)&np.isclose(details.horizon,.01)]
            confusion.append(dict(method=name,split=partition,TP=int((d.reference_ignites&d.prediction_ignites).sum()),
                FN=int((d.reference_ignites&~d.prediction_ignites&d.complete).sum()),TN=int((~d.reference_ignites&~d.prediction_ignites&d.complete).sum()),
                FP=int((~d.reference_ignites&d.prediction_ignites).sum()),unresolved=int((~d.complete&~d.prediction_ignites).sum())))
            traces=np.load(root/'traces'/f'{name}_{partition}.npz')
            for i in np.flatnonzero(split==partition):
                truth=states[i];trace=traces[f'trajectory_{int(i)}'];complete=len(trace)==len(truth)
                equ=equilibrium_mask(truth,eq[i]);equ=np.r_[equ,equ[-1]]
                crossed=np.flatnonzero(truth[:,0]>=truth[0,0]+400)
                pre=np.arange(len(truth))<(int(crossed[0]) if len(crossed) else len(truth))
                masks={'induction':pre,'reacting_relaxation':~pre&~equ,'equilibrium':equ}
                for regime,mask in masks.items():
                    row=dict(method=name,split=partition,trajectory=int(i),regime=regime,reference_points=int(mask.sum()),
                             covered_points=int(mask[:len(trace)].sum()),complete=complete)
                    if complete and mask.any():
                        delta=trace[mask,1:]-truth[mask,1:];major=truth[mask,1:]>=1e-3
                        row.update(T_rmse=float(np.sqrt(np.mean((trace[mask,0]-truth[mask,0])**2))),Y_rmse=float(np.sqrt(np.mean(delta*delta))),
                            major_Y_rmse=float(np.sqrt(np.mean(delta[major]**2))) if major.any() else np.nan,
                            minor_Y_rmse=float(np.sqrt(np.mean(delta[~major]**2))) if (~major).any() else np.nan)
                    else:row.update(T_rmse=np.nan,Y_rmse=np.nan,major_Y_rmse=np.nan,minor_Y_rmse=np.nan)
                    regimes.append(row)
            e=pd.read_csv(root/f'selection_{name}_{partition}.csv');a=e[e.accepted]
            if name.startswith('medoid'):
                selection.append(dict(method=name,split=partition,accepted_steps=len(a),switches=int(a.switched.sum()),
                    switch_rate=float(a.switched.sum()/max(len(a)-a.trajectory.nunique(),1)),minimum_valid=int(a.valid_count.min()),mean_valid=float(a.valid_count.mean())))
                usage.extend(dict(method=name,split=partition,seed=int(seed),selected_steps=int(count)) for seed,count in a.selected_seed.value_counts().items())
    confusion=pd.DataFrame(confusion);confusion.to_csv(root/'confusion.csv',index=False)
    pd.DataFrame(regimes).to_csv(root/'regime_errors.csv',index=False)
    pd.DataFrame(selection).to_csv(root/'selection_statistics.csv',index=False);pd.DataFrame(usage).to_csv(root/'representative_usage.csv',index=False)
    # Independent thermodynamic diagnostic, not a constraint used in selection.
    gas=mechanism();entropy_rows=[]
    for partition in ('val','test'):
        for i in np.flatnonzero(split==partition):
            true_entropy=[]
            for state in states[i]:
                gas.TPY=state[0],pressures[i],state[1:];true_entropy.append(gas.entropy_mass)
            for name in ['Cantera',*methods]:
                trace=states[i] if name=='Cantera' else np.load(root/'traces'/f'{name}_{partition}.npz')[f'trajectory_{int(i)}']
                values=true_entropy if name=='Cantera' else []
                if name!='Cantera':
                    for state in trace:
                        gas.TPY=state[0],pressures[i],state[1:];values.append(gas.entropy_mass)
                values=np.asarray(values);change=np.diff(values)
                entropy_rows.append(dict(method=name,split=partition,trajectory=int(i),decrease_tolerance=1e-6,
                    decreasing_fraction=float(np.mean(change < -1e-6)) if len(change) else np.nan,
                    largest_step_drop=float(max(0.,-change.min())) if len(change) else np.nan,
                    minimum_delta_from_initial=float((values-values[0]).min())))
    entropy=pd.DataFrame(entropy_rows);entropy.to_csv(root/'entropy_diagnostics.csv',index=False)
    # Fine peak timing and distribution tails; no fit/selection occurs here.
    enriched=[]
    for partition in ('val','test'):
        ref=pd.read_csv(root/f'reference_ignition_{partition}.csv').set_index('trajectory')
        for _,r in details[details.split==partition].iterrows():
            row=r.to_dict();tau=float(ref.loc[int(r.trajectory),'tau_dTdt']);tau=tau if tau<=r.horizon else np.nan
            row['reference_tau_peak_refined']=tau
            row['peak_relative_error_refined']=abs(r.predicted_tau_dTdt-tau)/tau if np.isfinite(tau) and np.isfinite(r.predicted_tau_dTdt) else np.nan
            enriched.append(row)
    enriched=pd.DataFrame(enriched);enriched.to_csv(root/'ignition_comparison.csv',index=False)
    def result_table(partition,horizon):
        rows=[]
        for name in methods:
            d=summary[(summary.method==name)&(summary.split==partition)&np.isclose(summary.horizon,horizon)].iloc[0]
            e=enriched[(enriched.method==name)&(enriched.split==partition)&np.isclose(enriched.horizon,horizon)]
            pairs=e.tau400_relative_error_refined.dropna()
            rows.append([name,f'{100*d.completion_fraction:.1f}',f'{100*d.physical_fraction:.1f}',f'{100*d.ignition_classification:.1f}',
                f'{100*pairs.median():.1f} ({len(pairs)})',f'{100*pairs.quantile(.95):.1f}' if len(pairs) else 'N/A',
                f'{d.temperature_profile_mean:.1f}',f'{d.temperature_profile_pooled:.1f}',f'{d.species_profile_mean:.3e}'])
        return markdown_table(['방법','완주 %','물리 허용 %','점화 구분 %','τ400 중앙오차 % (쌍 수)','τ400 P95 %','T profile 평균 K','T pooled K','Y profile 평균'],rows)
    ids=np.flatnonzero(split=='test');time_axis=np.arange(states.shape[1])*dt*1e3
    colors={'mean10':'tab:blue','medoid10':'tab:orange','mean100':'tab:green','medoid100':'tab:red'}
    f,axes=plt.subplots(3,6,figsize=(20,10),sharex=True)
    for i,ax in zip(ids,axes.flat):
        ax.plot(time_axis,states[i,:,0],'k--',lw=2,label='Cantera')
        for name in methods:
            trace=np.load(root/'traces'/f'{name}_test.npz')[f'trajectory_{int(i)}']
            ax.plot(time_axis[:len(trace)],trace[:,0],lw=.9,color=colors[name],label=name)
        ax.set(xlabel='Time (ms)',ylabel='T (K)',title=f'IC {i}: T0={states[i,0,0]:.0f} K',ylim=(150,3500))
    axes.flat[-1].legend(fontsize=6);f.suptitle('Fresh 10 ms training, equilibrium-thinned; same data/budget per member')
    f.tight_layout();f.savefig(fig/'all_temperature_profiles.png',dpi=170);plt.close(f)
    f,axes=plt.subplots(2,2,figsize=(12,8))
    for i,ax in zip([44,5,64,73],axes.flat):
        ax.plot(time_axis,states[i,:,0],'k--',lw=2,label='Cantera')
        for name in methods:
            trace=np.load(root/'traces'/f'{name}_test.npz')[f'trajectory_{int(i)}'];ax.plot(time_axis[:len(trace)],trace[:,0],color=colors[name],lw=1,label=name)
        ax.set(xlabel='Time (ms)',ylabel='T (K)',title=f'Previously difficult IC {i}')
    axes.flat[-1].legend(fontsize=7);f.tight_layout();f.savefig(fig/'difficult_initial_conditions.png',dpi=180);plt.close(f)
    f,axes=plt.subplots(3,3,figsize=(13,10));i=int(ids[0])
    for j,ax in enumerate(axes.flat):
        ax.plot(time_axis,np.maximum(states[i,:,j+1],1e-20),'k--',lw=2,label='Cantera')
        for name in methods:
            trace=np.load(root/'traces'/f'{name}_test.npz')[f'trajectory_{i}'];ax.plot(time_axis[:len(trace)],np.maximum(trace[:,j+1],1e-20),color=colors[name],lw=.9,label=name)
        ax.set(xlabel='Time (ms)',ylabel='Y',yscale='log',ylim=(1e-20,1.1),title=str(data['species'][j]))
    axes.flat[-1].legend(fontsize=6);f.suptitle(f'Nine species, first held-out IC {i}; log display floor only')
    f.tight_layout();f.savefig(fig/'species_profiles.png',dpi=170);plt.close(f)
    f,ax=plt.subplots(figsize=(8,6))
    for name in methods:
        d=enriched[(enriched.method==name)&(enriched.split=='test')&np.isclose(enriched.horizon,.01)]
        m=d.reference_tau400_refined.notna()&d.predicted_tau400.notna()
        ax.scatter(d.loc[m,'reference_tau400_refined']*1e3,d.loc[m,'predicted_tau400']*1e3,label=name,color=colors[name],s=20)
    ax.plot([.002,10],[.002,10],'k--');ax.set(xscale='log',yscale='log',xlabel='Cantera ignition delay (ms)',ylabel='ANN ignition delay (ms)');ax.legend(fontsize=8)
    f.tight_layout();f.savefig(fig/'ignition_delay.png',dpi=180);plt.close(f)
    r=pd.DataFrame(regimes);r=r[(r.split=='test')&(r.reference_points>0)]
    f,axes=plt.subplots(1,2,figsize=(12,4))
    for name in methods:
        a=r[r.method==name].groupby('regime')[['T_rmse','Y_rmse']].mean().reindex(['induction','reacting_relaxation','equilibrium'])
        axes[0].plot(a.index,a.T_rmse,'o-',label=name,color=colors[name]);axes[1].plot(a.index,a.Y_rmse,'o-',label=name,color=colors[name])
    axes[0].set_ylabel('Regime T RMSE (K)');axes[1].set_ylabel('Regime species RMSE')
    for ax in axes:ax.tick_params(axis='x',rotation=15);ax.legend(fontsize=7)
    f.tight_layout();f.savefig(fig/'regime_errors.png',dpi=180);plt.close(f)
    # Report template follows below.
    cost_table=markdown_table(['방법','분할','18개 10ms 실행(s)','DNN/역변환(s)','후보 보정(s)','선택(s)'],
        [[r.method,r.split,f'{r.seconds:.2f}',f'{r.neural_decode:.2f}',f'{r.candidate_repair+r.mean_repair:.2f}',f'{r.selection:.2f}'] for _,r in timing.iterrows()])
    entropy_table=markdown_table(['방법','엔트로피 감소 step 비율 평균 %','최대 단일 step 감소 J/(kg K)','최저 S−S0 J/(kg K)'],
        [[name,f'{100*g.decreasing_fraction.mean():.1f}',f'{g.largest_step_drop.max():.3e}',f'{g.minimum_delta_from_initial.min():.3e}']
         for name,g in entropy[entropy.split=='test'].groupby('method')])
    confusion_table=markdown_table(['방법','분할','TP','FN','TN','FP','실패 전 미검출'],
        [[r.method,r.split,r.TP,r.FN,r.TN,r.FP,r.unresolved] for _,r in confusion.iterrows()])
    training=pd.read_csv(root/'training_groups.csv');count=len(json.loads((root/'members.json').read_text()))
    before=int(selected.total_candidates.sum());before_eq=int(selected.available_equilibrium.sum());after=int(selected.selected.sum());after_eq=int(selected.selected_equilibrium.sum())
    test_long=summary[(summary.split=='test')&np.isclose(summary.horizon,.01)].set_index('method')
    claim='100개 medoid와 평균의 개선 여부는 아래 조건별 수치와 실패 양상을 함께 해석해야 한다.'
    comparisons=[]
    for rule in ['mean','medoid']:
        a=test_long.loc[f'{rule}10'];b=test_long.loc[f'{rule}100']
        comparisons.append(f'- {rule}: 10→100개에서 점화 구분 {100*a.ignition_classification:.1f}%→{100*b.ignition_classification:.1f}%, T profile 오차 평균 {a.temperature_profile_mean:.1f}→{b.temperature_profile_mean:.1f} K, 완주율 {100*a.completion_fraction:.1f}%→{100*b.completion_fraction:.1f}%.')
    tests=Path('results/long100_all_tests.log').read_text().splitlines()[-1]
    text=f'''# 100-member ensemble: 10 ms training and a posteriori evaluation

## 실행 범위

사용자 요청대로 모델 수를 **100개**, 훈련·0-D 평가 시간을 **10 ms**로 늘렸다. 원본 데이터와 과거 결과는 그대로 보존했다.
이번 훈련 데이터는 10 ms Cantera 궤적에서 평형을 downsampling한 새 데이터이며, perturbation은 추가하지 않았다.
데이터와 정규화가 바뀌므로 기존 0.4-ms 모델을 재사용하지 않고 **100개 모두 새로 학습**했다.
같은 새 데이터에서 학습한 첫 10개를 nested 대조군으로 삼았다. 예전 10개와 이번 100개를 직접 비교해 모델 수만의 효과라고 주장하지 않는다.

**모든 supervised pair와 모든 ANN step은 dt=2 μs**다. 입력 시간은 선별되어 불규칙하지만 target은 항상 정확히 한 step 뒤다.
전체 profile 평가는 5,000개 연속 전이를 모두 실행했다. 평가를 downsample하지 않았고 중간 Cantera 정답 상태를 입력하지 않았다.

{claim}

{chr(10).join(comparisons)}

## 평형 downsampling

- Train/validation/test 초기조건은 기존과 같은 54/18/18개다. 초기 T 900–1600 K, P 1–5 atm, 당량비 .5–2의 9종 수소 반응계다.
- Dense train 후보 {before:,}쌍 중 **{after:,}쌍**을 선택했다. 평형 비중은 **{100*before_eq/before:.1f}%→{100*after_eq/after:.1f}%**다.
- 각 train 궤적에서 non-equilibrium/induction 최대 320개, equilibrium anchor 최대 8개를 남겼다.
- 평형 판정은 단순 작은 dT/dt가 아니다. 같은 초기 엔탈피·압력·원소 inventory의 Cantera HP 평형과 비교해 T 차이 ≤1 K, 종별 상대차이 ≤5%(floor 1e-8), 종별 L1차이 ≤1e-4를 25개 연속 전이 동안 만족해야 한다.
- 반응이 거의 없는 초기 induction 상태는 HP 평형에서 멀면 non-equilibrium으로 유지한다.
- 시작 상태, 시간 층화, 농도·온도 변화의 arc length를 함께 사용해 긴 induction 구간과 빠른 반응 front를 보존했다.
- 평형 샘플을 모두 제거하면 equilibrium 부근의 잔차/장기 drift를 학습하기 어려워지므로 소수 anchor를 남겼다.

[데이터 보고서](LONG_DATA_REPORT.md)와 [선택 시점 CSV](results/long10ms_data/selected_inputs.csv)에 모든 parent ID·입력/정답 시점을 남겼다.
Sampler와 정규화는 train만 사용한다. validation/test 궤적은 10 ms 전체를 평가한다.

## 동일 구조·동일 학습 예산

모든 모델은 64–64–64 tanh DNN, Box–Cox γ=.1, Adam, batch 512, 10,000 optimizer update다.
시드는 11,22,…,1100의 서로 다른 100개다. 총 모델 파라미터는 100×9,738개이고 멤버별 가중치는 독립적이다.
원래 데이터/Box–Cox 선언과 달리 종별 signed rate를 학습하는 것이 아니라 다음 T,Y를 학습한다.

계산 효율을 위해 8개씩 grouped training했지만 초기화·미니배치 RNG·Adam 모멘트·gradient norm clip은 멤버별로 분리했다.
Loss는 멤버 차원에서 평균하지 않고 합산하여 group 크기로 gradient가 줄어들지 않게 했다. 독립 실행과 같은 update/RNG를 쓰는지 수치 테스트했다.
전체 represented budget은 **{count*10000:,} member-updates**다. 100개 ensemble 총 비용이 10개와 같다고 주장하지 않는다.
실제 grouped training wall time 합은 **{training.seconds.sum():.1f}초**, 전체 실행 누적 wall time은 **{manifest['total_execution_seconds']:.1f}초**다.
GPU/HPC는 사용하지 않았으며 사용자 승인 CPU 최대 90분 범위 안에서 수행했다.

## 평균과 medoid

매 step 같은 현재 상태를 모든 멤버에 넣는다. 각 제안은 [0,1] decoder guard, 양수·H/O/N 보존 투영, 현재 엔탈피/정압 조건의 온도 복원을 거친다.
평균 방식은 유효 후보 Y를 평균한 뒤 엔탈피 온도를 다시 복원한다. Medoid는 training-scaled T/Y 거리 합이 최소인 **실제 후보 하나**를 선택한다.
거리 정의는 앞선 validation에서 사용한 scaled 방식을 고정하고, 그 scale 값만 새 train 데이터에서 다시 적합했다. 이번 test 결과로 metric을 바꾸지 않았다.
대표 교체 hysteresis·adaptive dt·chemical ODE fallback은 없다. 전부 무효면 실패로 기록한다.

10개/100개와 mean/medoid의 네 가지 policy를 별도 self-fed 궤적으로 실행했다. 각 policy 내부의 모든 모델은 같은 현재 상태를 받으며, 다른 policy의 상태를 공유하지 않는다.
이 비교는 하나의 고정 100-member 집합과 그 첫 10개 subset이다. 여러 독립 100-member ensemble을 반복 학습한 통계라고 해석하지 않는다.

## 10 ms validation

{result_table('val',.01)}

## 10 ms held-out test

{result_table('test',.01)}

각 행은 같은 test 초기조건 18개다. 온도/종 profile 오차는 시간적분 RMSE를 궤적별 계산한 뒤 평균했다. Pooled T는 squared error를 모든 완주 궤적에 합쳐 계산한 값이다.
미완주 전체 profile error는 NaN이며 짧은 prefix 오차를 전체 정확도로 바꾸지 않았다. 완주/물리 허용 비율과 반드시 함께 읽어야 한다.
τ400 중앙값·P95는 reference와 ANN이 모두 점화한 쌍에서만 계산하며 쌍 수를 함께 표기했다.

### 점화 혼동행렬

{confusion_table}

점화 전에 실패한 계산은 정상 미점화/TN으로 세지 않는다. Reference가 10 ms 안에 점화하지 않으면 right-censored다.
평가 시간을 2→10 ms로 늘리면 reference의 점화/미점화 label 자체도 달라질 수 있다. 과거 2-ms 가짜 점화가 단순히 긴 창에서 true가 되었다고 지연시간이 정확해진 것은 아니다.
따라서 지연 오차의 tail과 전체 profile을 함께 제공한다. τ400 기준 reference만 0.2 μs로 정밀 측정했고, ANN 및 데이터 dt는 계속 2 μs다.
최대 dT/dt 기준 지연 오차는 `ignition_comparison.csv`에 별도로 있다.

## 같은 새 모델의 처음 2 ms

{result_table('test',.002)}

위 표는 같은 10-ms self-fed 실행의 앞 2 ms를 평가한 값이다. 이전 0.4-ms 훈련 연구와 데이터 분포가 다르므로 직접적인 모델 수 ablation으로 섞지 않았다.

## 구간별 profile과 대표 교체

Induction, reacting/relaxation, equilibrium을 reference trajectory로 나눠 시간점별 T/Y RMSE와 주요종/미량종 오차를 `regime_errors.csv`에 제공했다.
Induction은 reference의 T0+400 K 이전이며, 반응/이완은 그 이후 평형 마스크 밖, 평형은 위의 HP 평형 기준을 만족하는 구간이다.
장시간 평형 데이터가 전체 시간 평균을 지배해 빠른 점화/미량종 오차를 가리는 것을 피하기 위한 추가 평가다.
이 regime 구분은 평가에만 사용하며 모델 입력이나 test 기반 훈련 선택에는 쓰지 않았다.

대표 seed와 유효 후보 수를 모든 step에서 저장했다. 교체 빈도는 `selection_statistics.csv`에 있다.
100개로 늘리면 common bias가 사라진다고 보장할 수 없다. 대표가 자주 바뀌는 것이 관측되어도 오차의 원인인지 판단하려면 별도 hysteresis 통제 실험이 필요하다.

### 보존성과 올바른 chemistry는 다르다

100개 평균/medoid의 일부 test 궤적에서 온도가 각각 약 388 K/533 K까지 떨어졌다. 그림의 축도 이를 가리지 않도록 설정했다.
현재 물리 허용 지표는 양수성·질량/원소·온도 범위에 대한 것이며 반응 경로나 엔트로피 조건까지 보장하지 않는다.
단열·정압 closed reactor의 specific entropy는 감소하지 않아야 하므로, 추가로 Cantera 열역학에서 S(t)를 계산해 진단했다. 감소 판정 tolerance는 1e-6 J/(kg K)다.

{entropy_table}

이 진단은 model 선택이나 candidate filtering에 사용하지 않았다. 실제 Cantera 기준해의 감소는 roundoff 수준이지만 ANN의 감소는 훨씬 크다.
따라서 18/18 완주를 배포 가능한 안정성으로 해석할 수 없다. 더 많은 멤버가 shared model bias나 equilibrium 부근의 잘못된 동역학을 자동으로 고치지 못한 결과다.
평형 anchor 수/가중치, off-trajectory 학습과 entropy-aware 제약은 후속 validation 통제 실험 대상이며, 이번 test를 본 뒤 조용히 변경하지 않았다.

## 물리 보정 최적화의 검증

100개 후보별 보정 비용을 줄이기 위해 같은 weighted element projection을 batch 계산하고, 메커니즘의 NASA-7 계수로 ideal-gas enthalpy/cp와 HPY 온도 root를 vectorize했다.
이는 학습된 thermo 근사나 chemical ODE 호출이 아니다. 불안정한 projection/root는 원래 scalar projection/Cantera 열역학으로 처리한다. 화학 ODE fallback은 여전히 없다.

실제 초기·반응 중·평형 상태에서 나온 후보 864개를 원래 구현과 비교했을 때 최대 Y 차이 **1.11e-16**, T 차이 **2.88e-6 K**였고, 100개×18상태 보정 벤치마크는 약 13배 빨라졌다.
양수성·원소·엔탈피는 출력 구조로 강제한 것이며 ANN이 보존법칙을 학습했다고 주장하지 않는다.
다른 물리 backend의 미세 부동소수점 차이가 장기 민감도에 영향을 줄 수 있으므로 이번 10/100 비교에는 같은 vector backend를 공통 적용했다.

{cost_table}

시간은 이 CPU prototype의 진단 포함 rollout 시간이다. 이 표는 Cantera 대비 가속률이 아니며, 1000배 가속을 측정했다는 뜻도 아니다.

## 검증과 재현

테스트 결과: **{tests}**. NASA h/cp와 Cantera 일치, scalar/batch projection 일치, HPY root, model별 gradient clipping과 update 독립성, medoid chunking, downsampling의 train-only parent 및 고정 target dt를 검사했다.
원래 dataset와 과거 모델/결과는 변경하지 않았다. 새 dense/pair SHA256은 데이터 manifest, 100개 모델 SHA256은 members.json에 있다.

- [데이터 생성/선별](kinetics/long_data.py), [grouped 학습](kinetics/batched_training.py), [배치 물리 보정](kinetics/vector_physics.py)
- [설정](configs/committee100_10ms.json), [실험 runner](kinetics/long_committee_study.py), [보고서 생성](kinetics/long_committee_report.py)
- [요약 CSV](results/committee100_long10ms/summary.csv), [초기조건별 지표](results/committee100_long10ms/trajectories.csv), [점화/꼬리 오차](results/committee100_long10ms/ignition_comparison.csv), [혼동행렬](results/committee100_long10ms/confusion.csv)
- [종별 profile](results/committee100_long10ms/species_profiles.csv), [regime 오차](results/committee100_long10ms/regime_errors.csv), [대표 교체](results/committee100_long10ms/selection_statistics.csv), [시간](results/committee100_long10ms/timing.csv)
- [엔트로피 진단](results/committee100_long10ms/entropy_diagnostics.csv), [100개 모델 및 재로딩 검증](results/committee100_long10ms/reload_checks.csv)
- [모든 온도 profile](results/committee100_long10ms/figures/all_temperature_profiles.png), [어려운 초기조건](results/committee100_long10ms/figures/difficult_initial_conditions.png), [9종 profile](results/committee100_long10ms/figures/species_profiles.png), [점화지연](results/committee100_long10ms/figures/ignition_delay.png), [regime별 오차](results/committee100_long10ms/figures/regime_errors.png)

단계별 재실행은 README.md를 따른다. 기존 output은 보존하고 config output을 새 경로로 지정한다. `--resume`는 source/model hash와 config를 확인하여 완료된 그룹과 평가를 재사용한다.
향후 작업 브랜치는 사용자 요청에 따라 kisung이며, 초기 검증 완료 버전은 main에 게시한다.
'''
    Path('LONG_COMMITTEE_REPORT.md').write_text(text)
    (root/'report_manifest.json').write_text(json.dumps(dict(status='completed',report='LONG_COMMITTEE_REPORT.md',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('kinetics').glob('*.py'))}),indent=2))
    print('Long committee report written')


if __name__=='__main__':main()
