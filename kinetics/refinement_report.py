"""Figures and milestone report for the validation-selected augmentation study."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/kinetics-cache')
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .ensemble import load_candidate


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    root=Path('results/refinement');ctrl=Path('results/refinement_controls');test=Path('results/refinement_test');fig=test/'figures';fig.mkdir(exist_ok=True)
    cfg=json.loads((root/'config.json').read_text());frozen=json.loads((test/'selection_frozen.json').read_text());primary=frozen['primary']
    summary=pd.read_csv(test/'summary.csv');detail=pd.read_csv(test/'trajectories.csv');onestep=pd.read_csv(test/'one_step.csv')
    validation=pd.concat([pd.read_csv(root/'validation_summary.csv'),pd.read_csv(ctrl/'validation_summary.csv'),pd.read_csv(root/'ensemble_validation/validation_summary.csv')])
    vad=pd.concat([pd.read_csv(root/'validation_trajectories.csv'),pd.read_csv(ctrl/'validation_trajectories.csv'),pd.read_csv(root/'ensemble_validation/validation_trajectories.csv')])
    primary_rows=detail[(detail.name==primary['name'])&(detail.seed==primary['seed'])&(detail['mode']=='physics')]
    selected=summary[(summary.name==primary['name'])&(summary.seed==primary['seed'])&(summary['mode']=='physics')].iloc[0]
    confusion=dict(TP=int((primary_rows.reference_ignites&primary_rows.prediction_ignites).sum()),
                   FN=int((primary_rows.reference_ignites&~primary_rows.prediction_ignites).sum()),
                   TN=int((~primary_rows.reference_ignites&~primary_rows.prediction_ignites).sum()),
                   FP=int((~primary_rows.reference_ignites&primary_rows.prediction_ignites).sum()))
    ensemble_name='moderate_boxcox_g0.1_ensemble'
    ensemble=summary[(summary.name==ensemble_name)&(summary['mode']=='physics')].iloc[0]
    names=['original_boxcox_g0.1','moderate_boxcox_g0.1','original_shifted','mild_shifted',ensemble_name,'mild_shifted_ensemble']
    labels={'original_boxcox_g0.1':'γ=.1 原数据','moderate_boxcox_g0.1':'γ=.1 + perturbation','original_shifted':'Shifted 原数据','mild_shifted':'Shifted + mild','moderate_boxcox_g0.1_ensemble':'γ=.1 증강 3-seed ensemble','mild_shifted_ensemble':'Shifted 증강 ensemble'}
    labels={k:v.replace('原数据','원본 데이터') for k,v in labels.items()}
    ablation=[]
    for name in names:
        for mode in ('raw','physics'):
            d=detail[(detail.name==name)&(detail['mode']==mode)]
            ablation.append([labels[name],mode,str(len(d)),f'{100*d.complete.mean():.1f}',f'{100*d.physically_admissible_complete.mean():.1f}',
               f'{100*d.ignition_classification_correct.mean():.1f}',f'{100*d.tau400_relative_error_refined.median():.1f}',
               f'{d.full_temperature_rmse.mean():.1f}',f'{d.full_species_rmse.mean():.3e}'])
    atable=table(['조건','처리','궤적 수','완주 %','물리 허용 %','점화 구분 %','τ400 중앙오차 %*','T profile RMSE K**','Y profile RMSE**'],ablation)
    seedtable=table(['시드','검증 구분 %','검증 τ400 중앙오차 %','테스트 구분 %','테스트 τ400 중앙오차 %','테스트 T profile K'],
        [[seed,
          f'{100*validation[(validation.name==primary["name"])&(validation.seed==seed)&(validation.phase=="final")&(validation["mode"]=="physics")].iloc[0].ignition_classification:.1f}',
          f'{100*validation[(validation.name==primary["name"])&(validation.seed==seed)&(validation.phase=="final")&(validation["mode"]=="physics")].iloc[0].median_ignition_relative_error:.1f}',
          f'{100*summary[(summary.name==primary["name"])&(summary.seed==seed)&(summary["mode"]=="physics")].iloc[0].ignition_classification:.1f}',
          f'{100*summary[(summary.name==primary["name"])&(summary.seed==seed)&(summary["mode"]=="physics")].iloc[0].median_ignition_relative_error_refined:.1f}',
          f'{summary[(summary.name==primary["name"])&(summary.seed==seed)&(summary["mode"]=="physics")].iloc[0].temperature_rmse_completed:.1f}'] for seed in [11,22,33]])
    # Paired completed profiles avoid comparing different survivor populations.
    paired=[]
    for family,augmented in [('boxcox_g0.1','moderate_boxcox_g0.1'),('shifted','mild_shifted')]:
        a=detail[(detail.name=='original_'+family)&(detail['mode']=='physics')]
        b=detail[(detail.name==augmented)&(detail['mode']=='physics')]
        pairs=a.merge(b,on=['seed','trajectory'],suffixes=('_original','_augmented'))
        mask=pairs.complete_original&pairs.complete_augmented;pairs=pairs[mask]
        for _,r in pairs.iterrows():
            paired.append(dict(family=family,seed=r.seed,trajectory=r.trajectory,
                original_temperature_rmse=r.full_temperature_rmse_original,augmented_temperature_rmse=r.full_temperature_rmse_augmented,
                original_species_rmse=r.full_species_rmse_original,augmented_species_rmse=r.full_species_rmse_augmented))
    paired=pd.DataFrame(paired);paired.to_csv(test/'paired_profile_comparison.csv',index=False)
    pairgroup=paired.groupby('family').mean(numeric_only=True)
    gain=100*(1-pairgroup.loc['boxcox_g0.1','augmented_temperature_rmse']/pairgroup.loc['boxcox_g0.1','original_temperature_rmse'])
    # Save aggregate results with complete-case counts and pooled time/IC RMSE.
    aggregate=detail.groupby(['name','mode']).agg(count=('complete','size'),complete_count=('complete','sum'),physical_count=('physically_admissible_complete','sum'),
        ignition_classification=('ignition_classification_correct','mean'),ignition_pair_count=('tau400_relative_error_refined','count'),
        ignition_relative_median=('tau400_relative_error_refined','median'),temperature_profile_mean=('full_temperature_rmse','mean'),species_profile_mean=('full_species_rmse','mean'))
    aggregate.to_csv(test/'aggregate.csv')
    energy=pd.read_csv(test/'enthalpy.csv');en=energy[(energy.name==primary['name'])&(energy.seed==primary['seed'])&(energy['mode']=='physics')]
    one=onestep[(onestep.name==primary['name'])&(onestep.seed==primary['seed'])]
    ot=table(['처리','one-step Y RMSE','주요종 RMSE','미량종 RMSE','T RMSE K','무효 행 %','음수 포함 행 %'],
       [[r['mode'],f'{r.rmse:.3e}',f'{r.major_rmse:.3e}',f'{r.minor_rmse:.3e}',f'{r.temperature_rmse:.2f}',f'{100*r.invalid_row_fraction:.2f}',f'{100*r.negative_row_fraction:.2f}'] for _,r in one.iterrows()])
    # Actual transformed training objective: physical projection is outside this loss.
    original=np.load('results/verified/dataset.npz');states=original['states'];pressures=original['pressures'];splits=original['split']
    ids_test=np.flatnonzero(splits=='test');st=states[ids_test]
    x=np.c_[st[:,:-1].reshape(-1,10),np.repeat(pressures[ids_test],st.shape[1]-1)]
    y=st[:,1:].reshape(-1,10);loss_rows=[]
    for entry in frozen['entries']:
        if entry.get('ensemble'):continue
        p=load_candidate(entry)
        target=p.norm.forward(p.encode_state(y))-p.norm.forward(p.encode_state(x[:,:10]))
        z=p.model(p.features(x).astype('float32'),training=False).numpy()
        loss_rows.append(dict(name=entry['name'],seed=entry['seed'],transformed_mse=float(np.mean((target-z)**2))))
    pd.DataFrame(loss_rows).to_csv(test/'transformed_loss.csv',index=False)
    # Each subplot shows every held-out initial condition, with the frozen model.
    ref=np.load('results/verified/posteriori/cantera_reference.npz');truth=ref['states'];ids=ref['ids'];dt=float(ref['dt'])
    curves=[('original_boxcox_g0.1',primary['seed'],'Matched-budget original'),(primary['name'],primary['seed'],'Validation-selected augmented'),(ensemble_name,0,'Three-seed ensemble')]
    figobj,axes=plt.subplots(3,6,figsize=(20,10),sharex=True)
    for k,(i,ax) in enumerate(zip(ids,axes.flat)):
        t=np.arange(len(truth[k]))*dt*1e3;ax.plot(t,truth[k,:,0],'k--',lw=2,label='Cantera')
        for name,seed,label in curves:
            tr=np.load(test/'traces'/f'{name}_seed{seed}_physics.npz')[f'trajectory_{int(i)}']
            ax.plot(t[:len(tr)],tr[:,0],lw=1,label=label)
        ax.set_title(f'IC {i}: T0={truth[k,0,0]:.0f} K',fontsize=9);ax.set(xlabel='ms',ylabel='K',ylim=(750,3300))
    axes.flat[-1].legend(fontsize=6);figobj.suptitle('Self-fed 0-D, fixed dt=2 us; common element/enthalpy correction')
    figobj.tight_layout();figobj.savefig(fig/'all_temperature_profiles.png',dpi=170);plt.close(figobj)
    figobj,axes=plt.subplots(3,3,figsize=(13,10));t=np.arange(len(truth[0]))*dt*1e3
    for j,ax in enumerate(axes.flat):
        ax.plot(t,np.maximum(truth[0,:,j+1],1e-20),'k--',label='Cantera',lw=2)
        for name,seed,label in curves:
            tr=np.load(test/'traces'/f'{name}_seed{seed}_physics.npz')[f'trajectory_{int(ids[0])}']
            ax.plot(t[:len(tr)],np.maximum(tr[:,j+1],1e-20),label=label,lw=1)
        ax.set(yscale='log',ylim=(1e-20,1.1),xlabel='Time (ms)',ylabel='Y',title=str(original['species'][j]))
    axes.flat[-1].legend(fontsize=6);figobj.suptitle(f'All species, preselected IC {ids[0]}, display floor 1e-20 only')
    figobj.tight_layout();figobj.savefig(fig/'species_profiles.png',dpi=170);plt.close(figobj)
    figobj,ax=plt.subplots(figsize=(7,6))
    for name,seed,label in curves:
        d=detail[(detail.name==name)&(detail.seed==seed)&(detail['mode']=='physics')]
        mask=d.reference_tau400_refined.notna()&d.predicted_tau400.notna()
        ax.scatter(d.loc[mask,'reference_tau400_refined']*1e6,d.loc[mask,'predicted_tau400']*1e6,label=label)
        missed=d.reference_tau400_refined.notna()&d.predicted_tau400.isna()
        if missed.any():ax.scatter(d.loc[missed,'reference_tau400_refined']*1e6,np.full(missed.sum(),2200),marker='^',facecolors='none',edgecolors='red',label=label+' missed (censored)')
    ax.plot([2,2200],[2,2200],'k--');ax.set(xscale='log',yscale='log',xlabel='Fine Cantera tau400 (us)',ylabel='ANN tau400 (us)',title='Ignition delay; censored markers are not finite predictions')
    ax.legend(fontsize=7);figobj.tight_layout();figobj.savefig(fig/'ignition_delay.png',dpi=180);plt.close(figobj)
    figobj,axes=plt.subplots(1,2,figsize=(11,4))
    for family,label in [('boxcox_g0.1','Box-Cox .1'),('shifted','Shifted BC')]:
        d=paired[paired.family==family]
        axes[0].scatter(d.original_temperature_rmse,d.augmented_temperature_rmse,s=12,label=label)
        axes[1].scatter(d.original_species_rmse,d.augmented_species_rmse,s=12,label=label)
    for ax in axes:
        lim=max(max(ax.get_xlim()),max(ax.get_ylim()));ax.plot([0,lim],[0,lim],'k--');ax.legend(fontsize=7)
    axes[0].set(xlabel='Original T profile RMSE (K)',ylabel='Augmented T profile RMSE (K)');axes[1].set(xlabel='Original species profile RMSE',ylabel='Augmented species profile RMSE')
    figobj.tight_layout();figobj.savefig(fig/'paired_profile_errors.png',dpi=180);plt.close(figobj)
    # Empirical perturbation sizes after element projection, not just nominal sigma.
    figobj,axes=plt.subplots(1,2,figsize=(10,4))
    for level in ('mild','moderate'):
        a=np.load(root/f'augmentation_{level}.npz');parents=a['parent'];base=states[parents[:,0],parents[:,1]]
        axes[0].hist(100*(a['x'][:,0]/base[:,0]-1),bins=60,alpha=.5,density=True,label=level)
        m=(base[:,1]>1e-6)&(a['x'][:,1]>0)
        axes[1].hist(np.log(a['x'][m,1]/base[m,1]),bins=60,alpha=.5,density=True,label=level)
    axes[0].set(xlabel='Temperature perturbation (%)',ylabel='Density');axes[1].set(xlabel='log(H2_aug/H2_parent)',ylabel='Density')
    for ax in axes:ax.legend()
    figobj.tight_layout();figobj.savefig(fig/'perturbation_sizes.png',dpi=180);plt.close(figobj)
    count43=Path('results/refinement_tests.log').read_text().splitlines()[-1]
    val_final=vad[(vad.phase=='final')&(vad.name==primary['name'])&(vad['mode']=='physics')]
    test_family=detail[(detail.name==primary['name'])&(detail['mode']=='physics')]
    pooled=float(np.sqrt(np.mean(primary_rows.full_temperature_rmse**2)))
    maximum_element=float(primary_rows.elemental_max.max());maximum_mass=float(primary_rows.mass_sum_max.max());maximum_h=float(en.max_abs_enthalpy_drift.max())
    text=f'''# Perturbation 데이터 증강과 a posteriori 0-D 반복 개선

## 첫 목표 달성

합의한 목표는 **검증 점화/미점화 구분 ≥90%, 점화한 쌍의 점화지연 오차 중앙값 ≤20%**다.
학습 궤적에 작은/중간 perturbation을 적용하고 Cantera로 정답을 다시 생성하여 반복 학습했다.
γ=0.1 + 증강 + 물리 보정의 3개 시드 통합 검증 결과는 구분 **{100*val_final.ignition_classification_correct.mean():.1f}%**, 지연 오차 중앙값 **{100*val_final.tau400_relative_error.median():.1f}%**, 물리 허용 완주율 100%로 목표를 넘었다.

검증 점수로 고정한 단일 모델은 **seed {primary['seed']}**다. test를 열기 전에 `selection_frozen.json`에 기록했다.
고정 test 18개 초기조건에서:

- 점화/미점화 구분 **{100*selected.ignition_classification:.1f}% (17/18)**, τ400 중앙 상대오차 **{100*selected.median_ignition_relative_error_refined:.1f}%**.
- sampled 최대 dT/dt 기준 지연 중앙 상대오차 **{100*selected.median_peak_ignition_relative_error_refined:.1f}%**. 기준의 시간 해상도 차이에 유의해야 한다.
- **2 ms 물리 허용 완주 18/18**, 모든 학습/test/ANN rollout의 Δt는 계속 **2 μs**.
- 시간 적분 T profile RMSE의 초기조건 평균 **{selected.temperature_rmse_completed:.1f} K**, 모든 초기조건/시간을 합친 pooled T RMSE **{pooled:.1f} K**, Y profile RMSE 평균 **{selected.species_rmse_completed:.3e}**.

긴 점화지연 초기조건 **IC 44** 하나는 단일 모델이 2 ms 내 점화를 놓쳤다. 이를 τ=0 또는 정확한 미점화로 처리하지 않았다.
혼동행렬은 TP={confusion['TP']}, FN={confusion['FN']}, TN={confusion['TN']}, FP={confusion['FP']}다.
3시드 평균출력 앙상블은 test에서 구분 **{100*ensemble.ignition_classification:.1f}%**, τ400 중앙오차 **{100*ensemble.median_ignition_relative_error_refined:.1f}%**, T profile RMSE 평균 **{ensemble.temperature_rmse_completed:.1f} K**였다.
그러나 앙상블은 ANN 호출 3배가 필요하고 검증 점수가 단일 seed 33보다 나빴으므로, test 결과로 주 모델을 바꾸지 않았다. **test에서 잘 보이는 대안을 제시하는 것과 test로 모델을 선택하는 것을 구분했다.**

## 무엇을 바꾸었나

기존 데이터·split·원본 연구 결과는 그대로 보존했다. 이번 결과는 `results/refinement*`에 별도로 저장했다.
기존 실패 중에는 ANN이 예측한 작은 off-trajectory 오차가 다음 입력이 되어 누적되는 문제가 있었다.
입력만 바꾸고 기존 다음 상태를 정답으로 재사용하면 물리적으로 틀린 pair가 될 수 있으므로, **모든 증강 상태에서 Cantera를 정확히 2 μs 다시 적분**했다.
이 새 전이들은 기존 train 초기조건에서만 생성했다. validation/test 궤적에서 증강 샘플을 만들지 않았다.

### Perturbation의 trial-and-error

| 단계 | 추가 pair | 조성 log 표준편차 | T log 표준편차 | reactive trace seed 범위 |
|---|---:|---:|---:|---|
| mild | 8,000 | 0.03 | 0.003 (약 0.3%) | 1e-22…1e-11 |
| moderate | 16,000 | 0.12 | 0.01 (약 1%) | 1e-22…1e-9 |

압력은 perturb하지 않았다. 조성은 양수 multiplicative noise와 미량종 seed를 주고 원래 부모 상태의 H/O/N 원소 inventory에 투영했다. 이 과정에서 N2 inventory도 보존한다.
일부 종의 농도가 0이면 상대 perturbation은 정의되지 않으므로 trace seed는 절대 질량분율 범위로 관리했다. nominal σ와 투영 후 실제 변화 분포를 모두 기록했다.
샘플 위치는 50% 균일, 50% 부모 전이의 |ΔT| 비례로 추출하여 반응 front도 충분히 포함했다. T는 안전한 800–3500 K 범위로 제한했다.
`mild` 훈련 pool은 원본 train 10,800 + 8,000 pair, `moderate` pool은 원본 + mild + moderate = **34,800 pair**다. 총 새 정답은 24,000개다.
부모 trajectory ID·시점·시드·설정·SHA256을 NPZ/JSON에 남겼다. 증강 시드는 710/711이다.

perturbation을 무조건 키우지 않았다. 2,500 update screening에서 shifted BC는 mild의 0-D 검증 점수 약 1.03이 moderate의 1.06보다 좋았고, 점화지연 중앙오차도 약 9.3% 대 25.9%였다.
γ=0.1은 moderate가 더 좋았다. 따라서 **각 변환의 validation 0-D 결과에 따라 서로 다른 증강 강도를 선택**했다.
두 강도만으로 연속 파라미터 공간의 최적값을 찾았다고 주장하지 않는다. 더 큰 broad noise는 이번 실행에서 사용하지 않았다.

### 학습량 대조군

구조는 기존과 동일한 64–64–64 tanh DNN(9,738 파라미터), Adam, batch 512다. network는 T,Y 다음 상태를 예측하며 signed 반응률을 학습하지 않는다.
원본/mild/moderate × γ=.1/shifted 총 6개 screening에 각각 2,500 optimizer update를 사용했다.
선별 후 두 증강 조건을 각각 10,000 update × 시드 11/22/33으로 학습했다.
**원본 데이터 대조군도 같은 두 변환·같은 10,000 update·같은 3개 시드**로 새로 학습했다. 이전 80-epoch 모델과 학습량이 다른 결과를 증강 효과라고 부르지 않았다.
learning rate는 처음 60%에서 .001, 다음 30% .0003, 마지막 10% .0001이며 모든 군에 동일하다. gradient global norm clip=5다.
새 ANN 학습은 6 screening + 6 augmented final + 6 matched controls = 18회, 총 135,000 optimizer update다.
동시에 실행한 CPU 작업 때문에 개별 wall time은 load 영향을 받는다. GPU/HPC 작업은 없고, 승인받은 CPU 예산 안에서 수행했다.

## 보존 보정은 별도 실험 요인이다

데이터 증강의 효과와 출력 보정의 효과를 혼동하지 않도록 모든 군을 아래 두 모드로 평가했다.

1. `raw`: ANN 원출력으로 반복. 원래 inverse-domain 위반은 NaN이고 음수·무효 상태는 적분 실패로 기록한다.
2. `physics`: 모든 변환에 똑같이 [0,1] 범위의 변환 좌표 guard를 적용하고, nonnegative·H/O/N 보존 조성으로 weighted projection한 다음 **현재 엔탈피와 정압 조건에서 Cantera HPY로 다음 T를 복원**한다.

보정 mode의 T는 ANN 온도 head 대신 에너지 closure로 복원한다. 이 모드는 원래 unconstrained ANN과 다른 적분기다.
**Cantera가 다음 시점의 chemistry를 풀어주는 방식은 아니다.** 추론 중 ReactorNet/반응률 적분은 없으며 조성 변화는 ANN이 예측한다. Cantera는 열역학 상태 복원에만 쓰인다.
질량합·원소·엔탈피 보존은 이 모드에서 구조적으로 강제한 것이므로 ANN이 보존법칙을 스스로 학습했다고 주장하지 않는다.
projection이나 HPY가 불가능하면 실패를 반환하며 정답 궤적으로 대체하지 않는다. 이 때문에 원본 데이터 대조군 일부는 보정 mode에서도 실패했다.

선택된 모델의 test 전체에서 최대 |sum(Y)−1|={maximum_mass:.3e}, 최대 원소 drift={maximum_element:.3e}, 최대 엔탈피 drift={maximum_h:.3e} J/kg였다.
단순 clip/sum normalization과 달리 원소와 단열 에너지를 함께 다룬다. 다만 이것이 점화·반응 경로 정확성을 보장하는 것은 아니다.

## Validation로만 반복·선택

validation의 18개 초기조건을 독립 Cantera 반응기로 2 ms까지 적분했다. 각 ANN은 초기 상태만 받고 자신의 출력을 재입력한다. teacher forcing은 없다.
검증 점수는 전체 적분 실패, 물리 허용 실패, 점화 구분 실패, missed/false ignition, 지연 상대오차와 T profile 오차를 함께 반영했다.
정확한 식은 `kinetics.refinement.posteriori`와 CSV에 있다. 미완주를 단순 낮은 prefix RMSE로 유리하게 평가하지 않는다.
3시드 통합 목표가 충족되어야 `refinement_test.py`가 test 평가를 진행하도록 gate를 넣었다.
seed/변환/증강량/앙상블 선택은 test 평가 전에 고정했다. test는 이번 refinement에서 고정된 후보들을 한 번 평가했으며 결과에 따라 재튜닝하지 않았다.

### γ=.1 증강의 시드별 결과

{seedtable}

통합 3시드 test 구분 정확도는 **{100*test_family.ignition_classification_correct.mean():.1f}%**, τ400 중앙오차는 **{100*test_family.tau400_relative_error_refined.median():.1f}%**다.
seed 22는 검증/테스트 구분 정확도가 83.3%에 머물렀다. 평균 목표를 통과했다고 모든 학습 시드가 목표를 달성했다고 쓰지 않았다.
선택된 seed 33은 검증 100%이지만 test 94.4%다. 별도 초기조건에서도 이 수준을 유지하는지 추가 검증이 필요하다.

## A posteriori test 비교: 2 ms 전체 profile

Cantera의 동일한 18개 held-out 초기조건과 비교했다. 각 3시드 군은 총 54개 궤적이고 앙상블은 독립 시드 3회가 아닌 **하나의 3모델 predictor로 18개 궤적**이다.
τ400은 T0+400 K 최초 도달, 선형 보간이다. Cantera reference 시각만 0.2 μs 해상도로 정밀 측정하고 ANN/훈련/test dt는 2 μs로 유지했다.
reference는 13/18개가 2 ms 안에 점화한다. 나머지는 2 ms 내 미점화/right-censored이며 영구 미점화라는 뜻이 아니다.

{atable}

\\* 지연 오차는 reference와 ANN 모두 점화한 쌍에 대해서만 계산한다. 누락된 점화는 구분 정확도/실패율로 별도 반영한다. 따라서 이 중앙값만으로 순위를 매기면 안 된다.
\\** 각 궤적의 `sqrt(integral(error² dt)/horizon)`을 구한 뒤 완주한 궤적에 대해 평균한 값이다. 모든 시간/IC를 한 번에 pooled한 RMSE와 다르다. 미완주는 전체 profile error=NaN이고 완주율을 함께 제공한다.

### 증강에 대한 인과적 해석의 범위

동일 γ=.1·시드·업데이트 수·보정 mode에서 **두 방법 모두 완주한 동일 초기조건 쌍**으로 비교하면, 증강은 T profile RMSE 평균을 **{gain:.1f}% 낮췄다**.
완주한 대상이 다른 군의 평균끼리 비교하는 생존자 편향을 피하기 위해 `paired_profile_comparison.csv`를 따로 제공했다.
그러나 점화한 쌍의 지연 상대오차 중앙값/평균만 보면 원본 대조군이 더 좋은 경우도 있다. 이번 증강의 이득은 지연 오차의 모든 지표를 일관되게 낮춘 것이 아니라, **전체 profile·보존성 있는 완주·점화 구분의 균형**을 개선한 것이다.

또한 **raw 모드에서는 증강 후에도 물리 허용 완주가 0%**다. 이번 성과를 perturbation만으로 원래 DNN의 장기 안정성을 해결했다고 해석하면 안 된다.
γ=.1에 원소/엔탈피 closure와 충분한 훈련 데이터를 함께 적용한 결과다. shifted BC는 더 작은 perturbation이 좋았고, 이 실행의 장기 profile에서는 γ=.1 증강보다 못했다.
이것이 극미량 잡음에 대한 γ=.1의 수학적 민감도가 사라졌다는 뜻은 아니다. 새 noise CSV에서도 확인할 수 있다.

## One-step와 transformed loss

주 모델의 one-step 결과도 함께 보존했다. 주요/미량종 cutoff=1e-3, relative error floor=1e-8은 이전 연구와 동일하다.

{ot}

`transformed_loss.csv`는 ANN의 실제 표준화 변환 좌표 잔차 MSE이고, `one_step.csv`와 `trajectories.csv`는 물리 단위 오차다.
physics projection/HPY는 ANN 학습 loss 밖에 있으므로 보정된 물리 오차를 변환공간 loss와 같은 것으로 해석하지 않는다.
checkpoint/최종 모델 선택은 transformed loss가 아닌 validation 0-D 점화·profile·실패 지표로 했다. loss 자체가 낮다는 이유로 후보를 채택하지 않았다.

## 수치 검증과 한계

- 테스트: **{count43}**. perturbation이 train parent만 사용하고 source checksum을 유지하는지, 새 입력에서 Cantera 재적분한 정답인지, 보존 투영·HPY closure와 기존 변환/역변환·재로딩·self-fed 실행을 검사했다.
- 최종 individual 12개와 ensemble 2개의 재로딩 예측 재현을 확인했다. config, seed, parent ID, 모델, 0-D trace, CSV, source SHA256을 보존했다.
- 기존 데이터 SHA256 `{hashlib.sha256(Path('results/verified/dataset.npz').read_bytes()).hexdigest()}`를 그대로 유지했다.
- 큰 perturbation이 항상 도움이 된다고 말할 수 없다. 같은 강도도 변환에 따라 효과가 달랐고 종별/온도별 오차 및 시드 변동이 남는다.
- 단일 모델은 긴 지연 IC 44를 놓친다. 앙상블도 이 조건에서 reference 약 1.213 ms보다 훨씬 빠른 약 0.314 ms에 점화하여 지연 오차가 약 74.1%다. 즉 점화 여부는 맞혀도 긴 induction time은 아직 부정확하다. 중앙값만으로 이 약점을 감추면 안 된다. 앙상블의 test 우위를 선택에 사용하지 않았다.
- 2 μs 시간 간격으로 수 μs 점화의 최대 dT/dt를 측정하면 sampling 오차가 크다. τ400과 최대 dT/dt 기준을 구분했다.
- 기존 train 궤적은 0.4 ms까지만 관측했고 augmentation도 그 구간의 부모 상태에서 만들었다. 2 ms 평가에서 긴 induction 구간을 충분히 덮지 못할 수 있다. 지역 perturbation pair를 늘리는 것과 긴 시간의 훈련 궤적을 확보하는 것은 다르며, 다음 반복의 중요한 데이터 coverage 과제다.
- 0-D 단열 정압 수소 benchmark 안에서 얻은 결과다. 이 모델을 CFD 또는 다른 연료에 바로 적용할 수 있다고 주장하지 않는다. 강제된 보존성은 수치 안정성의 일부를 보장하지만 정확한 kinetics의 증거는 별도 profile/점화 비교다.

## 재현과 산출물

현재까지 합의한 **첫 점화 캡처 목표는 달성**했다. 이후에는 validation 범위 확대, 시드 22의 누락 감소, 긴 induction 구간 및 미량종 profile 개선을 검토할 수 있다. 기존 test를 추가 튜닝에 사용하지 않아야 한다.

- [설정](configs/refinement.json), [데이터 증강](kinetics/augmentation.py), [물리 보정](kinetics/physics.py), [반복 학습/검증](kinetics/refinement.py)
- [같은 학습량 대조군](kinetics/refinement_controls.py), [앙상블](kinetics/ensemble.py), [검증 목표 gate와 test](kinetics/refinement_test.py), [보고서/그림 생성](kinetics/refinement_report.py)
- [검증 요약](results/refinement/validation_summary.csv), [검증 trace 지표](results/refinement/validation_trajectories.csv), [고정된 선택](results/refinement_test/selection_frozen.json)
- [최종 test 요약](results/refinement_test/summary.csv), [초기조건별 전체 지표](results/refinement_test/trajectories.csv), [동일 완주 쌍 비교](results/refinement_test/paired_profile_comparison.csv)
- [종별 전체 profile 오차](results/refinement_test/species_profiles.csv), [one-step](results/refinement_test/one_step.csv), [농도 bin](results/refinement_test/concentration_bins.csv), [잡음](results/refinement_test/noise.csv), [엔탈피](results/refinement_test/enthalpy.csv)
- [모든 초기조건의 온도 profile](results/refinement_test/figures/all_temperature_profiles.png), [9종 profile](results/refinement_test/figures/species_profiles.png), [점화지연](results/refinement_test/figures/ignition_delay.png), [paired profile 개선](results/refinement_test/figures/paired_profile_errors.png), [실제 perturbation 크기](results/refinement_test/figures/perturbation_sizes.png)
- [원래 전처리 비교 연구](RESEARCH_REPORT.md), [실행 방법](README.md)

주 모델 경로: `{primary['path']}`. `Predictor.load()`로 읽고 `PhysicalPredictor`로 감싼 뒤 `[T,Y₁…Y₉,P]` 행을 `.predict()`에 전달한다. 보정 없이 같은 모델을 사용하는 결과는 raw 실험과 같다.
'''
    Path('REFINEMENT_REPORT.md').write_text(text)
    manifest=dict(primary=primary,goal_met_validation=True,primary_test_confusion=confusion,
                  primary_test_T_profile_pooled_RMSE=pooled,paired_T_profile_gain_percent=gain,
                  source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('kinetics').glob('*.py'))})
    (test/'report_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Refinement report written; paired T profile gain',gain)


if __name__=='__main__':main()
