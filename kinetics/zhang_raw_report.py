"""Report the Zhang multi-scale/raw full-state pipeline, including final-time failures."""
import argparse,json,os,hashlib
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .preprocess_report import table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/zhang_multiscale_raw'))
    parser.add_argument('--report',type=Path,default=Path('ZHANG_MULTISCALE_RAW_REPORT.md'))
    args=parser.parse_args();root=args.root
    cfg=json.loads((root/'config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['physics'] is False
    summary=pd.read_csv(root/'summary.csv');detail=pd.read_csv(root/'trajectories.csv');one=pd.read_csv(root/'one_step.csv')
    frozen=json.loads((root/'all_selections_frozen.json').read_text())
    generation=Path(cfg['training_pairs']).parent;data_meta=json.loads((generation/'manifest.json').read_text())
    with np.load(cfg['source']) as d:source={k:d[k] for k in d.files}
    with np.load(root/'reference20ms.npz') as d:
        refs={int(i):r for i,r in zip(d['ids'],d['states'])}
    figures=root/'figures';figures.mkdir(exist_ok=True)
    keys=['regime','gamma','method','split','horizon']
    extra=detail.groupby(keys,dropna=False).agg(mean_coverage=('coverage','mean'),
        median_end_ms=('completed_steps',lambda s:float(np.median(s))*cfg['dt']*1000),
        median_first_violation_ms=('first_physical_violation_seconds',lambda s:float(s.median())*1000)).reset_index()
    summary=summary.merge(extra,on=keys,validate='one_to_one')
    summary.to_csv(root/'summary_with_coverage.csv',index=False)
    methods=dict(single_best='tab:red',mean10='tab:blue',medoid10='tab:orange')
    ids=np.flatnonzero(source['split']=='test');dt=cfg['dt']
    for definition in frozen:
        label=definition['label'];traces={}
        for method in methods:
            with np.load(root/label/'traces'/f'{method}_test.npz') as d:
                traces[method]={int(i):d[f'trajectory_{int(i)}'] for i in ids}
        fig,axes=plt.subplots(3,6,figsize=(22,11))
        for i,ax in zip(ids,axes.flat):
            reference=refs[int(i)]
            ax.plot(np.arange(1,len(reference))*dt*1e3,reference[1:,0],'k--',lw=1.6,label='Cantera')
            notes=[]
            for method,color in methods.items():
                a=traces[method][int(i)];times=np.arange(1,len(a))*dt*1e3
                if len(a)>1:
                    ax.plot(times,a[1:,0],color=color,lw=.9,label=method)
                    ax.plot(times[-1],a[-1,0],marker='x',color=color,ms=3)
                else:ax.plot([],[],color=color,label=method)
                notes.append(f'{method}: {(len(a)-1)*dt*1e3:.3g} ms')
            ax.text(.02,.97,'\n'.join(notes),transform=ax.transAxes,va='top',fontsize=5.5)
            ax.set(xscale='log',xlim=(dt*1e3,.02*1e3),ylim=(200,6000),xlabel='Time (ms)',ylabel='T (K)',title=f'IC {i}, P={source["pressures"][i]/101325:.2f} atm')
            ax.grid(alpha=.2)
        axes.flat[-1].legend(fontsize=6)
        fig.suptitle(f'{label}: raw NN, physics OFF, requested final time 20 ms; text = actual end time')
        fig.tight_layout();fig.savefig(figures/f'{label}_temperature20ms.png',dpi=150);plt.close(fig)
    with np.load(cfg['training_pairs']) as d:new_x,new_y,k,kind=d['x'],d['y'],d['scale_k'],d['kind']
    with np.load('results/long10ms_data/training_pairs.npz') as d:old_x,old_y=d['x'],d['y']
    fig,axes=plt.subplots(3,3,figsize=(14,11))
    indices=[0,1,2,3,4,5,6,7,8]
    labels=['T','H2','H','O','O2','OH','H2O','HO2','H2O2']
    for j,label,ax in zip(indices,labels,axes.flat):
        ax.scatter(old_x[:,j],(old_y[:,j]-old_x[:,j])/dt,c='0.7',s=2,alpha=.3,label='old manifold')
        sc=ax.scatter(new_x[:,j],(new_y[:,j]-new_x[:,j])/dt,c=new_x[:,0],cmap='viridis',s=2,alpha=.5,label='multi-scale')
        ax.set(xlabel=label+' input',ylabel='fixed-dt change / s',yscale='symlog')
        if j:ax.set_xscale('log')
    axes.flat[0].legend(markerscale=3,fontsize=8)
    fig.suptitle('Training data coverage: old manifold vs independent multi-scale (no perturbation)')
    fig.tight_layout();fig.savefig(figures/'training_phase_coverage.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].hist(k[kind=='multiscale'],bins=np.arange(.5,26.5),rwidth=.8)
    axes[0].set(xlabel='log-scale exponent k',ylabel='training pairs',title='300 independent samples per scale')
    q=pd.read_csv(generation/'samples.csv').initial_hrr_W_m3.to_numpy()
    axes[1].hist(np.sign(q)*np.log10(1+np.abs(q)),bins=60)
    axes[1].set(xlabel='sign(Q) log10(1+|Q [W/m3]|)',ylabel='pairs',title='HRR is recorded, not used as a rejection threshold')
    fig.tight_layout();fig.savefig(figures/'sampling_and_HRR.png',dpi=150);plt.close(fig)
    selections=pd.DataFrame(frozen)[['regime','gamma','selected_seed','validation_score','usable_validation_model']]
    text=['# Zhang multi-scale sampling + full-state NN, physics OFF', '',
        '첨부된 Zhang et al., Combustion and Flame 245 (2022) 112319의 §2.3과 §2.5를 직접 읽고 구현한 '
        '**축소 비교 실험**이다. 기존 데이터 perturbation을 중단하고 독립적인 multi-scale log sampling으로 바꿨다. '
        'NN은 T/P/BCT(Y)의 변화량을 출력하고 온도·압력까지 raw 출력으로 self-feed한다. '
        '원소 보정, 엔탈피 복원, decoder clipping, 질량합 재정규화는 모두 없다.', '',
        '## 최종 해석 시간과 핵심 결과', '',
        '목표 해석 시간은 **10 ms와 20 ms**, dt=2 μs이다. 20 ms까지 한 번 전진하고 10 ms 시점도 평가했다. '
        '도중 중단되면 실제 도달 시간을 표시하며, 짧은 구간만 맞춘 결과를 10/20-ms 성공으로 세지 않는다. '
        '정상 숫자로 끝난 완주와 물리적으로 허용되는 완주를 구분한다. '
        '**Test의 물리 허용 완주는 모든 방법에서 0/18**이다. Multi-scale sampling이 수치 완주율을 높여도 '
        '질량합·원소 보존을 만족하는 장시간 kinetics 예측으로 이어지지는 않았다.', '']
    for regime,gamma in [('multiscale',.1),('multiscale',.2),('multiscale',.5),('manifold_control',.1)]:
        group=summary[(summary.regime==regime)&np.isclose(summary.gamma,gamma)&(summary.split=='test')&np.isclose(summary.horizon,.02)]
        for r in group.itertuples():
            text.append(f'- {regime}, γ={gamma:g}, {r.method}: 20-ms 수치 완주 {int(r.completed)}/18, '
                f'물리 허용 완주 {int(r.physical_completed)}/18, 실제 도달시간 중앙값 {r.median_end_ms:.5g} ms.')
    text += ['', '## 원문과 구현의 대응', '',
        table(pd.DataFrame([
            ['sampling','§2.5: T 800–3100 K, P .5–2 atm, 종별 독립 log sampling','같은 규칙; 주 비교는 기존 test를 덮도록 P .5–5 atm로 확장'],
            ['log scales','N2 .2–.9, H2/O2 1e−5–1, 나머지 하한 1e−k (k=1…25)','원문 주요 종 분류 유지; H2O도 small-scale 그룹'],
            ['normalization','§2.5는 simplex 정규화의 구체적 순서 미명시','sampled N2 보존, 나머지 raw log weights를 1−N2로 비례 정규화한 명시적 가정'],
            ['main samples','180,000 × 25 = 4,500,000','300 × 25 = 7,500'],
            ['hot supplement','T 1800–3100 K; 다른 종 1e−4–1; 400,000 초기 상태의 연속 두 step','667 초기 상태, 연속 두 step 1,334쌍'],
            ['total pairs','5,300,000','8,834: 기존 8,815쌍과 거의 같은 크기'],
            ['input/output','§2.3: x=[T,P,F(Y)], u=x_next−x; m0=mL=n+2','9종의 11입력·11출력; linear P(atm), literal BCT'],
            ['target distinction','F(Ynext)−F(Ynow)','F(Ynext−Ynow) 또는 signed physical increment가 아님'],
            ['normalization/loss','입력·출력 각각 Z-score, L1','train input/target만으로 별도 적합; standardized increment L1'],
            ['temperature inference','본문은 species 예측 뒤 엔탈피 기반 T 복원도 설명','사용자 요청으로 제외: T와 P를 포함한 모든 NN 출력을 그대로 사용'],
            ['architecture','원문은 GELU, 큰 DNN','이전 요청의 64–64–64 tanh 유지, 9,803 parameters'],
            ['chemistry','8종/16 reversible reactions로 기술','기존 9종/28반응 유지'],
            ['step size','논문에서 사용한 kinetics step과 현재 benchmark 차이','이전 비교와 동일하게 dt=2 μs 유지'],
            ['HRR screening','§2.5에 HRR 비교식/제외 임계값 미명시','임의 HRR 필터를 추가하지 않음; HRR은 기록/진단만 수행']
        ],columns=['항목','논문','이번 구현'])), '',
        '따라서 데이터 규모·반응기구·압력 범위·정규화 가정·작은 tanh 구조·physics-off 차이가 있는 '
        '방법론 비교이며, 논문의 전체 수치 결과를 재현했다는 주장이 아니다. '
        'log 범위는 정규화 전 raw weights에 적용되므로 최종 비질소 질량분율은 비례 재분배로 달라질 수 있다. '
        'N2 범위는 그대로 유지한다.', '',
        '## 압력 범위와 데이터 생성 검증', '',
        '논문 그대로 .5–2 atm인 데이터도 `results/zhang_multiscale_data`에 별도로 생성·보존했다. '
        '기존 test 18개 중 16개가 2 atm보다 높으므로 주 실험은 .5–5 atm에서 새로 생성한 '
        '`results/zhang_multiscale_matched_data`를 사용했다. 좁은 범위 데이터의 NN 결과라고 혼동하면 안 된다.', '',
        f'독립 초기 상태 {data_meta["independent_initial_states"]}개, 학습 쌍 {data_meta["pairs"]}개, '
        f'제외 초기 상태 {data_meta["rejected_initial_states"]}개. '
        f'정상 Cantera 해이면서 초기 HRR<0인 쌍 {data_meta["negative_initial_hrr_pairs"]}개는 유지했다. '
        f'라벨 원소 잔차 최대 {data_meta["max_element_residual"]:.3g}. '
        '유한성, Cantera 음수 roundoff 한계, 기존 모델 온도 지원 범위 200–6000 K만 라벨 적합성에 사용했다. '
        '이번에는 이 조건으로 실제 제외된 표본이 없다.', '',
        '독립 sampling 상태에는 기존 manifold의 시간 좌표가 없다. 각 라벨의 예측 간격은 2 μs이고, '
        '고온 보충은 연속된 두 2-μs 쌍이다. 이것이 곧 10/20-ms 후기 평형 분포를 충분히 포함했다는 뜻은 아니므로 '
        '최종 시간까지의 self-fed 평가와 후반 오차를 별도로 수행한다.', '',
        '## 학습과 고정된 단일 모델 선택', '',
        'γ=.1/.2/.5 multi-scale 각각 10개, 그리고 γ=.1의 기존 manifold 데이터 raw-I/O 대조군 10개를 새로 훈련했다. '
        '총 40개, 모델당 10,000 Adam update, batch512, lr .001→.0003→.0001, 멤버별 norm clip5. '
        '기존 matched-budget 학습 설정을 유지하고 loss만 원문의 L1으로 바꿨다. '
        '같은 raw 입출력·loss·구조·seed·update의 γ=.1 대조군에서 sampling 효과를 분리한다. '
        '기존 physics-on 결과와의 차이를 sampling 효과만으로 해석하지 않는다.', '',
        '각 그룹의 10개를 전체 validation 90,000 one-step 쌍에서 기존 물리 단위 selection_score로 정렬했다. '
        '동률은 seed 순서다. 모든 선택을 저장한 뒤 test/rollout 평가를 시작했다. '
        '모든 후보가 무효라 score가 정의되지 않으면 infinity로 기록하고 최저 seed는 실패 진단용으로만 유지한다. '
        '이 경우 usable model이라고 부르지 않는다.', '',table(selections), '',
        'Mean10/medoid10은 유한한 raw 후보만 사용한다. 온도/질량/원소를 위반한 유한 후보도 제외하거나 고치지 않는다. '
        'Mean10은 물리 상태 공간 평균, medoid10은 기존 train-scaled T/Y 거리로 실제 후보 하나를 선택하며 그 후보의 P도 함께 사용한다. '
        '최소 physics-correction 선택은 correction 자체가 없어 이번에는 정의하지 않는다.', '',
        '## 10 ms / 20 ms a posteriori 전체 결과', '',
        '완주 오류는 전체 horizon을 끝낸 궤적에서만 정의한다. 전부 중단되면 RMSE는 — 이며 0이 아니다. '
        '점화 분류/지연 수치는 raw 온도 crossing 진단이다. 물리 허용 완주가 0이면 정상 kinetics 예측으로 해석할 수 없다. '
        '지연은 비교 일관성을 위해 기존 T0+400 K crossing을 유지했다. Zhang의 최대 HRR 시점 정의와 구분한다. '
        '정밀 reference는 0.2 μs로 계산했고 NN step은 2 μs로 유지했다.', '']
    for split in ('val','test'):
        for horizon in cfg['horizons']:
            f=summary[(summary.split==split)&np.isclose(summary.horizon,horizon)].copy()
            f['coverage_%']=f.mean_coverage*100;f['ignition_%']=f.ignition_classification*100
            text += [f'### {split}, {horizon*1000:g} ms', '',table(f[['regime','gamma','method','completed','physical_completed','coverage_%','median_end_ms',
                'ignition_%','ignition_pairs','temperature_profile_mean','species_profile_mean']]),'']
    text += ['## 구간별 평가: 초기와 후기', '',
        '0–2 ms, 2–10 ms, 10–20 ms 각각을 끝까지 도달한 경우에만 해당 구간 RMSE를 계산한다. '
        '한 구간도 도달하지 못한 모델은 후기 오차가 작다고 평가하지 않는다.', '']
    parts=detail[(detail.split=='test')&np.isclose(detail.horizon,.02)].groupby(['regime','gamma','method']).agg(
        early_completed=('early_0_2ms_complete','sum'),early_T_RMSE=('early_0_2ms_T_RMSE','mean'),
        late_completed=('late_2_10ms_complete','sum'),late_T_RMSE=('late_2_10ms_T_RMSE','mean'),
        tail_completed=('tail_10_20ms_complete','sum'),tail_T_RMSE=('tail_10_20ms_T_RMSE','mean')).reset_index()
    text += [table(parts),'','## A priori test: 전체 90,000쌍', '',
        table(one[one.split=='test'][['regime','gamma','method','samples','selection_score','rmse','temperature_rmse','invalid_row_fraction','mass_sum_mae']]), '',
        '## 실패와 물리 진단', '',
        'BCT inverse의 수학적 정의역 밖 출력은 NaN으로 기록하고 중단한다. signed extension이나 guard clipping을 넣지 않았다. '
        '숫자가 유한해도 |T|>1e6 K, |Y|>1e3, |P|>1e15 Pa이면 수치 폭주로 중단한다. '
        '200–6000 K 온도 범위, Y∈[−1e−12,1+1e−12], mass/element drift≤.01, pressure drift≤1%는 '
        '추론을 바꾸지 않는 물리 허용성 진단에만 사용한다.', '']
    failures=detail[(detail.split=='test')&np.isclose(detail.horizon,.02)]
    counts=failures.groupby(['regime','gamma','method','status']).size().rename('trajectories').reset_index()
    text += [table(counts),'',
        table(failures.groupby(['regime','gamma','method']).agg(minimum_Y=('minimum_mass_fraction','min'),maximum_Y=('maximum_mass_fraction','max'),
            mass_drift=('mass_sum_max','max'),element_drift=('elemental_max','max'),pressure_drift=('pressure_relative_drift_max','max')).reset_index()), '',
        f'새 학습·선택·10/20-ms 평가 실행시간: **{manifest["execution_seconds"]:.1f}초**. '
        '기존 dataset/weights/results는 변경하지 않았다. 각 모델/분할의 직렬화 재로딩과 첫 self-fed step 일치를 확인했다. '
        '별도 검증 기록과 pytest 결과는 결과 디렉터리의 `verification.json` / `pytest.log`에 저장한다.', '',
        '## 그림과 재현', '',
        f'![Training phase coverage]({figures}/training_phase_coverage.png)', '',
        f'![Sampling and HRR]({figures}/sampling_and_HRR.png)', '']
    for d in frozen:
        label=d['label'];text += [f'### {label}: 18 test profiles, 20 ms', '',
            f'![{label} temperature profiles]({figures}/{label}_temperature20ms.png)','']
    text += ['그림의 시간축은 로그이며 t=0만 표시에서 제외한다. 온도 표시 범위는 200–6000 K이고 raw 수치는 바꾸지 않았다. '
        '곡선 끝의 ×와 패널 안의 시간은 실제 도달한 마지막 유한 상태다. 첫 step에서 실패하면 NN 곡선이 없고 0 ms로 표시한다.', '',
        f'- [IC별 원자료]({root}/trajectories.csv), [coverage 포함 요약]({root}/summary_with_coverage.csv)',
        f'- [입출력·physics-off 프로토콜]({root}/protocol.json), [고정 선택]({root}/all_selections_frozen.json)', '',
        '```bash','python -m kinetics.multiscale_data --config configs/zhang_multiscale_matched_data.json',
        'python -m kinetics.zhang_raw_study --config configs/zhang_raw_study.json',
        'python -m kinetics.zhang_raw_report','```', '',
        '재실행은 새 output 경로로 설정을 복사한다. `--resume`은 config·입력·등록 모델 hash·고정 singleton 선택을 확인한 뒤 진행한다. '
        '완료된 실험을 더 실행하려면 새로운 output을 사용한다. '
        '첨부 PDF는 분석에만 사용했고 저작권 있는 원문 파일을 저장소에 게시하지 않았다.']
    args.report.write_text('\n'.join(text)+'\n')
    write=dict(renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),simulation_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest())
    (root/'report_manifest.json').write_text(json.dumps(write,indent=2))
    print('WROTE',args.report)


if __name__=='__main__':main()
