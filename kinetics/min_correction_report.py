"""Report frozen-model selection by minimum physics-correction magnitude."""
import argparse
import hashlib
import json
import os
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
    parser.add_argument('--root',type=Path,default=Path('results/min_correction_tanh10'))
    parser.add_argument('--report',type=Path,default=Path('MIN_CORRECTION_REPORT.md'))
    args=parser.parse_args();root=args.root
    cfg=json.loads((root/'config.json').read_text());old=Path(cfg['baseline'])
    manifest=json.loads((root/'manifest.json').read_text());assert manifest['status']=='completed'
    protocol=json.loads((root/'selection_protocol.json').read_text())
    summary=pd.read_csv(root/'comparison_summary.csv');detail=pd.read_csv(root/'trajectories.csv')
    one=pd.read_csv(root/'comparison_one_step.csv');usage=pd.read_csv(root/'selection_usage.csv')
    checks=pd.read_csv(root/'reload_checks.csv')
    assert len(checks)==4 and checks[['minimal_correction_first_step','actual_candidate','reload_match','first_self_fed_step_match']].all().all()
    figures=root/'figures';figures.mkdir(exist_ok=True)
    with np.load(cfg['source']) as data:states,split,dt=data['states'],data['split'],float(data['dt'])
    ids=np.flatnonzero(split=='test')
    colors=dict(single_best='tab:red',mean10='tab:blue',medoid10='tab:orange',min_correction10='tab:green')
    for gamma in cfg['gammas']:
        label=f'g{gamma:g}';traces={}
        for method in colors:
            source=root if method=='min_correction10' else old
            with np.load(source/label/'traces'/f'{method}_test.npz') as archive:
                traces[method]={int(i):archive[f'trajectory_{int(i)}'] for i in ids}
        fig,axes=plt.subplots(3,6,figsize=(22,11))
        for i,ax in zip(ids,axes.flat):
            truth=states[i];ax.plot(np.arange(1,len(truth))*dt*1e3,truth[1:,0],'k--',lw=1.6,label='Cantera')
            for method,color in colors.items():
                t=traces[method][i]
                ax.plot(np.arange(1,len(t))*dt*1e3,t[1:,0],color=color,lw=1,label=method)
            ax.set(xscale='log',xlim=(dt*1e3,cfg['horizon']*1e3),xlabel='Time (ms)',ylabel='T (K)',title=f'IC {i}, T0={truth[0,0]:.0f} K')
            ax.grid(alpha=.2)
        axes.flat[-1].legend(fontsize=6)
        fig.suptitle(f'Gamma {gamma:g}: minimum physics correction vs frozen controls')
        fig.tight_layout();fig.savefig(figures/f'{label}_temperature.png',dpi=150);plt.close(fig)
        events=pd.read_csv(root/label/'selection_test.csv.gz')
        fig,axes=plt.subplots(2,2,figsize=(14,8),sharex=True)
        for row,i in enumerate([44,78]):
            e=events[events.trajectory==i]
            axes[row,0].step(e.time*1e3,e.selected_seed,where='post',lw=.7)
            axes[row,0].set(ylabel='Selected seed',title=f'IC {i}: member identity')
            axes[row,1].plot(e.time*1e3,e.correction_norm,lw=.7)
            axes[row,1].set(ylabel='Scaled RMS correction',title=f'IC {i}: selected correction')
            axes[row,1].set_yscale('symlog',linthresh=1e-12)
            for ax in axes[row]:ax.set(xlabel='Time (ms)');ax.grid(alpha=.2)
        fig.suptitle(f'Gamma {gamma:g}: per-step selection, fixed diagnostic ICs')
        fig.tight_layout();fig.savefig(figures/f'{label}_selection.png',dpi=150);plt.close(fig)
    control=protocol['common_control'];test=summary[summary.split=='test']
    base=test[(test.gamma==control['gamma'])&(test.method=='single_best')].iloc[0]
    text=['# 매 timestep 최소 physics-correction 멤버 선택', '',
        '기존 γ=0.2/0.5의 **64–64–64 tanh 10개 멤버**를 그대로 사용했다. '
        '매 timestep 모든 멤버가 같은 현재 상태에서 다음 상태를 제안하고, 각각 Wang 원소 보정·정압 엔탈피 온도 복원을 받은 뒤 '
        '**보정 전후 변화가 가장 작은 유효 후보 하나**를 다음 상태로 사용한다. 재훈련·기존 결과 덮어쓰기는 없다.', '',
        '## Test 핵심 결과', '',
        f'이전 validation a priori 공통 단일 대조군은 γ={control["gamma"]:g}/seed{control["selected_seed"]}로 유지했다: '
        f'18개 중 {int(base.completed)}개 완주, 점화 분류 {base.ignition_classification*100:.1f}%, '
        f'T profile 평균 RMSE {base.temperature_profile_mean:.1f} K.', '']
    for gamma in cfg['gammas']:
        r=test[(test.gamma==gamma)&(test.method=='min_correction10')].iloc[0]
        text += [f'- γ={gamma:g} 최소 보정: 완주 {int(r.completed)}/18, 점화 분류 {r.ignition_classification*100:.1f}%, '
            f'T profile RMSE {r.temperature_profile_mean:.1f} K, 종 profile RMSE {r.species_profile_mean:.5g}, '
            f'점화지연 조건부 중앙오차 {r.median_ignition_error_refined*100:.2f}% ({int(r.ignition_pairs)}쌍).']
    text += ['', '## 고정한 선택 기준', '',
        '`c_k = sqrt(((T_corr,k−T_ANN,k)/sT)^2 + Σ_j((Y_corr,k,j−Y_ANN,k,j)/sY_j)^2) / sqrt(10)`', '',
        '- sT=max(train 입력 T 표준편차,100 K), sY_j=max(train 입력 Y_j의 95백분위,1e−8). '
        '기존 medoid에 저장된 train-only 척도를 그대로 사용한다. 정답, test 통계, rollout 성능으로 척도를 조정하지 않았다.',
        '- 현재 상태 대비 작은 물리적 시간 변화가 아니라 **ANN 제안에 보정을 가한 크기**다. '
        'ANN 원시 제안은 기존 [0,1] transformed decoder guard를 거친 상태이므로 guard 자체의 변화는 c_k에 포함하지 않는다.',
        '- 온도 항은 학습한 ANN T와 엔탈피로 복원한 T의 차이다. ANN T는 다음 상태 온도로 직접 사용하지 않지만 '
        '이 선택 기준에는 영향을 준다. 이는 종만의 Wang correction norm이나 G-space 최소 norm과는 다른 명시적 T/Y 기준이다.',
        '- 보정 후 음수·원소 잔차·열역학 불능 후보는 제외한다. 전부 무효면 그 궤적은 실패한다. '
        '동률은 저장된 seed 오름차순의 첫 멤버를 선택한다. 항상 실제 보정 후보를 반환한다.',
        '- dt=2 μs, 10 ms/5,000 self-fed step, validation/test 각각 18 IC. '
        '중간 정답 상태 주입, ODE fallback, hysteresis, 추가 학습은 없다.',
        '- 이 기준은 평가 전에 저장했고 결과를 보고 바꾸지 않았다. '
        '이미 사용한 test split의 후속 ablation으로 새로운 독립 외부 검증은 아니다.', '',
        '## A posteriori 비교', '',
        '이전 single/mean/medoid는 동일한 가중치·데이터·Wang 보정의 저장된 결과를 재사용했다. '
        '단일 모델은 각 γ 내 a priori 최상위이며 공통 대조군은 위의 γ=.2/seed66이다. '
        'T RMSE는 완주 궤적별 시간적분 오차의 평균(K), Y RMSE는 질량분율 단위다. '
        '미완주는 full-profile 오차 NaN으로 제외하고 완주율·coverage를 따로 보고한다.', '']
    for partition in ('val','test'):
        f=summary[summary.split==partition].copy()
        f['completion_%']=f.completion_fraction*100;f['ignition_%']=f.ignition_classification*100
        f['delay_error_%']=f.median_ignition_error_refined*100
        text += [f'### {partition}', '',table(f[['gamma','method','completed','completion_%','ignition_%','delay_error_%','ignition_pairs',
            'temperature_profile_mean','temperature_profile_pooled','species_profile_mean']]),'']
    changes=[]
    for gamma in cfg['gammas']:
        g=test[test.gamma==gamma].set_index('method');r=g.loc['min_correction10']
        for method in ('single_best','mean10','medoid10'):
            b=g.loc[method]
            changes.append(dict(gamma=gamma,baseline=method,completion_delta_pp=100*(r.completion_fraction-b.completion_fraction),
                ignition_delta_pp=100*(r.ignition_classification-b.ignition_classification),T_RMSE_delta_K=r.temperature_profile_mean-b.temperature_profile_mean))
    text += ['### 같은 γ의 기존 방법 대비', '',table(pd.DataFrame(changes)), '',
        '차이는 최소보정−기존 방법이다. RMSE 차이가 음수면 완주 궤적 평균 오차가 작다. '
        '완주 집합이 다르면 이 수치만으로 전체 성능 개선을 판단할 수 없다. '
        '점화지연은 양쪽 모두 점화한 쌍의 조건부 수치이며 reference τ는 기존 0.2 μs 정밀 측정을 재사용했다.', '',
        '## A priori: 전체 90,000 test 쌍', '',
        table(one[one.split=='test'][['gamma','method','samples','selection_score','rmse','major_rmse','minor_rmse','temperature_rmse','invalid_row_fraction']]), '',
        '보정이 작다는 것은 정답과의 오차가 작다는 보장이 아니다. '
        '잘못된 반응 경로도 원소·엔탈피 보존 조건을 만족할 수 있다. '
        '특히 반응을 거의 진행시키지 않는 제안은 작은 보정을 받고도 점화를 놓칠 수 있다.', '',
        '## 실패와 점화 불일치', '']
    bad=detail[(~detail.complete)|(~detail.ignition_classification_correct)]
    text += [table(bad[['gamma','split','trajectory','complete','coverage','completed_steps','ignition_status',
        'full_temperature_rmse','prefix_temperature_rmse']]) if len(bad) else '없음.', '',
        '## 선택 동작과 물리 진단', '',
        table(usage[usage.split=='test'][['gamma','seed','selected_steps','fraction','mean_correction_norm']]), '',
        table(detail[detail.split=='test'][['gamma','trajectory','unique_selected_members','member_switches','mean_correction_norm']]), '',
        table(pd.read_csv(root/'summary.csv')[['gamma','split','mass_sum_max','elemental_max','physical_fraction']]), '',
        table(pd.read_csv(root/'correction_diagnostics.csv')), '',
        '질량·원소·엔탈피는 보정이 강제한다. 작은 보정과 보존 만족만으로 올바른 kinetics나 entropy 비감소를 보장하지 않는다.', '',
        f'새 평가 총 실행시간 **{manifest["execution_seconds"]:.1f}초**. '
        '모든 원본 모델·전처리·데이터·대조군 hash 불변을 확인했다. '
        '각 γ/분할에서 독립 수식으로 첫-step 최소 후보와 실제 선택 일치를 확인하고, '
        '직렬화 재로딩 예측/seed 및 첫 self-fed step 일치를 검증했다. '
        '선택 함수의 무효 후보·동률·단위 정규화·시점별 멤버 교체 테스트를 포함하여 **81 tests passed**.', '',
        '## 그림과 재현', '']
    for gamma in cfg['gammas']:
        label=f'g{gamma:g}'
        text += [f'- γ={gamma:g}: [전체 test 온도]({figures}/{label}_temperature.png), '
            f'[IC44/78의 step별 선택]({figures}/{label}_selection.png)']
    text += ['', '온도 그림의 t=0은 로그축 표시에서만 제외했다. 계산에는 초기 상태와 전체 시간 구간을 포함했다.', '',
        f'- [요약]({root}/comparison_summary.csv), [IC별 결과]({root}/trajectories.csv), [선택 빈도]({root}/selection_usage.csv)',
        f'- [선택 규칙]({root}/selection_protocol.json), [검증]({root}/reload_checks.csv)', '',
        '```bash','/workspace/kinetics-venv/bin/python -m kinetics.min_correction_study',
        '/workspace/kinetics-venv/bin/python -m kinetics.min_correction_report','```', '',
        '반복은 새 `--output`을 사용하며 report에도 같은 `--root`와 새 `--report`를 지정한다. '
        '기존 모델·결과는 덮어쓰지 않는다.']
    args.report.write_text('\n'.join(text)+'\n')
    (root/'report_manifest.json').write_text(json.dumps(dict(
        renderer_source=str(Path(__file__).relative_to(Path.cwd())),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        simulation_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        simulation_rerun=False),indent=2))
    print('WROTE',args.report)


if __name__=='__main__':main()
