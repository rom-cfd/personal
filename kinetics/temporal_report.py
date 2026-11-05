"""Complete-profile reports and independent checks for the temporal pilot."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def produce_report(root, source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from .temporal_pilot import METHODS, write_json

    root = Path(root)
    report = root / 'report'
    report.mkdir()
    with np.load(source) as d:
        ids = np.flatnonzero(d['split']=='test')
        references = d['states'][ids]
        variables = ['T'] + d['species'].tolist()
        dt = float(d['dt'])
        initial_conditions = d['initials'][ids]
    scores = pd.read_csv(root / 'evaluations/test/NRMSE.csv')
    rollout = pd.read_csv(root / 'evaluations/test/rollouts.csv')
    summary = pd.read_csv(root / 'evaluations/test/summary.csv')
    summary.to_csv(report / 'test_summary.csv', index=False)
    score_mean = scores[scores.predictor.str.endswith('mean3')]
    score_mean.pivot(index=['mode','trajectory','variable'], columns='method', values='nrmse').to_csv(report/'NRMSE_by_IC.csv')
    joined = []
    for mode in ('raw','repaired'):
        complete_sets = [set(rollout.loc[(rollout.method==method)&(rollout['mode']==mode)&
            rollout.predictor.str.endswith('mean3')&rollout.complete, 'trajectory']) for method in METHODS]
        common = set.intersection(*complete_sets)
        for method in METHODS:
            subset = score_mean[(score_mean.method==method)&(score_mean['mode']==mode)&score_mean.trajectory.isin(common)]
            for variable in variables:
                values = subset.loc[subset.variable==variable,'nrmse']
                joined.append(dict(mode=mode,method=method,variable=variable,common_completed=len(common),
                                   common_ICs=' '.join(map(str,sorted(common))),
                                   median_nrmse=float(values.median()) if len(values) else np.nan))
    pd.DataFrame(joined).to_csv(report/'common_completed_NRMSE.csv',index=False)
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'lines.linewidth':1.2})
    colors = {'single_step':'#777777','interval':'#0072B2','multistep':'#D55E00'}
    labels = {'single_step':'Single-step baseline','interval':'Interval DeepONet','multistep':'Self-fed multi-step'}
    for mode in ('raw','repaired'):
        traces = {}
        for method in METHODS:
            with np.load(root/f'evaluations/test/{method}_mean3_{mode}/traces.npz') as values:
                traces[method] = {int(k[3:]):values[k] for k in values.files}
        for zoom in (False,True):
            name = f'profiles_test_{mode}' + ('_ignition_zoom' if zoom else '')
            with PdfPages(report/f'{name}.pdf') as pdf:
                for trajectory, reference, ic in zip(ids,references,initial_conditions):
                    fig, axes = plt.subplots(3,10,figsize=(29,8.5),sharex=True)
                    center = int(np.argmax(np.diff(reference[:,0])))
                    if zoom:
                        left = max(0,center-50) if reference[:,0].max()-reference[0,0]>=50 else 0
                        right = min(len(reference)-1,left+100)
                    else:
                        left,right = 0,len(reference)-1
                    for row,method in enumerate(METHODS):
                        trace = traces[method][int(trajectory)]
                        pred_time = np.arange(len(trace))*dt*1e3
                        ref_time = np.arange(len(reference))*dt*1e3
                        case = rollout[(rollout.predictor==f'{method}_mean3')&(rollout['mode']==mode)&
                                       (rollout.trajectory==trajectory)].iloc[0]
                        for col,variable in enumerate(variables):
                            ax = axes[row,col]
                            ax.plot(ref_time,reference[:,col],color='black',label='Cantera',alpha=.85)
                            ax.plot(pred_time,trace[:,col],color=colors[method],ls='--',label='NN mean of 3')
                            if not case.complete:
                                ax.axvline(pred_time[-1],color='#cc0000',ls=':',lw=.8)
                            if col>0:
                                ax.set_yscale('symlog',linthresh=1e-12)
                            ax.set_xlim(ref_time[left],ref_time[right])
                            metric = scores[(scores.predictor==f'{method}_mean3')&(scores['mode']==mode)&
                                (scores.trajectory==trajectory)&(scores.variable==variable)].iloc[0].nrmse
                            text = f'NRMSE {metric:.3g}' if case.complete else f'failed {pred_time[-1]:.4g} ms'
                            ax.text(.03,.96,text,transform=ax.transAxes,va='top',fontsize=7,
                                    bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=1))
                            if row==0:
                                ax.set_title(variable+(' (K)' if col==0 else ' (mass fraction)'))
                            if col==0:
                                ax.set_ylabel(labels[method])
                            if row==2:
                                ax.set_xlabel('Time (ms)')
                    axes[0,0].legend(loc='lower right',fontsize=6)
                    fig.suptitle(f'Test IC {trajectory} | T0={ic[0]:.1f} K, P={ic[1]/101325:.3f} atm, phi={ic[2]:.3f} | {mode.upper()}\n'
                                 '2 us self-fed steps; full 10 ms NRMSE = RMSE / RMS(Cantera); species axes symlog (1e-12)',fontsize=12)
                    fig.tight_layout(rect=(0,0,1,.94))
                    pdf.savefig(fig)
                    # IC39 is an explicitly pre-existing diagnostic, not a newly selected best case.
                    if int(trajectory)==39:
                        suffix = '_zoom.png' if zoom else '_full.png'
                        fig.savefig(report/f'diagnostic_IC39_{mode}{suffix}',dpi=125)
                    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    ensemble=summary[summary.predictor.str.endswith('mean3')]
    x=np.arange(3)
    for mode,offset,color in [('raw',-.18,'#0072B2'),('repaired',.18,'#E69F00')]:
        s=ensemble[ensemble['mode']==mode].set_index('method').reindex(METHODS)
        axes[0].bar(x+offset,s.complete,width=.34,label=mode,color=color)
        axes[1].bar(x+offset,s.joint9_le_0p1,width=.34,label=mode,color=color)
    for ax in axes:
        ax.set_xticks(x,[labels[m] for m in METHODS],fontsize=8)
        ax.set_ylim(0,18.5)
        ax.set_ylabel('Test IC count (of 18)')
        ax.legend()
    axes[0].set_title('Completed 10 ms trajectories')
    axes[1].set_title('Complete AND all T + 8 reactive species NRMSE <= 0.1')
    fig.tight_layout()
    fig.savefig(report/'coverage.png',dpi=180)
    fig.savefig(report/'coverage.pdf')
    plt.close(fig)
    training=pd.read_csv(root/'training.csv')
    lines=['# 시간 입력 / 여러 스텝 DeepONet 예비 실험','',
        '동일한 54/18/18 초기조건 분할, gamma=0.2, 기존 8,815개 시작점을 사용했다. 원본 데이터는 변경하지 않았다.',
        '세 방법을 seed 11/22/33으로 각각 처음부터 학습했다. 각 모델은 활성 학습 계산 150초이며, 기존 10,000회 모델의 연속 학습이 아니다.',
        'soft constraint와 학습 중 물리 보정은 없다. 단일 스텝/다중 스텝 모델은 같은 P-only 구조(9,719 parameters), 시간 입력 모델은 9,783 parameters다.',
        '시간 입력 모델은 현재 T/Y/P를 branch, 앞으로 진행할 시간 간격을 trunk에 넣는다. 2/4/8/16 us 비율은 1/2,1/6,1/6,1/6이다.',
        '다중 스텝은 활성 시간의 각 1/4을 K=1/4/16/64에 배정하고 예측 상태를 되먹여 전체 경로에 역전파한다. 후반 K의 업데이트 수가 적을 수 있다.',
        '평가에서는 모든 방법이 2 us씩 5,000번 자기 예측을 되먹인다. seed별 결과와 물리 상태를 평균한 3-member ensemble을 구분했다.',
        'raw는 clipping/projection/enthalpy correction/invalid filtering이 없다. repaired는 기존 inverse-domain clipping, 원소 보존 투영, 엔탈피 온도 재구성, 유효 후보 선택 후 평균 및 재보정을 사용한다.',
        '완주는 유한/비음수 조성 및 200~6000 K 조건이다. raw에서 완주해도 원소/질량/엔탈피 보존이나 정확도를 보장하지 않는다.',
        'NRMSE = RMSE(Cantera, NN) / RMS(Cantera), 무차원 비율. 불완주에는 전체 10 ms NRMSE를 부여하지 않았다.',
        'worst9와 joint9는 T 및 8개 반응성 화학종을 함께 평가한다. 일정한 N2는 별도 NRMSE 표에 포함했다. 성공 기준 0.1은 예비 진단 기준이다.',
        '방법 선택은 테스트를 읽기 전에 검증 raw 완주 수, joint9 통과 수, worst9 중앙값 순으로 동결했다. gamma를 탐색하거나 전체 화학종 최적 gamma라고 결론 내리지 않았다.',
        '', '## Test ensemble (mean of 3)','',
        '| Method | Mode | Completed | joint9 <= 0.1 | Median worst9 among completed |',
        '|---|---|---:|---:|---:|']
    for row in ensemble.itertuples():
        lines.append(f'| {row.method} | {row.mode} | {row.complete}/18 | {row.joint9_le_0p1}/18 | {row.median_worst9:.6g} |')
    lines += ['', '## Training','', '| Method | Seed | Updates | Active seconds | End |', '|---|---:|---:|---:|---|']
    for row in training.itertuples():
        lines.append(f'| {row.method} | {row.seed} | {row.updates} | {row.active_seconds:.3f} | {row.termination} |')
    lines += ['', '## Files','',
        '- test_summary.csv: seed별 / mean3, raw / repaired 완주 및 joint9 성능',
        '- NRMSE_by_IC.csv: 각 테스트 IC/변수의 mean3 전체 구간 NRMSE',
        '- common_completed_NRMSE.csv: 세 방법이 공통 완주한 IC에서만 비교한 변수별 NRMSE',
        '- profiles_test_{raw,repaired}.pdf: 18개 테스트 각각 3x10 전체 구간 그림',
        '- profiles_test_{raw,repaired}_ignition_zoom.pdf: 같은 IC의 점화 근처 확대',
        '- diagnostic_IC39_*: 과거에 확인했던 IC39 고정 진단 그림. 이번 결과에서 고른 최고 사례가 아니다.',
        '- ../evaluations/{val,test}/one_step_NRMSE.csv: Cantera 현재 상태를 입력한 단일 스텝 진단. finite-only 수치는 불완전 표본 진단으로 전체 정확도와 구분.',
        '', '공통 완주 집합이 없으면 비교값을 비워 두었다. 낮은 온도 NRMSE만으로 전체 반응계가 정확하다고 평가하지 않는다.']
    (report/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    write_json(report/'complete.json',dict(status='passed',pdfs=4,test_ICs=len(ids)))


def verify(root, source, pairs):
    """Recompute stored profile scores independently, and check model/Adam evidence."""
    from .temporal_pilot import SOURCE_SHA,PAIR_SHA,sha,METHODS,tf
    root=Path(root)
    errors=[]
    checks=0
    with np.load(source) as d:
        reference=d['states']
        variables=['T']+d['species'].tolist()
    for split in ('val','test'):
        frame=pd.read_csv(root/f'evaluations/{split}/NRMSE.csv')
        roll=pd.read_csv(root/f'evaluations/{split}/rollouts.csv')
        for predictor in frame.predictor.unique():
            for mode in ('raw','repaired'):
                path=root/f'evaluations/{split}/{predictor}_{mode}/traces.npz'
                with np.load(path) as traces:
                    for name in traces.files:
                        ic=int(name[3:])
                        trace=traces[name]
                        full=len(trace)==len(reference[ic])
                        stored=frame[(frame.predictor==predictor)&(frame['mode']==mode)&(frame.trajectory==ic)].set_index('variable').loc[variables]
                        r=roll[(roll.predictor==predictor)&(roll['mode']==mode)&(roll.trajectory==ic)].iloc[0]
                        if bool(r.complete)!=full or int(r.completed_steps)!=len(trace)-1:
                            errors.append(f'completion {split}/{predictor}/{mode}/{ic}')
                        for j,variable in enumerate(variables):
                            value=stored.loc[variable,'nrmse']
                            if full:
                                truth=reference[ic,:,j]
                                expected=np.linalg.norm(truth-trace[:,j])/np.linalg.norm(truth)
                                if not np.isclose(value,expected,rtol=2e-10,atol=1e-12):
                                    errors.append(f'NRMSE {split}/{predictor}/{mode}/{ic}/{variable}')
                            elif not np.isnan(value):
                                errors.append('incomplete full score')
                            checks+=1
    members=json.loads((root/'final_members.json').read_text())
    for row in members:
        folder=root/'models'/f'{row["method"]}_seed{row["seed"]}'
        if sha(folder/'model.keras')!=row['model_sha256']:
            errors.append('model hash')
        reader=tf.train.load_checkpoint(str(folder/'optimizer_checkpoint'))
        key='optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE'
        if int(reader.get_tensor(key))!=row['updates']:
            errors.append('Adam iteration count')
        restored=tf.keras.models.load_model(folder/'model.keras',compile=True)
        if int(restored.optimizer.iterations.numpy())!=row['updates']:
            errors.append('Keras Adam iterations')
    if len(members)!=9 or set(m['method'] for m in members)!=set(METHODS):
        errors.append('member coverage')
    if sha(source)!=SOURCE_SHA or sha(pairs)!=PAIR_SHA:
        errors.append('original data changed')
    for mode in ('raw','repaired'):
        for zoom in ('','_ignition_zoom'):
            if not (root/f'report/profiles_test_{mode}{zoom}.pdf').is_file():
                errors.append('missing PDF')
    frozen=json.loads((root/'selection_frozen.json').read_text())
    if frozen['test_accessed']:
        errors.append('selection not frozen before test')
    return dict(status='passed' if not errors else 'failed',nrmse_checks=checks,
                models_verified=len(members),data_hashes_unchanged=sha(source)==SOURCE_SHA and sha(pairs)==PAIR_SHA,
                errors=errors)
