"""Ordinary ANN reports: same IC, Cantera/short/long lines, raw versus repaired."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def compare_phases(root, short):
    rows=[]
    for split in ('val','test'):
        a=pd.read_csv(short/f'evaluations/{split}/NRMSE.csv')
        b=pd.read_csv(root/f'evaluations/{split}/NRMSE.csv')
        keys=['method','predictor','split','mode','trajectory','variable']
        joined=a.merge(b,on=keys,suffixes=('_short','_long'),validate='one_to_one')
        joined['common_complete']=joined.complete_short&joined.complete_long
        joined.to_csv(root/f'report/{split}_short_long_NRMSE.csv',index=False)
        for (method,predictor,mode,variable),g in joined.groupby(['method','predictor','mode','variable']):
            common=g[g.common_complete]
            rows.append(dict(split=split,method=method,predictor=predictor,mode=mode,variable=variable,
                common_completed=len(common),common_ICs=' '.join(map(str,sorted(common.trajectory))),
                short_complete=int(g.complete_short.sum()),long_complete=int(g.complete_long.sum()),
                short_median=float(common.nrmse_short.median()) if len(common) else np.nan,
                long_median=float(common.nrmse_long.median()) if len(common) else np.nan))
    pd.DataFrame(rows).to_csv(root/'report/short_long_common_summary.csv',index=False)


def produce_report(root, source, short=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from .ann_multistep_study import METHODS,sha,write_json

    root=Path(root);short=Path(short) if short is not None else None
    report=root/'report';report.mkdir()
    with np.load(source) as d:
        ids=np.flatnonzero(d['split']=='test');truth=d['states'][ids]
        variables=['T']+d['species'].tolist();dt=float(d['dt']);initials=d['initials'][ids]
    scores=pd.read_csv(root/'evaluations/test/NRMSE.csv')
    roll=pd.read_csv(root/'evaluations/test/rollouts.csv')
    summary=pd.read_csv(root/'evaluations/test/summary.csv')
    summary.to_csv(report/'test_summary.csv',index=False)
    mean=scores[scores.predictor.str.endswith('mean3')]
    mean.pivot(index=['mode','trajectory','variable'],columns='method',values='nrmse').to_csv(report/'NRMSE_by_IC.csv')
    common=[]
    for mode in ('raw','repaired'):
        sets=[set(roll.loc[(roll.predictor==m+'_mean3')&(roll['mode']==mode)&roll.complete,'trajectory']) for m in METHODS]
        shared=set.intersection(*sets)
        for method in METHODS:
            sub=mean[(mean.method==method)&(mean['mode']==mode)&mean.trajectory.isin(shared)]
            for variable in variables:
                values=sub.loc[sub.variable==variable,'nrmse']
                common.append(dict(mode=mode,method=method,variable=variable,common_completed=len(shared),
                    common_ICs=' '.join(map(str,sorted(shared))),median_nrmse=values.median() if len(values) else np.nan))
    pd.DataFrame(common).to_csv(report/'common_completed_NRMSE.csv',index=False)
    if short is not None:
        compare_phases(root,short)

    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'lines.linewidth':1.2})
    labels={'single_step':'ANN single-step','multistep':'ANN self-fed multi-step'}
    for mode in ('raw','repaired'):
        saved={};previous={}
        for method in METHODS:
            with np.load(root/f'evaluations/test/{method}_mean3_{mode}/traces.npz') as d:
                saved[method]={int(k[3:]):d[k] for k in d.files}
            if short is not None:
                with np.load(short/f'evaluations/test/{method}_mean3_{mode}/traces.npz') as d:
                    previous[method]={int(k[3:]):d[k] for k in d.files}
        for zoom in (False,True):
            stem=f'profiles_test_{mode}'+('_ignition_zoom' if zoom else '')
            with PdfPages(report/f'{stem}.pdf') as pdf:
                for ic,reference,initial in zip(ids,truth,initials):
                    center=int(np.argmax(np.diff(reference[:,0])))
                    left=max(0,center-50) if zoom and np.ptp(reference[:,0])>=50 else 0
                    right=min(len(reference)-1,left+100) if zoom else len(reference)-1
                    fig,axes=plt.subplots(2,10,figsize=(29,6.5),sharex=True)
                    for row,method in enumerate(METHODS):
                        current=saved[method][int(ic)]
                        curves=[('Cantera',reference,'black','-')]
                        if short is not None:
                            curves.append(('ANN short mean3',previous[method][int(ic)],'#0072B2','--'))
                        curves.append(('ANN long mean3' if short is not None else 'ANN short mean3',current,
                                       '#D55E00' if short is not None else '#0072B2','--'))
                        case=roll[(roll.predictor==method+'_mean3')&(roll['mode']==mode)&(roll.trajectory==ic)].iloc[0]
                        for col,variable in enumerate(variables):
                            ax=axes[row,col];visible=[]
                            for label,trace,color,style in curves:
                                times=np.arange(len(trace))*dt*1e3
                                ax.plot(times,trace[:,col],label=label,color=color,ls=style)
                                if len(trace)>left:
                                    visible.extend(trace[left:min(right+1,len(trace)),col].tolist())
                            ax.set_xlim(left*dt*1e3,right*dt*1e3)
                            values=np.array(visible);values=values[np.isfinite(values)]
                            if col:
                                ax.set_yscale('symlog',linthresh=1e-12)
                                if len(values):ax.set_ylim(0,max(float(values.max())*1.4,1e-13))
                            elif len(values):
                                span=max(float(np.ptp(values)),.1)
                                ax.set_ylim(values.min()-.05*span,values.max()+.1*span)
                            metric=scores[(scores.predictor==method+'_mean3')&(scores['mode']==mode)&
                                          (scores.trajectory==ic)&(scores.variable==variable)].iloc[0].nrmse
                            text=f'NRMSE {metric:.3g}' if case.complete else f'failed at {(len(current)-1)*dt*1e3:.4g} ms'
                            ax.text(.03,.97,text,transform=ax.transAxes,va='top',fontsize=7,
                                    bbox=dict(facecolor='white',alpha=.75,edgecolor='none',pad=1))
                            if row==0:ax.set_title(variable+(' (K)' if col==0 else ' (mass fraction)'))
                            if col==0:ax.set_ylabel(labels[method])
                            if row==1:ax.set_xlabel('Time (ms)')
                    axes[0,0].legend(loc='lower right',fontsize=6)
                    fig.suptitle(f'Test IC {ic} | T0={initial[0]:.1f} K, P={initial[1]/101325:.3f} atm, phi={initial[2]:.3f} | {mode.upper()}\n'
                        'Ordinary 64/64/64 ANN; 2 us self-fed steps; full 10 ms NRMSE = RMSE/RMS(Cantera); species axes symlog',fontsize=11)
                    fig.tight_layout(rect=(0,0,1,.93));pdf.savefig(fig)
                    if int(ic)==39:
                        suffix='_zoom.png' if zoom else '_full.png'
                        fig.savefig(report/f'diagnostic_IC39_{mode}{suffix}',dpi=130)
                    plt.close(fig)
    ensemble=summary[summary.predictor.str.endswith('mean3')]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    x=np.arange(len(METHODS))
    for mode,offset,color in [('raw',-.18,'#0072B2'),('repaired',.18,'#E69F00')]:
        g=ensemble[ensemble['mode']==mode].set_index('method').reindex(METHODS)
        axes[0].bar(x+offset,g.complete,width=.34,label=mode,color=color)
        axes[1].bar(x+offset,g.joint9_le_0p1,width=.34,label=mode,color=color)
    for ax in axes:
        ax.set_xticks(x,[labels[m] for m in METHODS]);ax.set_ylim(0,18.5);ax.legend();ax.set_ylabel('Test ICs (of 18)')
    axes[0].set_title('Completed 10 ms');axes[1].set_title('Complete AND all reactive outputs NRMSE <= 0.1')
    fig.tight_layout();fig.savefig(report/'coverage.png',dpi=180);fig.savefig(report/'coverage.pdf');plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11,7))
    members=json.loads((root/'final_members.json').read_text())
    for row,method in enumerate(METHODS):
        for member in members:
            if member['method']!=method:continue
            tag=f'{method}_seed{member["seed"]}'
            current=pd.read_csv(root/f'models/{tag}/history.csv')
            offset=0.
            if short is not None:
                old=pd.read_csv(short/f'models/{tag}/history.csv')
                offset=json.loads((short/f'models/{tag}/training.json').read_text())['active_seconds']
                current=pd.concat([old,current.assign(active_seconds=current.active_seconds+offset)],ignore_index=True)
            axes[row,0].plot(current.active_seconds,current.loss,label=f'seed {member["seed"]}')
            axes[row,1].plot(current.active_seconds,current.learning_rate,label=f'seed {member["seed"]}')
        axes[row,0].set_title(labels[method]+' | sampled loss (K changes in short phase)')
        axes[row,1].set_title(labels[method]+' | learning rate')
        for ax in axes[row]:ax.set_yscale('log');ax.set_xlabel('Cumulative active training seconds');ax.legend()
    fig.tight_layout();fig.savefig(report/'loss_and_lr.pdf');fig.savefig(report/'loss_and_lr.png',dpi=160);plt.close(fig)

    historical_family='temporal_pilot_long' if short is not None else 'temporal_pilot'
    historical=Path('results')/historical_family/'run_001/report/test_summary.csv'
    if historical.is_file():
        old=pd.read_csv(historical);old=old[old.method.isin(METHODS)&old.predictor.str.endswith('mean3')].copy()
        old['architecture']='P-only DeepONet';new=ensemble.copy();new['architecture']='ordinary ANN'
        pd.concat([old,new],ignore_index=True).to_csv(report/'historical_deeponet_comparison.csv',index=False)
        old_scores=pd.read_csv(historical.parent.parent/'evaluations/test/NRMSE.csv')
        old_scores=old_scores[old_scores.method.isin(METHODS)&old_scores.predictor.str.endswith('mean3')]
        keys=['method','mode','trajectory','variable']
        matched=old_scores.merge(mean,on=keys,suffixes=('_deeponet','_ann'),validate='one_to_one')
        matched['common_complete']=matched.complete_deeponet&matched.complete_ann
        matched.to_csv(report/'historical_deeponet_matched_IC_NRMSE.csv',index=False)
        comparison=[]
        for (method,mode,variable),g in matched.groupby(['method','mode','variable']):
            shared=g[g.common_complete]
            comparison.append(dict(method=method,mode=mode,variable=variable,common_completed=len(shared),
                deeponet_median=shared.nrmse_deeponet.median() if len(shared) else np.nan,
                ann_median=shared.nrmse_ann.median() if len(shared) else np.nan))
        pd.DataFrame(comparison).to_csv(report/'historical_deeponet_common_summary.csv',index=False)
    config=json.loads((root/'config.json').read_text())
    lines=['# 일반 ANN 단일 스텝 / 다중 스텝 비교','',
        '모델은 입력11 → tanh64 → tanh64 → tanh64 → 출력10의 일반 MLP(9,738 parameters)이다. branch/trunk 또는 시간 입력은 없다.',
        '기존 8,815개 학습 시작점, train/val/test=54/18/18 IC, gamma=0.2, 학습 자료에만 맞춘 기존 정규화를 사용했다. 원본 데이터를 바꾸지 않았다.',
        '단기: 각 방식 seed 11/22/33을 처음부터 각각 활성 학습150초. 단일 스텝은 K1, 다중 스텝은 시간의 각1/4을 K1/4/16/64에 배정했다.',
        '장기: 단기 결과 게시를 원격 bw에서 확인한 후 정확한 가중치와 Adam 상태를 이어 모델당 최대150초 추가한다. 다중 스텝은 K64 유지. LR 하한1e-6에서 고정 training probe가 6회 더 개선되지 않을 때만 평탄 종료이며, 나머지는 시간상한 종료다.',
        'soft constraint 및 학습 중 물리 보정은 없다. loss는 변환·정규화 상태 오차의 평균이고, 다중 스텝은 자기 예측을 되먹이며 모든 중간 시점에 역전파한다.',
        '모든 검증/테스트 예측은2us씩5,000번 자기 예측을 되먹여10ms까지 진행한다. seed별 결과와 물리 상태를 평균한 mean3를 구분한다.',
        'raw에는 clipping/원소투영/엔탈피 온도보정/유효 후보 필터링이 없다. repaired는 이 후처리 시스템을 사용하며 soft constraint와 구분한다.',
        'NRMSE=RMSE(Cantera,NN)/RMS(Cantera), 무차원 비율. 불완주에는 전체 구간 점수를 주지 않는다. joint9는 T+8반응종 모두NRMSE<=0.1이고 N2는 별도 표에 포함한다.',
        '완주는 유한값, 비음수 조성, 200~6000K 조건이며 정확도나 보존법칙 만족을 뜻하지 않는다. 초기0.1ms 점수도 원본 평가 표에 포함했다.',
        '방법 선택은 검증 raw 완주수, joint9 통과수, worst9 중앙값 순으로 테스트를 읽기 전에 동결한다. gamma 최적화는 수행하지 않았다.',
        'PDF는18개 테스트 각각2x10칸이고 장기 PDF에는 Cantera/단기/장기 3개 선을 함께 그렸다. 개별 그림 축 범위는 표시된 시간 창 안의 값으로 잡았다.',
        '', '| Method | Mode | Completed | joint9<=0.1 | Median worst9 among completed |','|---|---|---:|---:|---:|']
    for row in ensemble.itertuples():lines.append(f'| {row.method} | {row.mode} | {row.complete}/18 | {row.joint9_le_0p1}/18 | {row.median_worst9:.6g} |')
    (report/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    outputs={p.name:sha(p) for p in report.iterdir() if p.is_file()}
    write_json(report/'complete.json',dict(status='passed',test_ICs=18,axes_per_profile=20,
        profile_PDFs=4,includes_short_and_long=short is not None,output_sha256=outputs))


def verify(root, source, pairs):
    from .ann_multistep_study import SOURCE_SHA,PAIR_SHA,METHODS,sha,tf
    root=Path(root);errors=[];checks=0
    with np.load(source) as d:reference=d['states'];variables=['T']+d['species'].tolist()
    for split in ('val','test'):
        frame=pd.read_csv(root/f'evaluations/{split}/NRMSE.csv')
        roll=pd.read_csv(root/f'evaluations/{split}/rollouts.csv')
        for predictor in frame.predictor.unique():
            for mode in ('raw','repaired'):
                with np.load(root/f'evaluations/{split}/{predictor}_{mode}/traces.npz') as traces:
                    for key in traces.files:
                        ic=int(key[3:]);trace=traces[key];full=len(trace)==len(reference[ic])
                        scores=frame[(frame.predictor==predictor)&(frame['mode']==mode)&(frame.trajectory==ic)].set_index('variable').loc[variables]
                        row=roll[(roll.predictor==predictor)&(roll['mode']==mode)&(roll.trajectory==ic)].iloc[0]
                        if bool(row.complete)!=full or int(row.completed_steps)!=len(trace)-1:errors.append('completion mismatch')
                        for j,variable in enumerate(variables):
                            expected=np.linalg.norm(reference[ic,:,j]-trace[:,j])/np.linalg.norm(reference[ic,:,j]) if full else np.nan
                            if not np.isclose(scores.loc[variable,'nrmse'],expected,rtol=2e-10,atol=1e-12,equal_nan=True):errors.append('NRMSE mismatch')
                            checks+=1
    members=json.loads((root/'final_members.json').read_text())
    if len(members)!=6 or set(m['method'] for m in members)!=set(METHODS):errors.append('member coverage')
    for member in members:
        folder=root/'models'/f'{member["method"]}_seed{member["seed"]}'
        if sha(folder/'model.keras')!=member['model_sha256']:errors.append('model hash')
        model=tf.keras.models.load_model(folder/'model.keras',compile=True)
        if model.count_params()!=9738 or [l.units for l in model.layers if isinstance(l,tf.keras.layers.Dense)]!=[64,64,64,10]:errors.append('not an ordinary ANN')
        if int(model.optimizer.iterations.numpy())!=member['updates']:errors.append('Adam restore')
        reader=tf.train.load_checkpoint(str(folder/'optimizer_checkpoint'))
        if int(reader.get_tensor('optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE'))!=member['updates']:errors.append('Adam checkpoint')
    metadata=json.loads((root/'report/complete.json').read_text())
    for name,expected in metadata['output_sha256'].items():
        if sha(root/'report'/name)!=expected:errors.append('report artifact hash')
    for mode in ('raw','repaired'):
        for suffix in ('','_ignition_zoom'):
            pdf=root/f'report/profiles_test_{mode}{suffix}.pdf'
            if len(re.findall(rb'/Type\s*/Page\b',pdf.read_bytes()))!=18:errors.append('profile PDF pages')
    if sha(source)!=SOURCE_SHA or sha(pairs)!=PAIR_SHA:errors.append('original data changed')
    if json.loads((root/'selection_frozen.json').read_text())['test_accessed']:errors.append('test selection leakage')
    return dict(status='passed' if not errors else 'failed',nrmse_checks=checks,
        models_verified=len(members),ordinary_ANN_parameters=9738,data_hashes_unchanged=sha(source)==SOURCE_SHA and sha(pairs)==PAIR_SHA,errors=errors)
