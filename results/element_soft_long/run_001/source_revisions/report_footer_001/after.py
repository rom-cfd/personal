"""Cantera / baseline10k / soft10k / soft-long, same7x10 layout and NRMSE."""
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,ScalarFormatter
from kinetics.p_only_training import sha256,write_json
from kinetics.ann_deeponet_study import check_deadline
from kinetics.soft_constraint_evaluation import TEAMS
from kinetics.element_long_study import event
from kinetics.element_long_training import joint_monitor,joint_check
from scripts.plot_kinetics_7x10_comparison import COLUMNS,STYLE,BLUE,ORANGE,BLACK
from scripts.report_kinetics_nrmse import profile_errors,number,VARIABLES
from scripts.report_soft_constraint_comparison import LABELS

GRAY='#64748B'
ROWS=[(LABELS[name].replace(' DeepONet10','\nDeepONet10'),f'base_{name}',f'short_{name}',f'long_{name}') for name,_ in TEAMS]
METHODS=[(f'{stage}_{name}',LABELS[name],label,index) for index,(name,_) in enumerate(TEAMS,1)
         for stage,label in [('base','baseline'),('short','soft10k'),('long','softlong')]]


def limits(reference,traces,i):
    ranges=[]
    for k in range(10):
        values=[reference[:,k]]+[t[i][:,k] for t in traces.values()]
        lo=min(float(a.min()) for a in values);hi=max(float(a.max()) for a in values)
        if k==0:
            pad=max((hi-lo)*.045,20.);ranges.append((lo-pad,hi+pad))
        elif k==9 and hi-lo < .02*max(hi,1e-12):
            # Do not amplify machine-precision changes of inert N2.
            pad=max(.02*hi,1e-6);ranges.append((max(0,lo-pad),hi+pad))
        else:ranges.append((0.,max(hi*1.08,1e-16)))
    return ranges


def profile_figure(data,record,traces,i,mode):
    reference=data['states'][i];dt=record['dt'];horizon=(len(reference)-1)*dt*1000
    ylimits=limits(reference,traces,i)
    fig,axes=plt.subplots(7,10,figsize=(31,20.2),sharex=True,squeeze=False)
    fig.subplots_adjust(left=.087,right=.990,bottom=.062,top=.900,wspace=.36,hspace=.38)
    status=[]
    for row,(label,base,short,long) in enumerate(ROWS):
        models=[(base,GRAY),(short,BLUE),(long,ORANGE)]
        failures=[]
        for name,color in models:
            trace=traces[name][i]
            if len(trace)<len(reference):
                failures.append(('base' if name==base else ('10k' if name==short else 'long'),(len(trace)-1)*dt*1000))
        for k,ax in enumerate(axes[row]):
            for name,color in models:
                trace=traces[name][i];tt=np.arange(len(trace))*dt*1000
                ax.plot(tt,trace[:,k],color=color,lw=1.05,alpha=.95,zorder=2)
                if len(trace)<len(reference):
                    ax.plot(tt[-1],trace[-1,k],marker='x',color=color,ms=3.5,mew=.9,zorder=5)
            ax.plot(np.arange(len(reference))*dt*1000,reference[:,k],color=BLACK,
                    ls=(0,(4,2.5)),lw=1.1,zorder=3)
            ax.set_xlim(0,horizon);ax.set_ylim(*ylimits[k])
            ax.set_xticks([0,2,4,6,8,10]);ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
            formatter=ScalarFormatter(useOffset=False,useMathText=True);formatter.set_powerlimits((-3,3))
            ax.yaxis.set_major_formatter(formatter);ax.yaxis.get_offset_text().set_size(7.5)
            ax.grid(True,which='major',ls=':',lw=.45,color='#b7c1cc',alpha=.65)
            ax.tick_params(pad=2)
            if row==0:ax.set_title(COLUMNS[k],pad=12,fontweight='semibold')
            if row==6:ax.set_xlabel('Time [ms]',labelpad=5)
        position=axes[row,0].get_position()
        fig.text(.009,(position.y0+position.y1)/2,f'{row+1}  {label}',
                 ha='left',va='center',fontsize=11,fontweight='semibold',linespacing=1.5)
        if failures:
            text='\n'.join(f'{which} stopped: {ms:.3f} ms' for which,ms in failures)
            axes[row,0].text(.985,.97,text,transform=axes[row,0].transAxes,
                ha='right',va='top',fontsize=6.8,color='#555555',
                bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=1.8))
        status.append(dict(row=row+1,variables=10,line_counts=[len(ax.lines) for ax in axes[row]],
            base_method=base,short_method=short,long_method=long,failures=failures))
    t0,p0,phi=data['initials'][i]
    fig.suptitle(f'Soft constraint continuation: Cantera vs NN  |  Test IC {i:02d}',x=.54,y=.980,
                 fontsize=21,fontweight='semibold')
    fig.text(.54,.954,f'$T_0$ = {t0:.1f} K     $P_0$ = {p0/101325:.3f} atm     '
        +r'$\phi_0$'+f' = {phi:.3f}     |     {mode.capitalize()} self-fed rollout, '+r'$\Delta t = 2\,\mu$s',
        ha='center',va='center',fontsize=12,color='#414b57')
    handles=[Line2D([0],[0],color=BLACK,ls=(0,(4,2.5)),lw=1.6,label='Cantera reference'),
        Line2D([0],[0],color=GRAY,lw=1.6,label='Baseline10k (mu=0)'),
        Line2D([0],[0],color=BLUE,lw=1.6,label='Soft10k (mu=10)'),
        Line2D([0],[0],color=ORANGE,lw=1.6,label='Soft long (mu=10)')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.941),ncol=4,
               frameon=False,fontsize=11.5,handlelength=3.5,columnspacing=3)
    fig.text(.54,.021,'Shared linear y scale within each variable column. '
        'All curves use saved 2 us samples. Failed predictions stop at their last accepted state (x).\n'
        'Baseline/soft-short:10000 updates. Soft-long: exact Adam/RNG continuation. Mixed teams use the corresponding ANN10 + DeepONet10. '
        'Long stopping requires total, MSE and element L1 plateaus jointly; wall-time stops remain distinct.',
        ha='center',va='center',fontsize=9.5,color='#4b5563',linespacing=1.5)
    if len(fig.axes)!=70:raise AssertionError('Figure must contain exactly7x10 axes')
    return fig,status


def table_figure(frame,common,mode,*,trajectory=None,initial=None):
    summary=trajectory is None
    fig,ax=plt.subplots(figsize=(22,13));ax.axis('off')
    fig.subplots_adjust(left=.02,right=.98,bottom=.16,top=.83)
    rows=[];row_colors=[];row_training=[]
    for method,label,training,group in METHODS:
        part=frame[frame.method==method].set_index('variable')
        values=[]
        for variable in VARIABLES:
            r=part.loc[variable]
            values.append(number(r['median_nrmse'] if summary else r['nrmse'])
                if summary or r['complete'] else '—')
        status=f"{int(part.loc['T','completed_all_test_ICs'])}/18" if summary else (
            'Complete' if part.loc['T','complete'] else f"Stop {part.loc['T','last_accepted_ms']:.3f} ms")
        rows.append([f'{group}  {label}',training]+values+[status])
        color='#f1f5f9' if group%2==0 else '#ffffff'
        row_colors.append([color]*13);row_training.append(training)
    table=ax.table(cellText=rows,colLabels=['Model / same-state mean','Training']+[r'$T$']+COLUMNS[1:]+[
        'Complete ICs' if summary else '10 ms status'],cellColours=row_colors,
        cellLoc='center',colLoc='center',bbox=[0,0,1,1],
        colWidths=[.207,.06]+[.064]*10+[.093])
    table.auto_set_font_size(False);table.set_fontsize(9.5)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#d8dee6');cell.set_linewidth(.45)
        if row==0:
            cell.set_facecolor('#213248');cell.set_text_props(color='white',fontweight='semibold',fontsize=11)
            cell.set_height(cell.get_height()*1.12)
        elif col==0:
            cell.set_text_props(ha='left',fontweight='semibold');cell.PAD=.035
        elif col==1:cell.set_text_props(color=GRAY if row_training[row-1]=='baseline' else (BLUE if row_training[row-1]=='soft10k' else ORANGE),fontweight='bold')
        elif col==12 and not summary and str(rows[row-1][-1]).startswith('Stop'):
            cell.set_text_props(color='#aa3126',fontweight='bold')
    if summary:
        title='Soft constraint | NRMSE summary (util.py: RMS normalization)'
        subtitle=f'Median of per-IC NRMSE on the same {len(common)} complete test ICs for every model'
        extra='Common ICs: '+', '.join(map(str,common))+'.'
    else:
        t,p,phi=initial
        title=f'Cantera vs NN | Test IC {trajectory:02d} | NRMSE (util.py)'
        subtitle=f'T0 = {t:.1f} K    P0 = {p/101325:.3f} atm    phi0 = {phi:.3f}    |    {mode.capitalize()} self-fed, 10 ms'
        extra='All columns compare the NN curve with the same IC Cantera curve. Every graph row includes baseline10k, soft10k and soft-long.'
    fig.suptitle(title,x=.5,y=.96,fontsize=22,fontweight='semibold')
    fig.text(.5,.913,subtitle,ha='center',va='center',fontsize=12,color='#475569')
    fig.text(.5,.871,r'$\mathrm{NRMSE}=\sqrt{\mathrm{mean}[(y_{Cantera}-y_{NN})^2]}/\sqrt{\mathrm{mean}[y_{Cantera}^{2}]}$',
             ha='center',va='center',fontsize=14)
    fig.text(.024,.11,'Exact util.py defaults: norm="rms", reference A=Cantera. '
        'Every time sample has equal weight; entries are dimensionless ratios, without x100.\n'
        'N/A: zero Cantera RMS with a nonzero prediction. Constant nonzero references, including N2, are valid. '
        '—: prediction stopped early; full-horizon NRMSE is unavailable.\n'
        'Baseline/soft-short:10000 updates. Soft-long continues the saved10k states at mu10. Each mixed stage uses the corresponding ANN10 and DeepONet10.\n'
        +extra,ha='left',va='top',fontsize=10,color='#475569',linespacing=1.55)
    return fig




def traces_for(root,cfg,data,mode):
    ids=np.flatnonzero(data['split']=='test').astype(int).tolist();origin=Path(cfg['origin'])
    baseline=json.loads((origin/'baseline_traces.json').read_text());traces={};sources={}
    for name,_ in TEAMS:
        info=baseline[f'{name}_{mode}'];path=Path(info['path'])
        if sha256(path)!=info['sha256']:raise ValueError('Baseline trace changed')
        sources[str(path)]=info['sha256']
        with np.load(path) as f:traces[f'base_{name}']={i:f[f'trajectory_{i}'] for i in ids}
        for stage,parent in [('short',origin),('long',root)]:
            marker=sorted((parent/'evaluations'/f'{name}_{mode}_test').glob('*/complete.json'))[-1]
            m=json.loads(marker.read_text());path=marker.parent/'traces.npz'
            if sha256(path)!=m['output_sha256']['traces.npz']:raise ValueError('Frozen soft trace changed')
            sources[str(path)]=sha256(path)
            with np.load(path) as f:traces[f'{stage}_{name}']={i:f[f'trajectory_{i}'] for i in ids}
    for profiles in traces.values():
        for i,trace in profiles.items():
            assert trace.shape[1]==10 and np.isfinite(trace).all()
            np.testing.assert_array_equal(trace[0],data['states'][i,0])
    return traces,sources


def normalized_metrics(data,traces,ids,mode):
    rows=[]
    for i in ids:
        for method,label,training,group in METHODS:
            trace=traces[method][i];r=profile_errors(data['states'][i],trace,float(data['dt']))
            for k,variable in enumerate(VARIABLES):
                rows.append(dict(trajectory=i,method=method,model=label,training=training,group=group,
                    variable=variable,mode=mode,complete=r['complete'],points=r['points'],
                    last_accepted_ms=(len(trace)-1)*float(data['dt'])*1000,nrmse_status=r['status'][k],
                    nrmse=float(r['nrmse'][k]),full_rmse=float(r['full_rmse'][k]),
                    prefix_rmse=float(r['prefix_rmse'][k]),reference_rms=float(r['reference_rms'][k])))
    frame=pd.DataFrame(rows)
    common=sorted(set.intersection(*[set(p.loc[p.complete,'trajectory']) for _,p in frame.groupby('method')]))
    summary=[]
    for method,label,training,group in METHODS:
        for variable in VARIABLES:
            p=frame[(frame.method==method)&(frame.variable==variable)]
            values=p.loc[p.trajectory.isin(common),'nrmse'].dropna()
            summary.append(dict(method=method,model=label,training=training,group=group,variable=variable,
                mode=mode,common_IC_count=len(common),common_ICs=json.dumps(common),
                valid_nrmse_IC_count=len(values),median_nrmse=values.median(),mean_nrmse=values.mean(),
                std_nrmse=values.std(ddof=0),q25_nrmse=values.quantile(.25),q75_nrmse=values.quantile(.75),
                completed_all_test_ICs=int(p.complete.sum())))
    return frame,pd.DataFrame(summary),common


def training_plots(root,destination):
    records=json.loads((root/'final_members.json').read_text());rows=[];mini=[];table=[]
    for r in records:
        state=json.loads(Path(r['checkpoint'],'state.json').read_text())
        rows.extend(dict(family=r['family'],**h) for h in state['full_train_history'])
        mini.extend(dict(family=r['family'],**h) for h in state['history'])
        table.append({k:v for k,v in r.items() if k not in ('full_train','timing','monitor')})
    history=pd.DataFrame(rows);batch=pd.DataFrame(mini)
    history.to_csv(destination/'observed_full_train_components.csv',index=False)
    batch.to_csv(destination/'observed_minibatch_components.csv',index=False)
    pd.DataFrame(table).to_csv(destination/'training_summary.csv',index=False)
    fig,axes=plt.subplots(4,3,figsize=(17,13),sharex=True)
    for axrow,family in zip(axes,('ann','p_only','tp','tp_phi')):
        selected=history[history.family==family]
        for ax,key in zip(axrow,('data_loss','element_loss','full_train_loss')):
            pivot=selected.pivot(index='seed',columns='update',values=key)
            for values in pivot.to_numpy():ax.plot(pivot.columns,values,lw=.6,alpha=.23,color=BLUE)
            common=pivot.dropna(axis=1)
            mean=np.exp(np.log(np.maximum(common.to_numpy(),1e-300)).mean(axis=0))
            ax.plot(common.columns,mean,lw=1.5,color=ORANGE)
            ax.set_yscale('log');ax.set_title(f'{family}: {key}');ax.grid(True,which='both',ls=':',alpha=.3)
            ax.set_xlabel('Optimizer update');ax.set_ylabel(key)
    fig.suptitle('Soft long mu10: all member tails (thin); geometric mean on common observed updates (thick). Element term is unweighted.')
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(destination/'training_loss_components.png',dpi=160)
    fig.savefig(destination/'training_loss_components.pdf');plt.close(fig)
    return records



def report(root,cfg,snapshot,data,deadline):
    before=time.perf_counter();destination=root/'reports'/f'report_{time.time_ns()}'
    destination.mkdir(parents=True,exist_ok=False);summaries={};sources={}
    revision=root/'source_revisions/report_footer_001/revision.json'
    write_json(destination/'report_source.json',dict(path=__file__,sha256=sha256(__file__),
        prepared_sha256=snapshot['source_sha256']['scripts/report_element_long_comparison.py'],
        revision=json.loads(revision.read_text()) if revision.exists() else None))
    ids=np.flatnonzero(data['split']=='test').astype(int).tolist();record=dict(dt=float(data['dt']),ids=ids)
    with plt.rc_context(STYLE):
        records=training_plots(root,destination)
        for mode in ('repaired','raw'):
            check_deadline(deadline);traces,used=traces_for(root,cfg,data,mode);sources.update(used)
            frame,summary,common=normalized_metrics(data,traces,ids,mode)
            parent=destination/mode;parent.mkdir();png=parent/'png';png.mkdir();tables=parent/'tables';tables.mkdir()
            frame.to_csv(parent/'per_IC_NRMSE.csv',index=False)
            summary.to_csv(parent/'common_IC_NRMSE_summary.csv',index=False)
            summaries[mode]=dict(common_ICs=common,common_IC_count=len(common))
            fig=table_figure(summary,common,mode);fig.savefig(parent/'NRMSE_summary.pdf')
            fig.savefig(parent/'NRMSE_summary.png',dpi=160);plt.close(fig)
            with PdfPages(parent/'profiles_and_NRMSE.pdf') as pdf:
                for page,i in enumerate(ids,1):
                    check_deadline(deadline);fig,status=profile_figure(data,record,traces,i,mode)
                    pdf.savefig(fig);fig.savefig(png/f'test_IC{i:02d}_7x10.png',dpi=160);plt.close(fig)
                    fig=table_figure(frame[frame.trajectory==i],common,mode,trajectory=i,initial=data['initials'][i])
                    pdf.savefig(fig);fig.savefig(tables/f'test_IC{i:02d}_NRMSE.png',dpi=160);plt.close(fig)
                    event(root,phase='report_page_finished',mode=mode,page=page,trajectory=i)
            write_json(parent/'complete.json',dict(pages=36,axes_per_profile=70,metric_rows=len(frame),
                common_ICs=common,output_sha256={str(p.relative_to(parent)):sha256(p) for p in parent.rglob('*') if p.is_file()}))
    old=pd.read_csv(Path(cfg['origin'],'member_diagnostics/loss_components.csv'))
    new=pd.read_csv(root/'member_diagnostics/loss_components.csv')
    diagnostics=pd.concat([old,new],ignore_index=True)
    diagnostics.to_csv(destination/'member_loss_components.csv',index=False)
    diagnostics.groupby(['stage','family','split'])[['data_MSE','physical_element_L1','invalid_inverse_row_fraction']].mean().to_csv(
        destination/'mean_member_loss_components.csv')
    previous=json.loads((root/'original_results_sha256.json').read_text())
    for path,h in previous.items():
        if sha256(path)!=h:raise ValueError(f'Preexisting result changed: {path}')
    verified=checks=0
    for r in records:
        state=json.loads(Path(r['checkpoint'],'state.json').read_text())
        assert r['updates']==state['update']==state['optimizer_iteration'] and r['updates']>=10000
        assert state['signature']['mu']==10
        marker=json.loads(Path(r['checkpoint'],'complete.json').read_text())
        for name,h in marker['output_sha256'].items():
            if sha256(Path(r['checkpoint'],name))!=h:raise ValueError('Terminal checkpoint changed')
        saved=state['plateau_monitor'];replay=joint_monitor(cfg['plateau'],saved['long_run_sha256'],saved['origin_update'],saved['origin_seconds'])
        for row in saved['checks']:
            joint_check(state['full_train_history'],row['update'],replay);checks+=1
        if replay['checks']!=saved['checks'] or replay['components']!=saved['components']:
            raise ValueError('Joint plateau replay differs')
        verified+=1
    flat=sum(r['reason']=='joint_plateau' for r in records);caps=sum(r['reason']=='time_limit' for r in records)
    lines=[
      'Soft constraint mu10 장기 학습과 비교를 완료했다.',
      f'40개 모델: 전체손실/MSE/원소L1 공동 평탄 {flat}개, 배정 시간상한 종료 {caps}개.',
      f'종료 update 범위 {min(r["updates"] for r in records):,}–{max(r["updates"] for r in records):,}.',
      '모든 모델은 기존soft10k Adam/iteration/RNG에서 이어갔다. mu10·전처리·입력분할·architecture·batch512·clip5·절대LR(.0001이후유지)는 바꾸지 않았다.',
      '학습 순서는 ANN10→P-only10→T/P10→T/P/phi10이며, 평가·그림도 순차였다. 전체6시간 중40분을 평가/보고서에 유보했다.',
      '평탄은 전체train관측1k, 검사5k, 최소50k, 연속20k창의q25/median/q75상대변화1%/1%/3%,3회연속10k지속이다. 전체손실·MSE·원소L1 모두 동시에 통과해야 한다.',
      '테스트는 종료점 선택이나 가중치 조정에 사용하지 않았다. 미도달 상한 종료는 수렴으로 표시하지 않았다.',
      '기존loss-only signed BoxCox 역변환 확장과 엄격한 실제추론 도메인, raw/repaired규칙을 유지했다. 새로운penalty나ODEfallback은 없다.',
      '',
      '그림7행×10변수의4개선: 검정Cantera, 회색기존무제약10k, 파랑soft10k, 주황soft장기. mixed팀은 각단계의ANN과DeepONet을 모두 사용한다.',
      '각그림 바로뒤에 같은IC의NRMSE표를 배치했다. raw/repaired각각36페이지, 전체18testIC를 보존했다.',
      'NRMSE=RMSE(Cantera,NN)/RMS(Cantera), 첨부util.py기본값과동일한무차원비율이며 모든시간점에같은가중치를준다.',
      '미완주에는전체NRMSE를부여하지않는다. 전체요약은21개모델구성이모두완주한동일IC집합의중앙값이다. 빈교집합이면N/A로표시하고완주율을함께보인다.',
      f'Repaired 공통 IC: {summaries["repaired"]["common_ICs"]}',
      f'Raw 공통 IC: {summaries["raw"]["common_ICs"]}',
      '보존은repaired후처리에서강제된다. 학습된보존성은별도member진단과raw결과로확인해야한다.',
      f'기존결과파일 {len(previous)}개 SHA256불변.40개endpoint와공동평탄검사 {checks}개를재검산했다.',
      '',
      '[Repaired 그림+NRMSE](repaired/profiles_and_NRMSE.pdf) · [요약표](repaired/NRMSE_summary.pdf)',
      '[Raw 그림+NRMSE](raw/profiles_and_NRMSE.pdf) · [요약표](raw/NRMSE_summary.pdf)',
      '[전체loss성분](training_loss_components.pdf) · [개별member진단](member_loss_components.csv)',
      '새결과는현재클라우드저장상태이며이번장기실험에대한commit/push는하지않았다.','']
    (destination/'REPORT_KO.md').write_text('\n'.join(lines))
    write_json(destination/'verification.json',dict(terminal_checkpoints=verified,joint_checks_recomputed=checks,
        preexisting_files_preserved=len(previous),joint_plateau_members=flat,time_limited_members=caps,comparison=summaries))
    archive=destination/'element_soft_long_comparison_bundle.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in destination.rglob('*'):
            if p.is_file() and p!=archive:z.write(p,arcname=p.relative_to(destination))
    write_json(destination/'complete.json',dict(output_sha256={str(p.relative_to(destination)):sha256(p)
        for p in destination.rglob('*') if p.is_file()}))
    event(root,phase='report_finished',path=str(destination),seconds=time.perf_counter()-before)
    write_json(root/'FINAL_ARTIFACTS.json',dict(report=str(destination),report_complete_sha256=sha256(destination/'complete.json'),
        members=40,mu=10,joint_plateau_members=flat,time_limited_members=caps,comparison=summaries,
        finished_unix=time.time(),pipeline_seconds=time.time()-json.loads((root/'budget.json').read_text())['started_unix']))
    return destination
