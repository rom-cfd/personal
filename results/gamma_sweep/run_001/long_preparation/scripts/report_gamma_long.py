"""Paired short/long7x10 figures, exact NRMSE and replayed scalar plateau checks."""
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
from kinetics.ann_deeponet_study import check_deadline
from kinetics.gamma_long_evaluation import stage_profiles
from kinetics.gamma_sweep_training import gamma_tag
from kinetics.soft_constraint_evaluation import TEAMS
from kinetics.soft_constraint_study import event
from kinetics.p_only_training import sha256,write_json
from kinetics.deeponet_long_training import new_monitor,check_plateau
from scripts.report_gamma_sweep import normalized_metrics,summarize,profile_figure,table_figure,GAMMAS,COLORS
from scripts.report_soft_constraint_comparison import limits
from scripts.report_kinetics_nrmse import VARIABLES
from scripts.plot_kinetics_7x10_comparison import STYLE


def paired_summaries(frames):
    joined=pd.concat([f.assign(stage=s,method=s+'_'+f.method) for s,f in frames.items()],ignore_index=True)
    complete={m:set(p.loc[p.complete,'trajectory']) for m,p in joined.groupby('method')}
    common=sorted(set.intersection(*complete.values()))
    within={t:sorted(set.intersection(*(complete[m] for m in joined.loc[joined.team==t,'method'].unique()))) for t,_ in TEAMS}
    a=summarize(joined,{t:common for t,_ in TEAMS},'all84short_long')
    b=summarize(joined,within,'within_team_all12short_long')
    for f in (a,b):f['stage']=f.method.str.split('_').str[0]
    return joined,a,b,common,within


def paired_errors(a,b,left='short',right='long'):
    rows=[]
    for (team,gamma,variable),part in a.groupby(['team','gamma','variable'],sort=False):
        other=b[(b.team==team)&(b.gamma==gamma)&(b.variable==variable)]
        common=sorted(set(part.loc[part.complete,'trajectory'])&set(other.loc[other.complete,'trajectory']))
        av=part.loc[part.trajectory.isin(common),'nrmse'].dropna();bv=other.loc[other.trajectory.isin(common),'nrmse'].dropna()
        rows.append(dict(team=team,model=part.iloc[0].model,gamma=gamma,variable=variable,
            common_ICs=json.dumps(common),common_IC_count=len(common),
            **{left+'_median_NRMSE':av.median(),right+'_median_NRMSE':bv.median(),
               left+'_completed':int(part.complete.sum()),right+'_completed':int(other.complete.sum()),
               left+'_valid_NRMSE_ICs':len(av),right+'_valid_NRMSE_ICs':len(bv)}))
    return pd.DataFrame(rows)


def stage_figure(data,traces,all_traces,i,split,mode,stage):
    fig=profile_figure(data,traces,i,split,mode)
    fig._suptitle.set_text(f'Pure-MSE gamma {stage} | {split.capitalize()} IC {i:02d}')
    ylimits=limits(data['states'][i],all_traces,i)
    for j,ax in enumerate(fig.axes):ax.set_ylim(*ylimits[j%10])
    for text in fig.legends[0].get_texts():
        if 'reused reference' in text.get_text():text.set_text('gamma=0.1 (paired origin)')
    fig.texts[-1].set_text('Short=paired10000-update origins; long=exact state continuation with150s cumulative new-work allowance per model.\n'
        'Gamma.1 ANN/T-P origins were rebuilt because Adam state was missing; bootstrap time is included. Historical controls remain separate.\n'
        'Both stages share y scales. Pure MSE; scalar train-loss plateau or time-limit termination. Actual updates and origins are exported.')
    return fig


def stage_table(frame,common,split,mode,stage,**kwargs):
    fig=table_figure(frame,common,split,mode,**kwargs)
    fig._suptitle.set_text(f'{stage.capitalize()} | '+fig._suptitle.get_text())
    fig.texts[-1].set_text('NRMSE=RMSE(Cantera,NN)/RMS(Cantera), exact user util.py default; equal time samples, raw ratio without ×100.\n'
        'Incomplete profiles have no full-horizon score. Summary cohorts are explicitly matched across short and long; empty sets are N/A.\n'
        'Short origins:10000 updates. Long:variable actual updates; train-MSE plateau and150s active-time stops are distinguished. Gamma.1 ANN/T-P short origins are rebuilt.')
    return fig


def training_outputs(root,destination,records):
    pd.DataFrame([dict({k:v for k,v in r.items() if k not in ('monitor','allowance','last_full_train')},
        new_work_seconds=r['allowance']['consumed_seconds'],member_budget_seconds=150) for r in records]).to_csv(destination/'training_summary.csv',index=False)
    rows=[]
    for r in records:
        if r.get('borrowed_short_endpoint'):continue
        state=json.loads((Path(r['checkpoint'])/'state.json').read_text())
        history=[h for h in state['full_train_history'] if h['update']>=10000]
        if not history or history[0]['full_train_loss']<=0:continue
        base=history[0]['full_train_loss']
        rows.extend(dict(h,gamma=r['gamma'],family=r['family'],loss_ratio_to_short=h['full_train_loss']/base) for h in history)
    history=pd.DataFrame(rows,columns=['seed','update','full_train_loss','gamma','family','loss_ratio_to_short'])
    history.to_csv(destination/'observed_full_train_MSE.csv',index=False)
    fig,axes=plt.subplots(4,6,figsize=(24,13),sharex=False)
    for row,family in enumerate(('ann','p_only','tp','tp_phi')):
        for col,gamma in enumerate(GAMMAS):
            ax=axes[row,col];part=history[(history.family==family)&(history.gamma==gamma)]
            if len(part):
                for _,p in part.groupby('seed'):ax.plot(p['update'],p.loss_ratio_to_short,color=COLORS[gamma],lw=.6,alpha=.25)
                pivot=part.pivot(index='seed',columns='update',values='loss_ratio_to_short').dropna(axis=1)
                if len(pivot.columns):ax.plot(pivot.columns,np.exp(np.log(np.maximum(pivot.to_numpy(),1e-300)).mean(axis=0)),color=COLORS[gamma],lw=1.5)
            group=[r for r in records if r['gamma']==gamma and r['family']==family]
            ax.set_title(f'{family}, g={gamma:g}\nplateau {sum(r["reason"]=="plateau" for r in group)}/10',fontsize=10)
            ax.set_yscale('log');ax.grid(True,which='both',ls=':',alpha=.3);ax.set_xlabel('Absolute optimizer update',fontsize=8)
            if col==0:ax.set_ylabel('MSE / paired-short MSE')
    fig.suptitle('Pure train-MSE continuation: individual observed tails; thick mean only on common observed updates. No physical penalty.')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(destination/'training_loss_ratios.pdf');fig.savefig(destination/'training_loss_ratios.png',dpi=150);plt.close(fig)


def verify(root,cfg,snapshot):
    records=json.loads((root/'final_members.json').read_text());origins=json.loads((root/'paired_short_members.json').read_text());checks=0
    if len(records)!=240:raise ValueError('Need240 frozen endpoints')
    for r in records:
        origin=next(x for x in origins if (x['gamma'],x['family'],x['seed'])==(r['gamma'],r['family'],r['seed']))
        cp=Path(r['checkpoint']);marker=json.loads((cp/'complete.json').read_text());state=json.loads((cp/'state.json').read_text())
        for name,expected in marker['output_sha256'].items():
            if sha256(cp/name)!=expected:raise ValueError('Long checkpoint changed')
        with np.load(cp/'adam.npz') as f:
            if int(f['v0'])!=r['updates']:raise ValueError('Actual Adam update differs')
        if state['update']!=r['updates'] or state['optimizer_iteration']!=r['updates']:raise ValueError('Update provenance differs')
        before=json.loads((Path(origin['checkpoint'])/'model.json').read_text());after=json.loads((cp/'model.json').read_text())
        for key in ('output_mean','output_scale','input_mean','input_scale','epsilon'):
            np.testing.assert_array_equal(before[key],after[key])
        if r['family']=='tp_phi':
            for key in ('phi_mean','phi_scale'):np.testing.assert_array_equal(before[key],after[key])
        if before['spec']['gamma']!=after['spec']['gamma']:raise ValueError('Gamma changed during continuation')
        old_state=json.loads((Path(origin['checkpoint'])/'state.json').read_text())
        for key in ('history','full_train_history'):
            if state[key][:len(old_state[key])]!=old_state[key]:raise ValueError('Original history prefix changed')
        allowance=r['allowance']
        if allowance['seconds']!=150 or allowance['open_visit'] is not None:raise ValueError('Member allowance differs')
        if r.get('borrowed_short_endpoint'):
            if r['added_updates'] or r['reason']!='member_time_limit':raise ValueError('Borrowed short endpoint mislabeled')
            continue
        saved=state['plateau_monitor']
        if state['signature'].get('loss')!='transformed_residual_MSE_only':raise ValueError('Unexpected training loss')
        if saved['long_run_sha256']!=sha256(root/'run.json'):raise ValueError('Long lineage differs')
        replay=new_monitor(cfg['plateau'],origin_update=10000,origin_seconds=saved['origin_seconds'])
        for row in saved['checks']:check_plateau(state['full_train_history'],row['update'],replay)
        if replay['checks']!=saved['checks'] or replay['plateau']!=saved['plateau']:raise ValueError('Plateau replay differs')
        if (r['reason']=='plateau')!=bool(saved['plateau']):raise ValueError('Time cap labeled as plateau')
        checks+=len(saved['checks'])
    for mapping in ('source_sha256','origin_sha256'):
        for path,expected in snapshot[mapping].items():
            if sha256(path)!=expected:raise ValueError('Frozen source changed')
    previous=json.loads((root/'original_results_sha256.json').read_text())
    for path,expected in previous.items():
        if sha256(path)!=expected:raise ValueError(f'Preexisting result changed: {path}')
    return records,checks,len(previous)


def report(root,cfg,snapshot,data,deadline):
    before=time.perf_counter();destination=root/'reports'/f'report_{time.time_ns()}';destination.mkdir(parents=True,exist_ok=False)
    write_json(destination/'run_snapshot.json',snapshot)
    selection=json.loads((root/'long_validation_selection.json').read_text());write_json(destination/'long_validation_selection.json',selection)
    records=json.loads((root/'final_members.json').read_text());sources={};comparisons={}
    with plt.rc_context(STYLE):
        training_outputs(root,destination,records)
        for split in cfg['evaluation_splits']:
            comparisons[split]={};ids=np.flatnonzero(data['split']==split).astype(int).tolist()
            for mode in ('repaired','raw'):
                check_deadline(deadline);profiles={};frames={}
                for stage in ('short','long'):
                    profiles[stage],used=stage_profiles(root,cfg,snapshot,data,stage,split,mode)
                    sources.update({x['path']:x['sha256'] for x in used.values()})
                    frames[stage]=normalized_metrics(data,profiles[stage],ids,split,mode)[0]
                joined,summary,within_summary,common,within=paired_summaries(frames)
                folder=destination/split/mode;folder.mkdir(parents=True);png=folder/'png';png.mkdir();tables=folder/'tables';tables.mkdir()
                joined.to_csv(folder/'per_IC_NRMSE.csv',index=False);summary.to_csv(folder/'common_IC_NRMSE_summary.csv',index=False)
                within_summary.to_csv(folder/'within_team_NRMSE_summary.csv',index=False)
                pair=paired_errors(frames['short'],frames['long']);pair.to_csv(folder/'same_gamma_short_long_summary.csv',index=False)
                if split=='test':
                    pieces=[pair[(pair.team==t)&(pair.gamma==v['chosen_gamma'])] for t,v in selection['choices'].items() if v['chosen_gamma'] is not None]
                    selected=pd.concat(pieces,ignore_index=True) if pieces else pair.iloc[:0]
                    selected.to_csv(folder/'validation_selected_same_gamma_short_long.csv',index=False)
                historical=pd.read_csv(Path(snapshot['origin_report'])/split/mode/'per_IC_NRMSE.csv')
                paired_errors(historical[historical.gamma==.1],frames['short'][frames['short'].gamma==.1],left='historical',right='paired_short').to_csv(folder/'gamma01_historical_vs_paired_short.csv',index=False)
                comparisons[split][mode]=dict(common_ICs=common,within_team_common_ICs=within)
                with PdfPages(folder/'within_team_NRMSE.pdf') as pdf:
                    for team,_ in TEAMS:
                        for stage in ('short','long'):
                            s=within_summary[(within_summary.team==team)&(within_summary.stage==stage)].copy();s['method']=s.method.str[len(stage)+1:]
                            fig=stage_table(s,within[team],split,mode,stage,team=team);pdf.savefig(fig);plt.close(fig)
                all_traces={s+'_'+m:p for s,t in profiles.items() for m,p in t.items()}
                with PdfPages(folder/'profiles_and_NRMSE.pdf') as pdf:
                    for page,i in enumerate(ids,1):
                        for stage in ('short','long'):
                            check_deadline(deadline);fig=stage_figure(data,profiles[stage],all_traces,i,split,mode,stage)
                            pdf.savefig(fig);fig.savefig(png/f'{split}_IC{i:02d}_{stage}_7x10.png',dpi=150);plt.close(fig)
                            fig=stage_table(frames[stage][frames[stage].trajectory==i],common,split,mode,stage,trajectory=i,initial=data['initials'][i])
                            pdf.savefig(fig);fig.savefig(tables/f'{split}_IC{i:02d}_{stage}_NRMSE.png',dpi=140);plt.close(fig)
                            event(root,phase='report_page_finished',split=split,mode=mode,stage=stage,page=page,trajectory=i)
                write_json(folder/'complete.json',dict(pages=72,within_team_pages=14,common_ICs=common,
                    output_sha256={str(p.relative_to(folder)):sha256(p) for p in folder.rglob('*') if p.is_file()}))
                del profiles,all_traces
    records,checks,preserved=verify(root,cfg,snapshot);flat=sum(r['reason']=='plateau' for r in records)
    write_json(destination/'trace_sources.json',sources)
    write_json(destination/'verification.json',dict(endpoints=240,actual_Adam_iterations_verified=240,scalar_plateau_checks_recomputed=checks,
        plateau_members=flat,time_limited_members=240-flat,original_files_preserved=preserved,rebuilt_short_origins=20,
        exact_saved_state_origins=220,comparison=comparisons))
    (destination/'REPORT_KO.md').write_text('\n'.join([
        '6개gamma,4개계열,10개seed의240개모델을 pure MSE로 장기학습했다. 물리soft constraint는없다.',
        f'훈련MSE 평탄{flat}/240, 시간상한{240-flat}/240. 모델별새활성계산150초이며gamma.1 ANN/T-P20개의필수10k재생성시간도포함했다.',
        '나머지220개는기존Adam/RNG를정확히이어받았다. gamma.1 ANN/T-P20개는역사적가중치의exact continuation이아닌재생성10k와짝지은continuation이다.',
        'gamma01_historical_vs_paired_short.csv에역사적gamma.1과새paired-short의차이를명시했다. 이전결과는보존했다.',
        '평탄기준은최소50k, 관측1k/검사5k, 연속20k창q25/median/q75상대변화1%/1%/3%,10k지속이다. 추가update상한과LR리셋은없다.',
        '개별150초를bootstrap/continuation방문간누적하며대기/평가시간은제외한다. 재개는새150초를부여하지않는다. 실제시간/update/종료사유를수록했다.',
        '각IC마다short7x10+NRMSE표,long7x10+NRMSE표를연속배치했다. 두stage의변수열y축범위는같다. 각그림은Cantera+6개gamma곡선이다.',
        'NRMSE=RMSE(Cantera,NN)/RMS(Cantera), user util.py기본값,동일시간점가중치,무차원비율. 미완주는전체점수없음,빈공통집합은N/A다.',
        '전체/팀내요약은short+long이공통완주한IC이며,same_gamma_short_long_summary.csv는각gamma/팀의paired공통IC를별도로쓴다.',
        'long gamma 선택은validation repaired완주수우선,동률공통IC온도NRMSE로고정한뒤test를평가했다. 선택gamma의동일gamma short/long test비교도제공한다.',
        '기존에관찰한test의탐색적추가학습평가다. 모든종에대한단일최적gamma 또는수렴을검증했다고표현하지않는다.',
        f'240개실제Adam update,평탄검사{checks}개와기존파일{preserved}개의SHA보존을검증했다.','',
        '[Test repaired short/long 그림+표](test/repaired/profiles_and_NRMSE.pdf)',
        '[Test raw short/long 그림+표](test/raw/profiles_and_NRMSE.pdf)',
        '[팀별gamma short/long NRMSE](test/repaired/within_team_NRMSE.pdf)',''])+'\n')
    for split in cfg['evaluation_splits']:
        for mode in ('repaired','raw'):
            check_deadline(deadline)
            with zipfile.ZipFile(destination/f'{split}_{mode}_bundle.zip','x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                shared=[p for p in destination.iterdir() if p.is_file() and p.suffix!='.zip']
                for p in shared+[p for p in (destination/split/mode).rglob('*') if p.is_file()]:z.write(p,arcname=p.relative_to(destination))
    write_json(destination/'complete.json',dict(output_sha256={str(p.relative_to(destination)):sha256(p) for p in destination.rglob('*') if p.is_file()}))
    budget=json.loads((root/'budget.json').read_text())
    write_json(root/'FINAL_ARTIFACTS.json',dict(report=str(destination),report_complete_sha256=sha256(destination/'complete.json'),
        members=240,plateau_members=flat,time_limited_members=240-flat,comparison=comparisons,
        validation_selection=selection,pipeline_seconds=time.time()-budget['started_unix']))
    event(root,phase='report_finished',path=str(destination),seconds=time.perf_counter()-before)
    return destination
