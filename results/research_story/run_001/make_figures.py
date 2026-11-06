"""Publication-style figures derived only from saved research data."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/kinetics-cache')
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.patches import Rectangle

from prepare_facts import ROOT,REPO,source,REFS,TEAMS,LABELS

FONT=ROOT/'fonts/NanumGothic-Regular.ttf'
font_manager.fontManager.addfont(str(FONT))
font_manager.fontManager.addfont(str(ROOT/'fonts/NanumGothic-Bold.ttf'))
plt.rcParams.update({'font.family':['NanumGothic','DejaVu Sans'],'font.size':12,'axes.unicode_minus':False,
    'mathtext.fontset':'dejavusans',
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,
    'figure.facecolor':'white','savefig.facecolor':'white','axes.titleweight':'bold'})
FACTS=json.loads((ROOT/'sources/facts.json').read_text())
FIGS=[]
BLUE='#2563EB';ORANGE='#EA8B23';NAVY='#10243A';TEAL='#087F8C';RED='#C93D3D';GRAY='#8A98A7'


def save(fig,name,caption,group,sources):
    # ASCII exponent labels avoid missing minus glyphs in mixed Korean math fonts.
    for ax in fig.axes:
        for axis,scale in [(ax.xaxis,ax.get_xscale()),(ax.yaxis,ax.get_yscale())]:
            if scale in ['log','symlog']:
                axis.set_major_formatter(FuncFormatter(lambda value,pos: '0' if value==0 else f'{value:.0e}'.replace('e-0','e-').replace('e+0','e+')))
                if scale=='log':axis.set_major_locator(LogLocator(numticks=5))
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    FIGS.append(dict(name=name,caption=caption,group=group,sources=sources))


def counts_chart(series,methods,labels,name,caption):
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for phase,offset,color in [('short',-.18,BLUE),('long',.18,ORANGE)]:
        rows=FACTS['series'][series[phase]]
        selected=[next(r for r in rows if r['method']==method and r['mode']=='raw') for method in methods]
        for ax,col in zip(axes,['complete','joint9_le_0p1']):
            values=[r[col] for r in selected];bars=ax.bar(np.arange(len(methods))+offset,values,width=.34,label='단기' if phase=='short' else '장기',color=color)
            ax.bar_label(bars,padding=3,fontsize=11)
    for ax in axes:
        ax.set_xticks(np.arange(len(methods)),labels);ax.set_ylim(0,20);ax.set_yticks([0,6,12,18]);ax.set_ylabel('테스트 수 / 18');ax.legend(loc='upper right')
    axes[0].set_title('끝까지 계산된 경우');axes[1].set_title('9개 출력이 모두 정확도 문턱을 통과')
    save(fig,name,caption,'시간·다중 스텝',[f'results/{series[p]}/run_001/report/test_summary.csv' for p in ['short','long']])


def heatmap(ax,matrix,xlabels,ylabels,title,vmax=18):
    im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=vmax,aspect='auto')
    ax.set_xticks(range(len(xlabels)),xlabels);ax.set_yticks(range(len(ylabels)),ylabels);ax.set_title(title)
    ax.grid(False)
    for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,str(int(value)),ha='center',va='center',color='white' if value>vmax*.55 else NAVY,fontsize=12)
    return im


def main():
    s=FACTS['sampling'];selected=pd.read_csv(source('results/long10ms_data/selected_inputs.csv'))
    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    bars=axes[0].bar(['전체 후보','선택된 학습쌍'],[s['dense_pairs'],s['selected_pairs']],color=[GRAY,BLUE]);axes[0].set_yscale('log');axes[0].bar_label(bars,labels=[f'{s["dense_pairs"]:,}',f'{s["selected_pairs"]:,}'],padding=4);axes[0].set_title('학습 입력 시점만 줄였다');axes[0].set_ylim(1000,600000)
    bins=np.linspace(0,10,41)
    axes[1].hist([selected.loc[~selected.equilibrium,'input_time']*1000,selected.loc[selected.equilibrium,'input_time']*1000],bins=bins,stacked=True,color=[BLUE,ORANGE],label=['비평형/유도 구간','평형 앵커'])
    axes[1].set_yscale('log');axes[1].set_xlabel('선택된 입력 시각 [ms]');axes[1].set_ylabel('학습쌍 수');axes[1].set_title('정답은 항상 입력의 2 μs 뒤');axes[1].legend(fontsize=10)
    save(fig,'01_sampling','270,000개 후보에서 8,815개 시작점 선택. 입력을 줄여도 목표 시간 간격은 2 μs로 유지된다.','데이터',['results/long10ms_data/selected_inputs.csv','results/long10ms_data/sampling_summary.csv'])

    main=pd.DataFrame(FACTS['main_baseline']).set_index('method')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    keys=['mean10','mean100'];colors=[BLUE,ORANGE]
    b=axes[0].bar(['ANN 10 평균','ANN 100 평균'],main.loc[keys,'temperature_profile_mean'],color=colors);axes[0].bar_label(b,fmt='%.1f',padding=4);axes[0].set_ylabel('IC별 온도 RMSE 평균 [K]');axes[0].set_ylim(0,330);axes[0].set_title('main의 10 ms 기준 결과')
    b=axes[1].bar(['ANN 10 평균','ANN 100 평균'],[18,18],color=colors);axes[1].bar_label(b,padding=3);axes[1].set_ylim(0,20);axes[1].set_yticks([0,6,12,18]);axes[1].set_ylabel('완주 수 / 18');axes[1].set_title('두 결과 모두 물리 보정을 포함')
    save(fig,'02_main_committee','main의 물리 보정된 기준 비교: 모델 수 100개가 10개보다 온도 오차를 줄이지 못했다. raw 100-member 결과로 해석하면 안 된다.','출발점',['results/committee100_long10ms/summary.csv'])

    rows=pd.DataFrame(FACTS['lambda_short']);fig,ax=plt.subplots(figsize=(10.8,4.8));matrix=np.zeros((7,5),int);weights=[0,1,10,100,1000]
    for i,team in enumerate(TEAMS):
        for j,w in enumerate(weights):matrix[i,j]=int(rows[(rows.team==team)&(rows.weight==w)&(rows['mode']=='raw')].iloc[0].complete)
    heatmap(ax,matrix,[str(w) for w in weights],LABELS,'λ별 raw 10 ms 완주 수 — 모두 10,000 update');ax.set_xlabel('원소 보존 벌점 가중치 λ')
    save(fig,'03_lambda_short','각 칸은 18개 테스트 중 완주 수. λ별 raw 정확도 기준 통과는 전부 0개였다.','Soft constraint',['results/penalty_weight_sweep/run_001/reports/report_1793729103103453738/raw/per_IC_NRMSE.csv'])

    rows=pd.DataFrame(FACTS['soft10']);fig,ax=plt.subplots(figsize=(10.8,4.6));matrix=np.zeros((7,3),int)
    for i,team in enumerate(TEAMS):
        for j,stage in enumerate(['baseline','soft10k','softlong']):matrix[i,j]=int(rows[(rows.team==team)&(rows.training==stage)&(rows['mode']=='raw')].iloc[0].complete)
    heatmap(ax,matrix,['MSE 10k','λ=10 10k','λ=10 장기'],LABELS,'λ=10의 추가 학습에서도 넓은 정확도는 얻지 못했다')
    save(fig,'04_soft10_long','raw 정확도 문턱 통과는 모든 칸에서 0개. 장기 열은 별도 전체 6시간 예산 실험이며 10k 열과 동일 업데이트 비교가 아니다.','Soft constraint',['results/element_soft_long/run_001/reports/report_1793705344040514166/raw/per_IC_NRMSE.csv'])

    gamma=pd.DataFrame(FACTS['gamma']);gammas=[.05,.1,.2,.3,.5,1.]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,stage,title in zip(axes,['short','long'],['기준 10k','추가 활성 계산 150초/모델']):
        matrix=np.zeros((7,6),int)
        for i,team in enumerate(TEAMS):
            for j,g in enumerate(gammas):matrix[i,j]=int(gamma[(gamma.team==team)&(gamma.gamma==g)&(gamma.stage==stage)&(gamma['mode']=='raw')].iloc[0].complete)
        heatmap(ax,matrix,[str(g) for g in gammas],LABELS,title);ax.set_xlabel('Box–Cox γ')
    save(fig,'05_gamma_completion','γ에 따라 완주 수가 달라졌지만 완주가 정확도 통과를 뜻하지는 않는다. γ=.1 일부 기준점은 Adam 부재로 재생성한 이력이 있다.','Box–Cox',['results/gamma_long/run_001/reports/report_1793852690561933910/test/raw/per_IC_NRMSE.csv'])
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,mode in zip(axes,['raw','repaired']):
        matrix=np.zeros((7,6),int)
        for i,team in enumerate(TEAMS):
            for j,g in enumerate(gammas):matrix[i,j]=int(gamma[(gamma.team==team)&(gamma.gamma==g)&(gamma.stage=='long')&(gamma['mode']==mode)].iloc[0].passed)
        heatmap(ax,matrix,[str(g) for g in gammas],LABELS,'보정 없음' if mode=='raw' else '물리 보정 포함',vmax=3);ax.set_xlabel('Box–Cox γ')
    save(fig,'06_gamma_accuracy','장기 γ 실험의 정확도 동시 통과 수. raw의 유일한 통과는 P-only γ=.2의 IC39였다. 보정 포함 결과도 최대 3/18이었다.','Box–Cox',['results/gamma_long/run_001/reports/report_1793852690561933910/test/raw/per_IC_NRMSE.csv','results/gamma_long/run_001/reports/report_1793852690561933910/test/repaired/per_IC_NRMSE.csv'])

    with np.load(ROOT/'sources/selected_traces.npz') as traces:
        reference=traces['IC7_reference'];prediction=traces['IC7_gamma02_ANNP_repaired'];times=np.arange(len(reference))*.002
        trap=FACTS['IC7_trap'];fig,axes=plt.subplots(1,2,figsize=(10.8,4.1))
        axes[0].plot(times,reference[:,0],color='black',label='Cantera');axes[0].plot(times,prediction[:,0],color=ORANGE,ls='--',label='NN + 물리 보정');axes[0].set(xlabel='시간 [ms]',ylabel='온도 [K]',title='IC07: 온도만 보면 오차가 작아 보인다');axes[0].legend(fontsize=10)
        vals=[trap['T_NRMSE'],trap['H2O_NRMSE'],trap['OH_NRMSE']];bars=axes[1].bar(['온도','H₂O','OH'],vals,color=[BLUE,RED,RED]);axes[1].set_yscale('log');axes[1].bar_label(bars,labels=[f'{v:.3g}' for v in vals],padding=5);axes[1].set_ylim(.0002,5000);axes[1].set_ylabel('NRMSE [무차원]');axes[1].set_title('온도 점수와 화학종 점수는 달랐다')
        save(fig,'07_metric_trap','IC07의 γ=.2 ANN+P 장기 repaired 예측. 온도 NRMSE 0.00161만으로 전체 반응계를 잘 맞춘다고 판단할 수 없다.','평가 해석',[ 'selected_trace_provenance.json'])
        fig,axes=plt.subplots(1,2,figsize=(10.8,4))
        for ax,trace,title,color in [(axes[0],reference,'Cantera ΔT',NAVY),(axes[1],prediction,'NN ΔT (물리 보정 포함)',ORANGE)]:
            ax.plot(times,trace[:,0]-trace[0,0],color=color);ax.set(xlabel='시간 [ms]',ylabel='초기 대비 온도 변화 [K]',title=title)
        save(fig,'08_metric_trap_delta','서로 다른 y축 범위를 명시한 온도 변화 비교. Cantera의 총 변화는 약0.0068K, NN은 약2.02K였다.','평가 해석',['selected_trace_provenance.json'])
        ref=traces['IC39_reference'];fig,axes=plt.subplots(2,3,figsize=(11.2,6))
        for ax,col,label in zip(axes.flat,[0,1,4,5,7,8],['T [K]','H₂','O₂','OH','HO₂','H₂O₂']):
            ax.plot(np.arange(len(ref))*.002,ref[:,col],color='black',label='Cantera')
            for key,text,color in [('IC39_legacy_TP_raw','T/P γ=.1 장기',BLUE),('IC39_gamma_P_raw','P-only γ=.2 장기',ORANGE),('IC39_temporal_pilot_long_single_step_raw','P-only 새 대조군',TEAL)]:
                v=traces[key];ax.plot(np.arange(len(v))*.002,v[:,col],ls='--',color=color,label=text)
            ax.set_xlim(0,.15);ax.set_title(label);ax.set_xlabel('시간 [ms]')
            if col:ax.ticklabel_format(style='sci',axis='y',scilimits=(-2,2),useMathText=False)
        axes[0,0].legend(fontsize=8)
        save(fig,'09_IC39_three_raw','세 raw 정확도 통과 설정 모두 같은 IC39였다. 그림은 점화 구간 확대이며 전체10ms/초기0.1ms 점수는 별도 표로 제시한다.','성공 사례의 범위',['results/raw_history_audit/run_001/passing_variable_NRMSE.csv','selected_trace_provenance.json'])

    counts_chart({'short':'temporal_pilot','long':'temporal_pilot_long'},['single_step','interval','multistep'],['P-only 대조','시간 간격','다중 스텝'],'10_deeponet_temporal','DeepONet계 시간 관련 비교는 각3개 평균, 모델당150초+최대150초의 동일 상한이다.')
    counts_chart({'short':'ann_multistep','long':'ann_multistep_long'},['single_step','multistep'],['일반 ANN 단일 스텝','일반 ANN 다중 스텝'],'11_ANN_results','일반 ANN에서는 단기·장기 모두 정확도 문턱 통과가 없었다.')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,family,title in [(axes[0],'ann_multistep','단기'),(axes[1],'ann_multistep_long','장기')]:
        data=FACTS[family+'_all_predictors'];names=['multistep_seed11','multistep_seed22','multistep_seed33','multistep_mean3']
        vals=[next(v['complete'] for v in data if v['predictor']==name and v['mode']=='raw') for name in names]
        bars=ax.bar(['seed11','seed22','seed33','3개 평균'],vals,color=[GRAY,GRAY,GRAY,BLUE]);ax.bar_label(bars,padding=4);ax.set_ylim(0,20);ax.set_yticks([0,6,12,18]);ax.set_title(title);ax.set_ylabel('raw 완주 수 / 18')
    save(fig,'12_ANN_seed_variation','일반 ANN 다중 스텝의 seed별 편차가 크며, 3개 평균이 최선의 단일 seed보다 완주율이 낮았다. 모든 설정의 정확도 통과는0이다.','ANN 다중 스텝',['results/ann_multistep/run_001/report/test_summary.csv','results/ann_multistep_long/run_001/report/test_summary.csv'])

    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    history_sources=[]
    for seed,color in zip([11,22,33],[BLUE,ORANGE,TEAL]):
        path=f'results/ann_multistep_long/run_001/models/multistep_seed{seed}/history.csv';history_sources.append(path)
        history=pd.read_csv(source(path));probe=history.dropna(subset=['probe_loss'])
        axes[0].plot(probe.active_seconds,probe.probe_loss,'o-',ms=3,color=color,label=f'seed{seed}')
        axes[1].step(history.active_seconds,history.learning_rate,where='post',color=color,label=f'seed{seed}')
    for ax in axes:ax.set_yscale('log');ax.set_xlabel('추가 활성 학습 시간 [s]');ax.legend(fontsize=9)
    axes[0].set_title('K=64 고정 training probe MSE');axes[1].set_title('복원된 Adam 상태에서 LR 조정')
    save(fig,'13_ANN_K64_probe','장기에서는 K64를 유지했다. seed11은평탄기준에 도달했지만 정확한10ms 예측을 얻지 못했다. 평탄은정확도 인증이 아니다.','ANN 다중 스텝',history_sources)

    fig,axes=plt.subplots(1,2,figsize=(11.2,4.2));groups=[('temporal_pilot_long','single_step','Deep P 단일'),('temporal_pilot_long','multistep','Deep P 다중'),('ann_multistep_long','single_step','ANN 단일'),('ann_multistep_long','multistep','ANN 다중')]
    for ax,mode in zip(axes,['raw','repaired']):
        values=[next(r['complete'] for r in FACTS['series'][family] if r['method']==method and r['mode']==mode) for family,method,_ in groups]
        bars=ax.bar([g[2] for g in groups],values,color=[NAVY,NAVY,BLUE,BLUE]);ax.bar_label(bars,padding=4);ax.set_ylim(0,20);ax.set_yticks([0,6,12,18]);ax.set_ylabel('완주 수 / 18');ax.set_title('보정 없음' if mode=='raw' else '물리 보정 포함')
    save(fig,'14_latest_completion','같은 최신 시간 예산을 쓴 두 아키텍처 비교. 완주 수의 비교이며 정확도와 보존법칙 만족을 뜻하지 않는다.','최신 비교',['results/temporal_pilot_long/run_001/report/test_summary.csv','results/ann_multistep_long/run_001/report/test_summary.csv'])

    variables=['T','H2','H','O','O2','OH','H2O','HO2','H2O2','N2'];table=pd.DataFrame(FACTS['pass_variables']).pivot(index='configuration',columns='variable',values='nrmse').reindex(columns=variables)
    order=['old_TP_long_gamma0.1','gamma_P_only_long_gamma0.2','current_single_step_long_gamma0.2'];table=table.reindex(order)
    fig,ax=plt.subplots(figsize=(12,2.5));ax.axis('off')
    artist=ax.table(cellText=[[f'{x:.3g}' for x in row] for row in table.to_numpy()],rowLabels=['T/P10 γ=.1 장기','P-only10 γ=.2 장기','P-only3 γ=.2 새 대조'],colLabels=variables,loc='center',cellLoc='center',colWidths=[.075]*10)
    artist.auto_set_font_size(False);artist.set_fontsize(10);artist.scale(1.05,2.)
    for (i,j),cell in artist.get_celld().items():
        cell.set_edgecolor('#D4DCE5')
        if i==0:cell.set_facecolor('#EAF1FB')
    save(fig,'15_pass_variable_table','raw 통과3설정의10변수 NRMSE. 세설정은모두IC39이며 N2를포함해도통과결과가같다.','부록',['results/raw_history_audit/run_001/passing_variable_NRMSE.csv'])

    fig,ax=plt.subplots(figsize=(11,2.7));ax.set_ylim(-.5,1.5);ax.set_xlim(0,300)
    for start,k,color in zip([0,37.5,75,112.5],[1,4,16,64],['#C8D9F7','#9DBEF3','#6295E8',BLUE]):
        ax.broken_barh([(start,37.5)],(.7,.5),facecolors=color);ax.text(start+18.75,.95,f'K={k}',ha='center',va='center',fontsize=13,color=NAVY if k<64 else 'white')
    ax.broken_barh([(150,150)],(.7,.5),facecolors=ORANGE);ax.text(225,.95,'장기: K=64 유지',ha='center',va='center',fontsize=13)
    ax.broken_barh([(0,150)],(-.2,.5),facecolors='#B9C3CE');ax.broken_barh([(150,150)],(-.2,.5),facecolors='#8C9BAA');ax.text(150,.05,'단일 스텝 대조군: K=1 유지',ha='center',va='center',fontsize=12)
    ax.set_yticks([.05,.95],['단일 스텝','다중 스텝']);ax.set_xlabel('누적 활성 학습 시간 [s]');ax.set_title('K는 학습에서 이어가는 횟수 — 실제 예측 간격은 계속2 μs');ax.axvline(150,color=RED,ls='--',lw=1);ax.grid(axis='y',visible=False)
    save(fig,'16_K_curriculum','짧은예측부터길게확장한다. K64에서도1~64번째오차를모두평균하며, 장기는최대150초추가이고평탄종료면더짧다.','학습 방식',['results/ann_multistep/run_001/config.json','results/ann_multistep_long/run_001/config.json'])
    manifest={v['path']:v for v in json.loads((ROOT/'sources/used_source_manifest.json').read_text())};manifest.update(REFS)
    (ROOT/'sources/used_source_manifest.json').write_text(json.dumps(list(manifest.values()),indent=2))
    (ROOT/'figures/figure_manifest.json').write_text(json.dumps(FIGS,indent=2,ensure_ascii=False))
    print('Curated figure pairs generated:',len(FIGS))


if __name__=='__main__':main()
