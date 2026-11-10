"""Report all frozen new ICs, including unsuccessful and final checkpoints."""
import csv,json,html,time,re
from pathlib import Path
import numpy as np
from .react_deeponet_experiment import write,csv_write,sha,FifthRootScale

LABELS={'best_pretrain':'Best pretrain 72.5k','last_pretrain':'Last pretrain 100k','previous_refinement':'Old K5 refinement','endpoint_selected':'Endpoint: selected','endpoint_final':'Endpoint: final','all_steps_selected':'All steps: selected','all_steps_final':'All steps: final','combined_selected':'Validation-selected combination'}


def report(root,deadline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import ListedColormap
    root=Path(root);out=root/'report'
    if any(out.iterdir()):raise FileExistsError('Do not overwrite report artifacts')
    (out/'figures').mkdir();rows=list(csv.DictReader((root/'evaluations/rollouts.csv').open()));cfg=json.loads((root/'config.json').read_text());selection=json.loads((root/'selection_frozen.json').read_text());variables=json.loads((root/'dataset.json').read_text())['variables'];models=list(selection['models']);repair_models=[m for m in models if selection['models'][m]['role']!='final'];summary=[]
    for split in sorted({r['split'] for r in rows}):
        for model in models:
            for mode in ['raw_full','repaired_full','representative_unconstrained']:
                g=[r for r in rows if r['split']==split and r['model']==model and r['mode']==mode]
                if not g:continue
                complete=[r for r in g if r['complete']=='True'];values=[float(r['worst_NRMSE']) for r in complete]
                summary.append(dict(split=split,model=model,mode=mode,joint=g[0]['joint'],total=len(g),complete=len(complete),physical_valid=sum(r['physical_valid']=='True' for r in g),curve_accuracy_pass=sum(r['curve_accuracy_pass']=='True' for r in g),joint_pass=sum(r['joint_pass']=='True' for r in g),joint_full_and_early=sum(r['joint_full_and_early']=='True' for r in g),median_worst_NRMSE=float(np.median(values)) if values else None))
    csv_write(out/'summary.csv',summary);paired=[]
    for s in summary:
        if s['model']=='best_pretrain':continue
        a={r['case_id']:r for r in rows if r['split']==s['split'] and r['model']=='best_pretrain' and r['mode']==s['mode'] and r['complete']=='True'}
        b={r['case_id']:r for r in rows if r['split']==s['split'] and r['model']==s['model'] and r['mode']==s['mode'] and r['complete']=='True'};common=sorted(set(a)&set(b))
        if common:paired.append(dict(split=s['split'],reference_model='best_pretrain',model=s['model'],mode=s['mode'],joint=s['joint'],common_completed_cases=len(common),reference_median_worst_NRMSE=float(np.median([float(a[i]['worst_NRMSE']) for i in common])),model_median_worst_NRMSE=float(np.median([float(b[i]['worst_NRMSE']) for i in common]))))
    csv_write(out/'common_completed_NRMSE.csv',paired)
    primary=[c for c in json.loads((root/'cases.json').read_text()) if c['split'].startswith('test_')];columns=[i for i,v in enumerate(variables) if v!='AR'];rep=[0,1,2,4,6];prep=json.loads((root/'preprocessing.json').read_text());scale=FifthRootScale(prep['minimum'],prep['maximum']).subset(rep)
    plt.rcParams.update({'font.size':6,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.12})
    with PdfPages(out/'raw_profiles.pdf') as raw_pdf,PdfPages(out/'repaired_profiles.pdf') as repair_pdf,PdfPages(out/'ignition_zoom.pdf') as zoom_pdf,PdfPages(out/'representative_curves.pdf') as rep_pdf:
        for case in primary:
            if time.perf_counter()>deadline:raise TimeoutError('Report guard reached; no success marker')
            traces={}
            for m in models:
                with np.load(root/'evaluations'/case['split']/m/f'IC{case["case_id"]}.npz') as a:traces[m]={k:a[k] for k in a.files}
            ref=traces[models[0]]['reference'];time_ms=np.arange(len(ref))*1e-4;temperature=ref[:,0];change=temperature[-1]-temperature[0]
            if change>50:
                progress=(temperature-temperature[0])/change;lo=int(np.flatnonzero(progress>=.01)[0]);hi=int(np.flatnonzero(progress>=.99)[0]);pad=max(100,(hi-lo)//3);limits=(max(0,lo-pad)*1e-4,min(len(ref)-1,hi+pad)*1e-4)
            else:limits=(0,.1)
            for mode,pdf in [('raw_full',raw_pdf),('repaired_full',repair_pdf),('zoom',zoom_pdf)]:
                names=repair_models if mode=='repaired_full' else models;fig,axes=plt.subplots(len(names),len(columns),figsize=(26,2*len(names)),squeeze=False)
                for row,m in enumerate(names):
                    for col,k in enumerate(columns):
                        ax=axes[row,col];ax.plot(time_ms,ref[:,k],c='black',lw=1.1,label='Cantera')
                        if mode=='zoom':
                            p=traces[m]['raw_full'];ax.plot(np.arange(len(p))*1e-4,p[:,k],c='#2563eb',ls='--',lw=.9,label='Raw NN')
                            if 'repaired_full' in traces[m]:
                                p=traces[m]['repaired_full'];ax.plot(np.arange(len(p))*1e-4,p[:,k],c='#d97706',ls=':',lw=.9,label='Repaired NN')
                        else:
                            p=traces[m][mode];ax.plot(np.arange(len(p))*1e-4,p[:,k],c='#2563eb',ls='--',lw=.9,label='NN')
                        ax.set_xlim(*(limits if mode=='zoom' else (0,1)))
                        if k:ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2),useOffset=False)
                        if row==0:ax.set_title(variables[k])
                        if row==len(names)-1:ax.set_xlabel('Time [ms]')
                        if col==0:ax.set_ylabel(LABELS[m])
                        if row==0 and col==0:ax.legend(fontsize=6)
                fig.suptitle(f'IC{case["case_id"]}: T0={case["T0"]:.3f} K, phi={case["phi"]:.6f}, P=1atm | NEW {case["domain"]} | 50 self-fed 20-us steps',fontsize=11);fig.tight_layout(rect=(0,0,1,.975));pdf.savefig(fig)
                if mode!='repaired_full':fig.savefig(out/'figures'/f'IC{case["case_id"]}_{mode}.png',dpi=90)
                plt.close(fig)
            fig,axes=plt.subplots(1,5,figsize=(17,4));colors=plt.cm.tab10(np.linspace(0,1,len(models)));normalize=lambda a:(np.sign(a)*np.abs(a)**.2-scale.minimum)/scale.span
            reference=normalize(ref[:,rep])
            for j,ax in enumerate(axes):
                ax.plot(time_ms,reference[:,j],c='black',lw=1.4,label='Cantera')
                for m,color in zip(models,colors):
                    a=normalize(traces[m]['representative_unconstrained']);ax.plot(np.arange(len(a))*1e-4,a[:,j],c=color,ls=':' if selection['models'][m]['role']=='final' else '--',lw=.9,label=LABELS[m])
                ax.set_title(variables[rep[j]]);ax.set_xlabel('Time [ms]');ax.set_ylabel('Fifth-root normalized [0,1]')
            handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=7);fig.suptitle(f'IC{case["case_id"]}: unconstrained curve diagnostic; negative values are reported separately',fontsize=10);fig.tight_layout(rect=(0,.14,1,.94));rep_pdf.savefig(fig);plt.close(fig)
    # Show every frozen new IC, never just a favorable example.
    fig,axes=plt.subplots(2,1,figsize=(19,7));cmap=ListedColormap(['#e7e5e4','#fde68a','#93c5fd','#15803d']);cmap.set_bad('#f8fafc')
    for ax,mode in zip(axes,['raw_full','repaired_full']):
        a=np.full((len(models),len(primary)),np.nan)
        for i,m in enumerate(models):
            for j,c in enumerate(primary):
                r=next((r for r in rows if r['case_id']==str(c['case_id']) and r['model']==m and r['mode']==mode),None)
                if r:a[i,j]=3 if r['joint_full_and_early']=='True' else (2 if r['joint_pass']=='True' else (1 if r['complete']=='True' else 0))
        ax.imshow(a,cmap=cmap,vmin=0,vmax=3,aspect='auto');ax.set_yticks(range(len(models)),[LABELS[m] for m in models],fontsize=8);ax.set_xticks(range(len(primary)),[str(c['case_id']) for c in primary],rotation=90,fontsize=6);ax.set_title(mode+': gray=incomplete, yellow=complete, blue=full accuracy, green=full+early accuracy; blank=not evaluated',fontsize=9)
    fig.tight_layout();fig.savefig(out/'coverage.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    chosen={s['name'] for s in selection['finalists']}
    for spec in cfg['candidates']:
        h=list(csv.DictReader((root/'models'/spec['name']/'training.csv').open()));ax=axes[0,0 if spec['family']=='endpoint' else 1];ax.plot([int(x['new_update']) for x in h],[float(x['loss']) for x in h],label=spec['name'],lw=1 if spec['name'] in chosen else .5,alpha=1 if spec['name'] in chosen else .5)
        if spec['name'] in chosen:
            axes[1,0].plot([int(x['new_update']) for x in h],[float(x['lr']) for x in h],label=spec['name']);s=json.loads((root/'models'/spec['name']/'selection.json').read_text());axes[1,1].plot([x['new_update'] for x in s['validations']],[x['pass_both'] for x in s['validations']],label=spec['name'])
    for ax,label in zip(axes.flat,['Endpoint MSE','Mean intermediate-state MSE','Adamax learning rate','Validation full9 AND early passes']):ax.set_ylabel(label);ax.set_xlabel('Additional optimizer updates');ax.legend(fontsize=6)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.set_yscale('log')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'loss_and_lr.{ext}',dpi=150)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    for name in ['reconstruction_uniform','reconstruction_arc']:
        h=list(csv.DictReader((root/'models'/name/'training.csv').open()));ax.plot([int(r['new_update']) for r in h],[float(r['loss']) for r in h],label=name)
    ax.set_yscale('log');ax.set_xlabel('Additional reconstruction-ANN updates');ax.set_ylabel('Same transformed-state MSE');ax.legend();fig.tight_layout();fig.savefig(out/'reconstruction_loss.png',dpi=150);plt.close(fig)
    table='<table><tr>'+''.join('<th>'+v+'</th>' for v in ['평가집합','모델','방식','전체수','완주','정확도+유효성','전체+초기 정확도'])+'</tr>'
    for r in summary:
        if r['split'].startswith('historical_'):continue
        table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['split'],r['model'],r['mode'],r['total'],r['complete'],r['joint_pass'],r['joint_full_and_early']])+'</tr>'
    table+='</table>'
    content='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>DeepONet 후속 학습 안정화</title><style>body{max-width:1200px;margin:40px auto;padding:0 18px;font:16px/1.8 system-ui;color:#172033}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #ccd5e1;padding:6px}th{background:#edf2f9}img{max-width:100%}</style>
<h1>DeepONet 후속 학습의 급격한 성능 저하를 분리한 실험</h1>
<p>이전 완료 실험의 모델·데이터·소스는 그대로 보존했다. 같은 100k 사전학습 가중치와 학습 배치에서 K5로 전환할 때 기울기가 크게 커지는 현상을 확인한 뒤, 학습률 및 Adam 상태만 바꾼 endpoint 대조군과 모든 중간 오차를 평균하는 별도 개발군을 비교했다. 후자는 원래 0D 논문의 끝점 전용 loss와 다르다. 물리 soft constraint, 훈련 중 clipping·보정은 없다.</p>
<p>6개 후보의 동일 업데이트 진단 후 검증만으로 각 family 한 개를 골랐다. 진단 마지막 가중치·Adam·RNG를 정확히 이어 학습했다. 선택 가중치와 마지막 가중치를 둘 다 평가하며, 선택 update가0이면 원래100k 모델로 되돌아간 결과임을 뜻한다. 사전학습 최적72.5k와100k, 이전118.7k 실패 가중치도 고정 대조군이다. loss 정의가 다른 두 family의 loss 숫자를 같은 오차로 직접 비교하면 안 된다.</p>
<p>복원 ANN은 동일한 원본 가중치·Adam·RNG에서 균일 시간 추출과 화학 변화량을 따른 시간 추출을 별도 비교했다. 구조·MSE·학습률·업데이트 목표는 같다. 기본 operator 비교는 원래 복원 ANN을 공통으로 쓴다. combined_selected 한 행만 검증에서 고른 operator/복원 ANN 조합이며, 어떤 조합인지 selection_frozen.json에 명시했다. 복원 ANN 변경의 효과를 operator 자체 개선과 혼동하지 않는다.</p>
<p>훈련99개 궤적과 기존 train-only 정규화를 그대로 썼다. 검증은 기존 보간12개와 새 외삽12개다. 별도로 고정한 새 보간16개·외삽16개는 선택 완료 후 처음 평가했다. historical 결과127개는 이미 본 테스트의 재평가이며 새로운 독립 성능 증거와 구분한다. 단일seed11,1atm,1ms,20us씩50번 자기 예측을 되먹인다.</p>
<p>통과는 각 변수 NRMSE=RMSE(Cantera,NN)/RMS(Cantera)가 모두0.1 이하인 경우다. 비율이며 백분율이 아니다. 전체9는T+8반응종, 대표5는T/H2/H/O2/H2O이고N2/AR은 별도다. 불완주에는 전체기간 점수를 주지 않는다. raw는 음수/온도범위 등의 상태 검사에서 중단한다. unconstrained는 음수도 허용한 곡선 진단이며 유효한 raw 성공으로 세지 않는다. repaired는 clipping+원소투영+엔탈피 보정이 있는 별도 추론 시스템이다. 전체+초기 통과는1ms 전체와초기0.1ms를 모두 만족해야 한다.</p>'''+table+'''
<p><a href="raw_profiles.pdf">Raw 전체변수</a> · <a href="repaired_profiles.pdf">보정 전체변수</a> · <a href="ignition_zoom.pdf">Cantera 점화구간 확대</a> · <a href="representative_curves.pdf">대표변수 곡선 진단</a> · <a href="../evaluations/NRMSE.csv">변수별 NRMSE</a> · <a href="summary.csv">집계표</a> · <a href="common_completed_NRMSE.csv">공통 완주조건 비교</a></p>
<img src="coverage.png" alt="모든 새 테스트 조건의 완주 및 정확도"><img src="loss_and_lr.png" alt="목표별 loss와 학습률, 검증 정확도">
<p>Public Burke10종25개 Cantera entry는 저자가 기술한20반응 파일 자체가 아니다. 비공개 융합 코드와 정확한 반응기구 일치가 확보되지 않은 한계는 남는다. 업데이트 목표나 시간상한 종료를 평탄 도달로 부르지 않는다. 이 보고서의 생성·검증 성공은 문헌 성능 재현 성공을 자동으로 뜻하지 않는다.</p></html>'''
    if cfg.get('software_fixture'):content=content.replace('<h1>','<p><b>SOFTWARE FIXTURE ONLY: not chemistry research results.</b></p><h1>',1)
    (out/'REPORT.html').write_text(content)
    for name in ['raw_profiles.pdf','repaired_profiles.pdf','ignition_zoom.pdf','representative_curves.pdf']:assert len(re.findall(rb'/Type\s*/Page\b',(out/name).read_bytes()))==len(primary)
    write(out/'complete.json',dict(status='passed',pages_per_case_pdf=len(primary),output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
