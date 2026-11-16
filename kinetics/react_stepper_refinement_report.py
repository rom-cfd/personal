"""Independent evaluations and reports; test access follows frozen selection."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,time
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from .react_stepper_reproduction import feedback,score,state_valid,sha,write,csv_write,REP,REST,ACTIVE

def evaluate(root,cfg,scale,models,deadline):
    from .react_stepper_refinement import load_operator,load_reconstruction
    selection=json.loads((root/'selection_frozen.json').read_text());assert selection['status']=='passed'
    write(root/'test_access_started.json',dict(at=datetime.now(timezone.utc).isoformat(),selection_sha256=sha(root/'selection_frozen.json')))
    cases=json.loads((root/'cases.json').read_text());reference={}
    for split in ['interpolation','extrapolation']:
        with np.load(root/'data'/f'test_{split}.npz') as f:array=f['states']
        for c,a in zip([c for c in cases if c['split']==split],array):reference[c['IC']]=a
    truth=np.stack([reference[c['IC']] for c in cases]);np.savez_compressed(root/'evaluations/reference.npz',states=truth)
    variables=['T','H2','H','O','O2','OH','H2O','HO2','H2O2','AR','N2'];metrics=[];rows=[]
    prep=json.loads((root/'preprocessing.json').read_text());minimum=np.asarray(prep['minimum'])[REP];span=np.asarray(prep['maximum'])[REP]-minimum
    for name,spec in models.items():
        assert sha(root/spec['operator'])==spec['operator_sha256'] and sha(root/spec['reconstruction'])==spec['reconstruction_sha256']
        operator=load_operator(root/spec['operator']);cnet=load_reconstruction(root/spec['reconstruction'])
        for stride in ([200,20] if name in ['baseline','combined_selected'] else [200]):
            if time.perf_counter()>=deadline:raise TimeoutError('Evaluation budget exhausted')
            write(root/'status.json',dict(phase='test_evaluation',model=name,step_us=stride/10))
            pred,info=feedback(operator,cnet,scale,truth[:,0],[c['phi'] for c in cases],stride,10000)
            np.savez_compressed(root/'evaluations'/f'{name}_{stride}.npz',prediction=pred,**info)
            for i,c in enumerate(cases):
                complete=bool(info['raw_boundary_valid'][i]);dense_valid=bool(state_valid(pred[i]).all());scores={}
                for grid in ['boundaries','dense']:
                    indices=np.arange(0,10001,stride if grid=='boundaries' else 1)
                    for window in ['full_1ms','early_0p1ms']:
                        ix=indices if window=='full_1ms' else indices[indices<=1000]
                        rmse,rms,nrmse=score(truth[i,ix],pred[i,ix]);scores[(grid,window)]=nrmse
                        for j,var in enumerate(variables):
                            metrics.append(dict(model=name,step_us=stride/10,**c,grid=grid,window=window,variable=var,
                                RMSE=float(rmse[j]),RMS_Cantera=float(rms[j]),diagnostic_NRMSE=float(nrmse[j]) if np.isfinite(nrmse[j]) else '',
                                raw_NRMSE=float(nrmse[j]) if complete and np.isfinite(nrmse[j]) else '',raw_complete=complete,dense_physical_valid=dense_valid))
                a=scores[('dense','full_1ms')];b=scores[('dense','early_0p1ms')];boundary=scores[('boundaries','full_1ms')]
                norm_true=2*(truth[i][:,REP]**.2-minimum)/span-1;v=pred[i][:,REP]
                norm_pred=2*(np.sign(v)*np.abs(v)**.2-minimum)/span-1
                rows.append(dict(model=name,step_us=stride/10,**c,raw_complete=complete,
                    first_invalid_step=int(info['first_invalid_step'][i]),
                    raw_boundary_joint9=bool(complete and np.all(boundary[ACTIVE]<=.1)),
                    raw_dense_joint5=bool(complete and np.all(a[REP]<=.1)),
                    raw_dense_joint5_full_early=bool(complete and np.all(a[REP]<=.1) and np.all(b[REP]<=.1)),
                    raw_dense_joint9=bool(complete and np.all(a[ACTIVE]<=.1)),
                    raw_dense_joint9_full_early=bool(complete and np.all(a[ACTIVE]<=.1) and np.all(b[ACTIVE]<=.1)),
                    dense_valid=dense_valid,dense_all_success=bool(dense_valid and np.all(a[ACTIVE]<=.1) and np.all(b[ACTIVE]<=.1)),
                    raw_normalized5_MAE=float(np.mean(np.abs(norm_pred-norm_true))) if complete else '',
                    minimum_boundary_Y=float(pred[i,::stride,1:].min()),minimum_dense_Y=float(pred[i,:,1:].min()),
                    maximum_boundary_sumY_error=float(np.max(np.abs(pred[i,::stride,1:].sum(axis=1)-1)))))
    csv_write(root/'evaluations/NRMSE.csv',metrics);csv_write(root/'evaluations/cases.csv',rows)
    summary=[]
    keys=['raw_complete','raw_boundary_joint9','raw_dense_joint5','raw_dense_joint5_full_early',
          'raw_dense_joint9','raw_dense_joint9_full_early','dense_valid','dense_all_success']
    for name in models:
        for step in [20,2]:
            for split in ['interpolation','extrapolation']:
                v=[r for r in rows if r['model']==name and r['step_us']==step and r['split']==split]
                if v:summary.append(dict(model=name,step_us=step,split=split,total=len(v),**{k:sum(r[k] for r in v) for k in keys}))
    csv_write(root/'report/summary.csv',summary)
    # Common-completion comparisons use identical ICs and variables, never each
    # model's separately successful subset.
    comparisons=[]
    for step in [20,2]:
        for split in ['interpolation','extrapolation']:
            complete={name:{r['IC'] for r in rows if r['model']==name and r['step_us']==step and r['split']==split and r['raw_complete']}
                      for name in ['baseline','combined_selected']}
            common=complete['baseline']&complete['combined_selected']
            for grid in ['boundaries','dense']:
                for window in ['full_1ms','early_0p1ms']:
                    for var in variables:
                        values={name:[r['raw_NRMSE'] for r in metrics if r['model']==name and r['step_us']==step and r['split']==split
                            and r['IC'] in common and r['grid']==grid and r['window']==window and r['variable']==var and r['raw_NRMSE']!='']
                            for name in ['baseline','combined_selected']}
                        comparisons.append(dict(step_us=step,split=split,grid=grid,window=window,variable=var,common_ICs=len(common),
                            baseline_median=float(np.median(values['baseline'])) if values['baseline'] else '',
                            combined_median=float(np.median(values['combined_selected'])) if values['combined_selected'] else ''))
    csv_write(root/'report/common_completed_NRMSE.csv',comparisons)
    # Given-truth reconstruction is diagnostic only; those states are never
    # fed to the self-fed model evaluation above.
    teacher=[];encoded=scale.forward(truth);ss=scale.subset(REST)
    for name,path in [('original','inputs/reconstruction.pt'),('combined',models['combined_selected']['reconstruction'])]:
        model=load_reconstruction(root/path)
        with torch.inference_mode():pred=ss.inverse(model(torch.from_numpy(encoded[:,:,REP].reshape(-1,5))).numpy()).reshape(len(cases),10001,6)
        for i,c in enumerate(cases):
            for window,end in [('full_1ms',10001),('early_0p1ms',1001)]:
                a=score(truth[i,:end][:,REST],pred[i,:end])[2]
                for j,var in enumerate(np.asarray(variables)[REST]):
                    teacher.append(dict(reconstruction=name,**c,window=window,variable=str(var),NRMSE=float(a[j]) if np.isfinite(a[j]) else '',
                        physical_valid=bool(np.isfinite(pred[i,:end]).all() and pred[i,:end].min()>=0),given_truth_diagnostic=True))
    csv_write(root/'evaluations/reconstruction_given_truth.csv',teacher)
    write(root/'evaluation_complete.json',dict(status='passed',cases=len(cases),models=list(models),selection_sha256=sha(root/'selection_frozen.json')))

def report(root,cfg,deadline):
    cases=json.loads((root/'cases.json').read_text());selection=json.loads((root/'selection_frozen.json').read_text());models=selection['models']
    truth=np.load(root/'evaluations/reference.npz')['states'];variables=['T','H2','H','O','O2','OH','H2O','HO2','H2O2','AR','N2']
    groups=[('operator_profiles.pdf',['baseline','uniform_selected','fixed20_selected'],False),
            ('combined_profiles.pdf',['baseline','combined_selected','cnet_only'],False),
            ('ignition_zoom.pdf',['baseline','combined_selected','cnet_only'],True),
            ('final_weights.pdf',['uniform_selected','uniform_final','fixed20_selected','fixed20_final'],False)]
    pages={}
    for filename,names,zoom in groups:
        names=[n for n in names if n in models]
        arrays={n:np.load(root/'evaluations'/f'{n}_200.npz')['prediction'] for n in names}
        with PdfPages(root/'report'/filename) as pdf:
            for i,c in enumerate(cases):
                if time.perf_counter()>=deadline:raise TimeoutError('Report deadline reached')
                fig,axes=plt.subplots(3,4,figsize=(19,12),layout='constrained');t=np.arange(10001)*.1
                peak=int(np.argmax(np.gradient(truth[i,:,0])));low=max(0,peak-200) if zoom else 0;high=min(10001,peak+401) if zoom else 10001
                for j,ax in enumerate(axes.flat):
                    if j==11:ax.axis('off');continue
                    ax.plot(t[low:high],truth[i,low:high,j],'k',lw=1.4,label='Cantera')
                    for name,pred in arrays.items():ax.plot(t[low:high],pred[i,low:high,j],ls='--',lw=1,label=name)
                    ax.set(title=variables[j]+(' [K]' if j==0 else ' mass fraction'),xlabel='time [microseconds]');ax.grid(alpha=.2)
                handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower right',fontsize=9)
                fig.suptitle(f"{c['split']} IC{c['IC']} T0={c['T0']:.3f}K phi={c['phi']:.6f}\n20us self-feedback; lines include interior diagnostics; no repair",fontsize=13)
                pdf.savefig(fig);plt.close(fig)
        pages[filename]=len(cases)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    summary=list(csv.DictReader((root/'report/summary.csv').open()))
    for ax,split in zip(axes,['interpolation','extrapolation']):
        names=['baseline','combined_selected','cnet_only'];keys=['raw_complete','raw_dense_joint5_full_early','raw_dense_joint9_full_early','dense_all_success']
        for j,key in enumerate(keys):
            v=[next(int(r[key]) for r in summary if r['model']==name and r['split']==split and float(r['step_us'])==20) for name in names]
            ax.bar(np.arange(3)+(j-1.5)*.18,v,.18,label=key)
        ax.set(xticks=np.arange(3),xticklabels=names,title=split,ylabel='new independent ICs');ax.legend(fontsize=7)
    fig.savefig(root/'report/coverage.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
    for path in sorted((root/'models').glob('*/training.csv')):
        if path.parent.name.startswith('cnet'):continue
        data=list(csv.DictReader(path.open()))
        if not data:continue
        x=[int(r['new_update']) for r in data]
        axes[0].plot(x,[float(r['loss']) for r in data],label=path.parent.name)
        axes[1].plot(x,[float(r['lr']) for r in data],label=path.parent.name)
    for ax in axes:
        ax.set_yscale('log');ax.legend(fontsize=8);ax.grid(alpha=.2)
        for n in [cfg['curriculum_stage_updates'],2*cfg['curriculum_stage_updates']]:ax.axvline(n,color='gray',ls=':')
    axes[0].set(ylabel='normalized endpoint MSE',title='K1->K2->K5; losses across different K are not directly comparable')
    axes[1].set(xlabel='additional updates',ylabel='learning rate')
    fig.savefig(root/'report/loss_and_lr.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for path in sorted((root/'models').glob('cnet_*/training.csv')):
        data=list(csv.DictReader(path.open()))
        if data:ax.plot([int(r['new_update']) for r in data],[float(r['loss']) for r in data],label=path.parent.name)
    ax.set(yscale='log',xlabel='additional updates',ylabel='normalized MSE');ax.legend();ax.grid(alpha=.2)
    fig.savefig(root/'report/reconstruction_loss.png',dpi=150);plt.close(fig)
    from html import escape
    table=''.join('<tr>'+''.join('<td>'+escape(str(r[k]))+'</td>' for k in ['model','step_us','split','total','raw_complete',
        'raw_dense_joint5_full_early','raw_dense_joint9_full_early','dense_all_success'])+'</tr>' for r in summary)
    choice=selection['combined_choice']
    html='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>20 μs React-DeepONet 후속 성능 재현</title>
<style>body{max-width:1250px;margin:32px auto;padding:0 20px;font:16px/1.8 system-ui;color:#173044}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #bdcbd8;padding:7px}img{max-width:100%}pre{white-space:pre-wrap}</style></head><body>
<h1>20 μs 되먹임 React-DeepONet의 후속 안정화 실험</h1>
<p>문헌의 현재 상태→짧은 미래→예측 상태 되먹임 방식을 유지했다. 전역 flexDeepONet 결과가 아니다.
기존 데이터99조건·기구·train-only5제곱근 정규화·42,236파라미터 구조는 같고, 사전학습72,500회 가중치·Adam 상태·RNG에서 이어갔다.
seed11 하나이며 앙상블/여러seed평균이 아니다. 물리 soft constraint, 학습/추론 보정, clipping은 없다.</p>
<p>4후보는 작은 LR1e-5/3e-6와 양의 균일0.1–40 μs/고정20 μs 간격을 비교했다.
각후보는 K1→K2→K5를 거치고, loss는 마지막 끝점MSE이며 모든 되먹임을 통해 미분한다.
각간격 family는 마지막 진단 검증값으로 하나를 고르고, 그 마지막 가중치·Adam·RNG에서 정확히 이어 학습했다.
선택 가중치는 원래update0도 포함해 보존했다. update0선택은 추가학습 개선이 없는 fallback이다.</p>
<p>복원ANN은 원래5→64→64→6 tanh 구조·MSE·가중치·Adam·RNG를 공통으로 사용해 균일시간과 절반균일/절반초기0.1ms 추출을 비교했다.
Operator 단독 비교는 원래 복원ANN을 공통사용한다. cnet_only는 기존operator+선택CNET,
combined_selected만 검증에서 고른 operator/CNET조합이다. CNET개선과operator개선을 섞지 않는다.</p>
<p>선택조합: <strong>CHOICE</strong>. 검증은 기존 보간12+외삽12조건이고,
새 독립 보간32+외삽32는 기존 테스트/소프트웨어fixture와 겹치지 않으며 최종 선택동결 뒤에만 열었다.
원래 학습 범위 밖의 외삽 조건을 학습데이터에 넣지 않았다.</p>
<p>실제진행은20 μs×50회=1ms. 0.1μs 내부질의는추가진행스텝이 아니다.
raw완주는실제로되먹이는끝점의유한성/비음수농도/온도200–6000K로 판정하며, 실패하면전체기간raw점수가없다.
내부곡선의음수와물리유효성은별도로기록한다. 보존법칙의정확한충족을이상태검사만으로주장하지않는다.</p>
<p>NRMSE=RMSE(Cantera,NN)/RMS(Cantera),무차원비율이며백분율아님.
joint5=T/H₂/H/O₂/H₂O, joint9=T+8반응종 각각≤0.1이며N₂/AR은별도다.
표는0.1μs 조밀격자의전체1ms와초기0.1ms를동시에통과한수다.
2μs는baseline/combined만별도로평가하며20μs성공을2μs성공으로부르지않는다.</p>
<table><tr><th>모델</th><th>진행μs</th><th>영역</th><th>수</th><th>경계raw완주</th><th>조밀전체+초기joint5</th><th>조밀전체+초기joint9</th><th>조밀상태유효성까지동시통과</th></tr>TABLE</table>
<p>선택모델과literal마지막가중치를둘다평가했다. 실제update/LR가다르면간격하나만의효과라고단정하지않는다.
시간상한/요청update종료는평탄도달이아니다. 같은IC/변수의공통raw완주집합으로만NRMSE중앙값을비교했다.</p>
<p><a href="operator_profiles.pdf">Operator 전체곡선</a> · <a href="combined_profiles.pdf">최종조합 전체곡선</a> ·
<a href="ignition_zoom.pdf">Cantera점화구간 가변확대</a> · <a href="final_weights.pdf">선택 vs마지막가중치</a> ·
<a href="../evaluations/NRMSE.csv">변수별NRMSE</a> · <a href="summary.csv">집계표</a> ·
<a href="common_completed_NRMSE.csv">공통완주NRMSE</a> · <a href="../evaluations/reconstruction_given_truth.csv">정답대표변수에서의복원진단</a></p>
<p>정답대표변수복원진단은CNET의한계를분리하기위한별도검사이고, 주되먹임평가에정답상태를넣은것이아니다.</p>
<img src="coverage.png"><img src="loss_and_lr.png"><img src="reconstruction_loss.png">
<p>원저자코드·가중치·정확한20반응파일의복제가아니다. 공개Burke10종25Cantera항목을사용했고,
ParamNet세출력곱융합은여전히독립재구현가정이다. 작은LR·K점진전환·고정간격·초기집중추출은선언한안정화변경이며원문그대로라고부르지않는다.
문헌의정규화MAE와물리NRMSE는다르다. 소프트웨어/해시검증통과를화학성능재현완료와혼동하지않는다.</p>
<h2>실제 학습량과 종료 이유</h2>TRAINING
<p><a href="../final_training.json">Operator 학습 기록</a> · <a href="../reconstruction_controls.json">복원 ANN 학습 기록</a> ·
<a href="../selection_frozen.json">검증 선택과 조합 비교</a> · <a href="../verification.json">NRMSE 재계산 검증</a></p></body></html>'''
    training=json.loads((root/'final_training.json').read_text()) if (root/'final_training.json').exists() else {}
    recon=json.loads((root/'reconstruction_controls.json').read_text())
    training_table='<table><tr><th>후보</th><th>추가 update</th><th>선택 update</th><th>선택 K</th><th>활성 초</th><th>마지막 LR</th><th>종료 이유</th></tr>'
    for name,r in {**training,**recon}.items():
        training_table+='<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in [name,r['last_new_update'],r['selected_new_update'],r.get('selected_K','해당 없음'),round(r['active_seconds'],2),r['last_lr'],r['stop_reason']])+'</tr>'
    training_table+='</table>'
    html=html.replace('CHOICE',escape(choice['operator']+' + '+choice['reconstruction'])).replace('TABLE',table).replace('TRAINING',training_table)
    (root/'report/REPORT.html').write_text(html)
    write(root/'report/complete.json',dict(status='passed',PDF_pages=pages,output_sha256={p.name:sha(p) for p in (root/'report').iterdir() if p.is_file() and p.name!='complete.json'}))

def verify(root):
    truth=np.load(root/'evaluations/reference.npz')['states'];variables=['T','H2','H','O','O2','OH','H2O','HO2','H2O2','AR','N2']
    metrics=list(csv.DictReader((root/'evaluations/NRMSE.csv').open()));count=0;undefined=0;max_difference=0.
    by_model=defaultdict(list)
    for row in metrics:by_model[(row['model'],int(round(float(row['step_us'])*10)))].append(row)
    for (name,stride),rows in by_model.items():
        with np.load(root/'evaluations'/f'{name}_{stride}.npz') as f:pred=f['prediction'];complete=f['raw_boundary_valid']
        for row in rows:
            i=int(row['IC']);j=variables.index(row['variable']);ix=np.arange(10001)[::stride if row['grid']=='boundaries' else 1]
            if row['window']=='early_0p1ms':ix=ix[ix<=1000]
            ref=truth[i,ix,j];nn=pred[i,ix,j];den=float(np.sum(ref*ref))
            if not den:assert row['diagnostic_NRMSE']=='';undefined+=1;continue
            value=float(np.sqrt(np.sum((nn-ref)**2)/den));actual=float(row['diagnostic_NRMSE'])
            assert np.isclose(value,actual,rtol=1e-10,atol=1e-12)
            if not complete[i]:assert row['raw_NRMSE']==''
            max_difference=max(max_difference,abs(value-actual));count+=1
        for i in range(len(truth)):
            boundary=pred[i,::stride];good=bool(np.isfinite(boundary).all() and np.all((boundary[:,0]>=200)&(boundary[:,0]<=6000)) and boundary[:,1:].min()>=0)
            assert good==bool(complete[i])
    return dict(status='passed',NRMSE_recomputations=count,zero_RMS_undefined_rows=undefined,maximum_difference=max_difference,
        no_full_horizon_score_for_incomplete=True,selected_final_separate=True,no_repair=True,
        test_selection_frozen_sha256=sha(root/'selection_frozen.json'),report_complete_sha256=sha(root/'report/complete.json'))
