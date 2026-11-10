"""Full test reporting; external tests are opened only after frozen selection."""
from pathlib import Path
import csv
import json
import os
import time
import numpy as np
import torch
from kinetics.flex_data import VARIABLES, SPECIES, PAPER_RMSE, atomic, sha, stamp, trajectory
from kinetics.flex_model import FlexDeepONet


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def scores(truth,prediction):
    rmse=np.sqrt(np.mean((prediction-truth)**2,axis=0))
    rms=np.sqrt(np.mean(truth**2,axis=0))
    nrmse=np.divide(rmse,rms,out=np.full_like(rmse,np.inf),where=rms>0)
    nrmse[(rms==0)&(rmse==0)]=0
    return rmse,nrmse


def predict(model,initial,times):
    out=[]
    with torch.no_grad():
        for start in range(0,len(times),2048):
            now=times[start:start+2048]
            x=torch.tensor(np.repeat(initial[None,:],len(now),axis=0),dtype=torch.float64)
            t=torch.tensor(np.maximum(now,1e-32),dtype=torch.float64)
            out.append(model.physical(x,t).numpy())
    return np.concatenate(out)


def build_report(root,deadline):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl')
    os.environ.setdefault('XDG_CACHE_HOME','/tmp/kinetics-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm
    import cantera as ct
    root=Path(root);report=root/'report';report.mkdir()
    evaluation=root/'evaluations';evaluation.mkdir()
    assert (root/'selection_frozen.json').exists()
    models={}
    for label,name in [('selected','selected.pt'),('final','last.pt')]:
        model=FlexDeepONet();state=torch.load(root/'models'/name,weights_only=False,map_location='cpu')
        model.load_state_dict(state['model']);model.eval();models[label]=model
    rows=[];summaries=[];checks=0;references=[];all_predictions={k:[] for k in models};heat=[];heat_names=[]
    gas=ct.Solution(str(root/'gri30.yaml'))
    elements=['H','O','N']
    element_matrix=np.array([[gas.n_atoms(s,e)*gas.atomic_weight(e)/gas.molecular_weights[gas.species_index(s)]
                              for s in SPECIES] for e in elements])
    recompute_records=[]
    def profiles(pdf,times,truth,preds,title,log_time=False):
        fig,axes=plt.subplots(4,5,figsize=(17,10),layout='constrained')
        for j,ax in enumerate(axes.flat):
            if j==19:ax.axis('off');continue
            scale=1 if log_time else 1e6
            mask=times>=1e-7 if log_time else np.ones(len(times),dtype=bool)
            ax.plot(times[mask]*scale,truth[mask,j],color='black',lw=1.6,label='Cantera')
            for label,color in [('selected','#c43c39'),('final','#2874a6')]:
                ax.plot(times[mask]*scale,preds[label][mask,j],color=color,lw=1.1,ls='--',label=label)
            if log_time:ax.set_xscale('log')
            ax.set_title(VARIABLES[j]+(' [K]' if j==0 else ' mass fraction'),fontsize=9)
            ax.set_xlabel('time [s]' if log_time else 'time [microseconds]',fontsize=8)
            ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4),useOffset=False)
            ax.tick_params(labelsize=7);ax.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower right',ncol=3,fontsize=9)
        fig.suptitle(title,fontsize=12);pdf.savefig(fig);plt.close(fig)
    with PdfPages(report/'all_profiles.pdf') as full_pdf, PdfPages(report/'ignition_zoom.pdf') as zoom_pdf:
        for split in ['paper_like_test','robustness_test']:
            data=np.load(root/'data'/f'{split}.npz')
            for index,(ic,initial,truth) in enumerate(zip(data['ics'],data['initial'],data['values'])):
                atomic(root/'status.json',dict(phase='evaluating',split=split,case=index+1,total=len(data['ics']),at=stamp()))
                times=data['times'];preds={label:predict(m,initial,times) for label,m in models.items()}
                idt=float(times[int(np.argmax(np.gradient(truth[:,0],times)))])
                lo=max(0.,.3*idt);hi=min(1.,max(3*idt,lo+1e-7))
                zoom_times=np.linspace(lo,hi,401)
                early_times=np.linspace(0,1e-4,501)
                dense_times=np.unique(np.r_[0.,early_times,zoom_times])
                _,dense_truth=trajectory(ic,dense_times,root/'gri30.yaml')
                dense_preds={label:predict(m,initial,dense_times) for label,m in models.items()}
                early_idx=np.searchsorted(dense_times,early_times);zoom_idx=np.searchsorted(dense_times,zoom_times)
                stored=dict(times=times,truth=truth,dense_times=dense_times,dense_truth=dense_truth,
                            initial=initial,ic=ic,variables=np.array(VARIABLES),early_indices=early_idx,zoom_indices=zoom_idx)
                for label in models:
                    prediction=preds[label]
                    finite=bool(np.isfinite(prediction).all() and np.isfinite(dense_preds[label]).all())
                    valid=bool(finite and np.min(prediction[:,1:])>=0 and np.min(dense_preds[label][:,1:])>=0
                               and min(prediction[:,0].min(),dense_preds[label][:,0].min())>=250
                               and max(prediction[:,0].max(),dense_preds[label][:,0].max())<=6000)
                    passes={}
                    for window,ref,pred in [('full_1s_log_grid',truth,prediction),
                        ('early_0p1ms_uniform',dense_truth[early_idx],dense_preds[label][early_idx]),
                        ('ignition_window',dense_truth[zoom_idx],dense_preds[label][zoom_idx])]:
                        rmse,nrmse=scores(ref,pred);checks+=19
                        passes[window]=bool(np.all(nrmse[:9]<=.1))
                        for j,var in enumerate(VARIABLES):
                            rows.append(dict(split=split,IC=index,model=label,window=window,variable=var,
                                RMSE=float(rmse[j]),RMS_Cantera=float(np.sqrt(np.mean(ref[:,j]**2))),
                                NRMSE=float(nrmse[j]),finite=finite,strict_physical_valid=valid))
                    worst=float(scores(truth,prediction)[1][:9].max())
                    if label=='selected':heat.append(scores(truth,prediction)[1]);heat_names.append(f'{split}:{index}')
                    total_drift=float(np.max(np.abs(prediction[:,1:].sum(1)-1)))
                    element_drift=float(np.max(np.abs(prediction[:,1:]@element_matrix.T-initial[1:]@element_matrix.T)))
                    summaries.append(dict(split=split,IC=index,phi=float(ic[0]),T0=float(ic[1]),model=label,
                        finite_complete=finite,strict_physical_valid=valid,
                        joint9_curve=passes['full_1s_log_grid'],
                        joint9_curve_full_and_early=passes['full_1s_log_grid'] and passes['early_0p1ms_uniform'],
                        joint9_curve_full_and_ignition=passes['full_1s_log_grid'] and passes['ignition_window'],
                        joint9_with_strict_physical_valid=passes['full_1s_log_grid'] and valid,
                        worst9_NRMSE=worst,minimum_mass_fraction=float(min(prediction[:,1:].min(),dense_preds[label][:,1:].min())),
                        maximum_sumY_error=total_drift,maximum_element_mass_error=element_drift,
                        Cantera_ignition_time_s=idt))
                    stored[label]=prediction;stored['dense_'+label]=dense_preds[label]
                destination=evaluation/f'{split}_IC{index:03d}.npz'
                np.savez_compressed(destination,**stored)
                # Reopen saved arrays and independently recompute every full/early/zoom score.
                reread=np.load(destination)
                for label in models:
                    for window,ref,pred in [('full_1s_log_grid',reread['truth'],reread[label]),
                        ('early_0p1ms_uniform',reread['dense_truth'][early_idx],reread['dense_'+label][early_idx]),
                        ('ignition_window',reread['dense_truth'][zoom_idx],reread['dense_'+label][zoom_idx])]:
                        norm=np.sqrt(np.mean((pred-ref)**2,axis=0))/np.sqrt(np.mean(ref**2,axis=0))
                        existing=[r for r in rows if r['split']==split and r['IC']==index and r['model']==label and r['window']==window]
                        assert np.allclose(norm,[r['NRMSE'] for r in existing],rtol=2e-12,atol=1e-14)
                title=f'{split} IC{index}; T0={ic[1]:.1f} K; phi={ic[0]:.5g}; direct operator, no feedback or correction'
                profiles(full_pdf,times,truth,preds,title,True)
                profiles(zoom_pdf,zoom_times,dense_truth[zoom_idx],{k:v[zoom_idx] for k,v in dense_preds.items()},title)
                if split=='paper_like_test':
                    references.append(truth)
                    for k,v in preds.items():all_predictions[k].append(v)
    write_csv(evaluation/'NRMSE.csv',rows);write_csv(evaluation/'case_summary.csv',summaries)
    reference=np.array(references)
    paper_rows=[]
    for label,predictions in all_predictions.items():
        rmse=np.sqrt(np.mean((np.array(predictions)-reference)**2,axis=(0,1)))
        for i,var in enumerate(VARIABLES):
            paper_rows.append(dict(model=label,variable=var,our_RMSE=float(rmse[i]),
                paper_TableS3_RMSE=PAPER_RMSE[i],ratio_to_paper=float(rmse[i]/PAPER_RMSE[i]),
                note='Independent10ICs; publishedICseed and authorweights unavailable. Same units, not identical samples.'))
    write_csv(report/'paper_RMSE_comparison.csv',paper_rows)
    aggregate=[]
    for split in ['paper_like_test','robustness_test']:
        for label in models:
            part=[r for r in summaries if r['split']==split and r['model']==label]
            aggregate.append(dict(split=split,model=label,total=len(part),
                **{field:sum(int(r[field]) for r in part) for field in ['finite_complete','strict_physical_valid',
                    'joint9_curve','joint9_curve_full_and_early','joint9_curve_full_and_ignition','joint9_with_strict_physical_valid']}))
    write_csv(report/'summary.csv',aggregate)
    fig,ax=plt.subplots(figsize=(13,9),layout='constrained')
    im=ax.imshow(np.maximum(np.array(heat),1e-8),aspect='auto',norm=LogNorm(vmin=1e-3,vmax=10),cmap='viridis')
    ax.set_xticks(np.arange(19),VARIABLES,rotation=70);ax.set_yticks(np.arange(len(heat_names)),heat_names,fontsize=6)
    fig.colorbar(im,ax=ax,label='NRMSE = RMSE / RMS(Cantera), ratio');fig.savefig(report/'coverage.png',dpi=160);plt.close(fig)
    with (root/'training.csv').open() as f:history=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    axes[0].semilogy([int(r['update']) for r in history],[float(r['train_MSE']) for r in history],label='train MSE')
    axes[0].semilogy([int(r['update']) for r in history],[float(r['validation_total_loss']) for r in history],label='validation total loss')
    axes[0].legend();axes[1].semilogy([int(r['update']) for r in history],[float(r['lr']) for r in history])
    for ax in axes:ax.set_xlabel('optimizer updates');ax.grid(alpha=.2)
    axes[1].set_ylabel('learning rate');fig.savefig(report/'loss_and_lr.png',dpi=180);plt.close(fig)
    text='''<!doctype html><meta charset="utf-8"><title>0D flexDeepONet reproduction</title>
<style>body{max-width:1100px;margin:30px auto;font:16px/1.7 sans-serif}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:7px}img{max-width:100%}</style>
<h1>0D 수소 점화 flexDeepONet 문헌 프로토콜 재현</h1>
<p>초기조건과 시간에서 전체 궤적을 직접 질의한다. 자기 예측 되먹임, 학습 중 물리 제약, 사후 물리 보정은 없다. 기존 React-DeepONet 20us 반복 예측과 다른 문제다.</p>
<p>원저자 설정의 GRI30·1atm·500조건·500시각·19변수·34038파라미터를 사용했다. 원저자 가중치와 원래 LHS 난수시드는 없어 동일 데이터/가중치의 복제라고 부를 수 없다. 시간상한은 평탄 판정이 아니다.</p>
<p>paper_like_test는 독립10조건, robustness_test는 별도32조건이다. 모든 조건의 그림을 제공한다. 선택은 원저자 방식의 시간점 검증 오차만으로 끝낸 뒤 테스트를 열었다.</p>
<p>joint9_curve는 온도+8반응종의 NRMSE가 모두0.1이하인 곡선 정확도다. 음수 등 물리 유효성은 별도 열이며 곡선 통과를 유효한 물리 상태와 혼동하지 않는다. NRMSE는 RMSE/RMS(Cantera)이며 백분율이 아니다. 전체1s는 문헌의 로그 시간격자, 초기0.1ms는 균일격자, 점화확대는 Cantera에 따른 가변창이다. TableS3와는 물리단위RMSE를 비교한다.</p>
<p><a href="all_profiles.pdf">전체변수 PDF</a> · <a href="ignition_zoom.pdf">점화확대 PDF</a> · <a href="../evaluations/NRMSE.csv">변수별 NRMSE</a> · <a href="paper_RMSE_comparison.csv">문헌RMSE 비교</a> · <a href="summary.csv">집계표</a></p>'''
    text+='<pre>'+json.dumps(aggregate,ensure_ascii=False,indent=2)+'</pre><img src="coverage.png"><img src="loss_and_lr.png">'
    (report/'REPORT.html').write_text(text)
    atomic(report/'verification.json',dict(status='passed',NRMSE_recomputations=checks,
        per_variable_ratio_not_percent=True,selected_and_final_separate=True,no_repair=True,
        all_external_ICs_plotted=True,selection_sha256=sha(root/'selection_frozen.json')))
    hashes={str(p.relative_to(report)):sha(p) for p in sorted(report.rglob('*')) if p.is_file()}
    atomic(report/'complete.json',dict(status='passed',at=stamp(),pages_per_pdf=len(heat),output_sha256=hashes))
