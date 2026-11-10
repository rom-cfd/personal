"""Report every IC, keeping curve agreement separate from state admissibility."""
import csv,json,html,re
from pathlib import Path
import numpy as np
from .react_deeponet_experiment import sha,write,csv_write,FifthRootScale

MODES=['representative_unconstrained','representative_raw','raw_full','repaired_full']
SUMMARY_MODES=MODES+['representative_protocol_final_unconstrained','representative_protocol_final_raw']
LABELS=['Curve diagnostic\nmay be nonphysical','Representative raw\nstrict validity','Full state raw','Full state repaired']


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    root=Path(root);out=root/'report'
    if any(out.iterdir()):raise FileExistsError('Do not overwrite a report')
    (out/'figures').mkdir();meta=json.loads((root/'dataset.json').read_text());variables=meta['variables'];columns=[j for j,v in enumerate(variables) if v!='AR'];rep=[0,1,2,4,6]
    rows=list(csv.DictReader((root/'evaluations/rollouts.csv').open()));selection=json.loads((root/'selection_frozen.json').read_text());groups=[]
    for split in sorted({r['split'] for r in rows}):
        for mode in SUMMARY_MODES:
            group=[r for r in rows if r['split']==split and r['mode']==mode];complete=[r for r in group if r['complete']=='True'];values=[float(r['worst_NRMSE']) for r in complete]
            groups.append(dict(split=split,mode=mode,total=len(group),numerically_complete=len(complete),strict_state_valid=sum(r['physical_valid']=='True' for r in group),
                curve_accuracy_pass=sum(r['curve_accuracy_pass']=='True' for r in group),joint_accuracy_and_validity=sum(r['joint_pass']=='True' for r in group),
                median_worst_NRMSE=float(np.median(values)) if values else None,maximum_worst_NRMSE=max(values) if values else None))
    csv_write(out/'summary.csv',groups)
    plt.rcParams.update({'font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
    count=len({r['case_id'] for r in rows});prep=json.loads((root/'preprocessing.json').read_text());scale=FifthRootScale(prep['minimum'],prep['maximum']).subset(rep)
    with PdfPages(out/'all_profiles.pdf') as pdf,PdfPages(out/'ignition_zoom.pdf') as zoom_pdf,PdfPages(out/'paper_representatives.pdf') as paper_pdf:
        for split in sorted({r['split'] for r in rows}):
            for case_id in sorted({int(r['case_id']) for r in rows if r['split']==split}):
                case=next(r for r in rows if int(r['case_id'])==case_id);archive=np.load(root/'evaluations'/split/f'IC{case_id}.npz');ref=archive['reference']
                for zoom in [False,True]:
                    fig,axes=plt.subplots(4,len(columns),figsize=(25,8.5),squeeze=False)
                    for k,mode in enumerate(MODES):
                        pred=archive[mode];info=next(r for r in rows if int(r['case_id'])==case_id and r['mode']==mode)
                        for c,column in enumerate(columns):
                            ax=axes[k,c];ax.plot(np.arange(len(ref))*1e-4,ref[:,column],color='black',lw=1.2,label='Cantera')
                            if not mode.startswith('representative_') or column in rep:
                                idx=rep.index(column) if mode.startswith('representative_') else column
                                ax.plot(np.arange(len(pred))*1e-4,pred[:,idx],color='#2563eb',ls='--',lw=1,label='NN')
                            else:ax.text(.5,.5,'Reconstruction\nnot included',ha='center',transform=ax.transAxes,fontsize=6)
                            if k==0:ax.set_title(variables[column])
                            if k==3:ax.set_xlabel('Time [ms]')
                            if c==0:ax.set_ylabel(LABELS[k]+'\n'+('complete' if info['complete']=='True' else 'incomplete'))
                            ax.set_xlim(0,.1 if zoom else 1);ax.ticklabel_format(axis='y',useOffset=False,style='plain')
                            if column:ax.ticklabel_format(axis='y',useOffset=False,style='sci',scilimits=(-2,2))
                            if k==0 and c==0:ax.legend(fontsize=6)
                    fig.suptitle(f'IC{case_id}: T0={case["T0"]} K, phi={case["phi"]}, P=1 atm | {split} | 50 self-fed 20-us windows',fontsize=12)
                    fig.tight_layout(rect=(0,0,1,.96));(zoom_pdf if zoom else pdf).savefig(fig);fig.savefig(out/'figures'/f'IC{case_id}_{"early" if zoom else "full"}.png',dpi=110);plt.close(fig)
                fig,axes=plt.subplots(1,5,figsize=(14,3.2));pred=archive['representative_unconstrained'];last=archive['representative_protocol_final_unconstrained']
                norm=lambda a:(np.sign(a)*np.abs(a)**.2-scale.minimum)/scale.span
                a,b,c=norm(ref[:,rep]),norm(pred),norm(last)
                for j,ax in enumerate(axes):
                    ax.plot(np.arange(len(a))*1e-4,a[:,j],color='black',label='Cantera');ax.plot(np.arange(len(b))*1e-4,b[:,j],color='#2563eb',ls='--',label='Validation-selected');ax.plot(np.arange(len(c))*1e-4,c[:,j],color='#d97706',ls=':',label='Protocol final')
                    ax.set_title(variables[rep[j]]);ax.set_xlabel('Time [ms]');ax.set_ylabel('Fifth-root normalized [0,1]')
                    if j==0:ax.legend(fontsize=7)
                fig.suptitle(f'IC{case_id}, {split}: unconstrained curve diagnostic; physical validity recorded separately',fontsize=10);fig.tight_layout();paper_pdf.savefig(fig);plt.close(fig);archive.close()
    history=list(csv.DictReader((root/'training.csv').open()));fig,axes=plt.subplots(1,2,figsize=(11,4))
    for phase,color in [('pretrain','#2563eb'),('refine','#d97706')]:
        h=[r for r in history if r['phase']==phase]
        if h:axes[0].plot([int(r['global_update']) for r in h],[float(r['loss']) for r in h],color=color,label=phase)
    axes[0].set_yscale('log');axes[0].set_ylabel('MSE: K1 or final K5 endpoint');axes[0].legend();axes[1].plot([int(r['global_update']) for r in history],[float(r['lr']) for r in history]);axes[1].set_ylabel('Adamax learning rate')
    for ax in axes:ax.set_xlabel('Global optimizer updates')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'loss_and_lr.{ext}',dpi=150)
    plt.close(fig)
    # Korean document; tables expose all counts instead of selecting a good IC.
    table='<table><tr>'+''.join('<th>'+v+'</th>' for v in ['평가 집합','방식','수치 완주','곡선 정확도','상태 유효성＋정확도'])+'</tr>'
    for r in groups:
        cells=[r['split'],r['mode'],f"{r['numerically_complete']}/{r['total']}",f"{r['curve_accuracy_pass']}/{r['total']}",f"{r['joint_accuracy_and_validity']}/{r['total']}"]
        table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in cells)+'</tr>'
    table+='</table>'
    content='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>DeepONet 상세 재구현</title><style>body{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.8 system-ui;color:#172033}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border:1px solid #ccd5e1}th{background:#edf2f9}a{color:#1d4ed8}img{max-width:100%}</style>
<h1>DeepONet 실패 분석 후 상세 방법 재구현</h1>
<p>저자 <a href="https://repository.lib.ncsu.edu/server/api/core/bitstreams/259ce066-870d-4034-8ec7-f140115facfd/content">학위논문 3장</a>을 기준으로 수정 MLP, 당량비 ParamNet, Adamax와 지수 학습률 감소, 단일 예측 사전학습 및 최종 5-step 오차 후속 학습을 구현했다. soft constraint와 학습 중 물리 보정은 없다.</p>
<p>공개 Burke 배포판의 H2/air 동등 축소 기구를 사용했다. 10종·25개 Cantera rate entry이며, 저자가 기술한 20반응 파일 자체를 확보한 것은 아니다. 기존 GRI 기반 참조, 실패한 모델과 결과는 보존했다. 정확한 비공개 코드와 가중치의 복제라고 주장하지 않는다. 융합식과 log(0) 수치 처리 등 남은 가정은 config.json에 있다.</p>
<p>학습 99개와 독립 검증 12개 초기 조건으로만 선택했다. 저자 격자의 미지 조건 111개와 별도로 동결한 새 보간 조건 16개는 선택 이후 평가했다. train 행의 예제는 학습 중 본 조건이다. 모든 추론은 20 μs씩 50번 자기 예측을 되먹이며 1 ms까지 진행한다. 창 내부의 0.1 μs 질의는 추가 진행 스텝이 아니다.</p>
<p><b>곡선 정확도와 물리적으로 유효한 상태를 구분한다.</b> representative_unconstrained는 정규화된 출력을 그대로 되먹이는 논문식 곡선 진단이며 음수 농도가 있어도 수치 계산을 계속한다. clipping·보정·정답 주입은 없다. 음수 농도 진단을 보정 없는 유효한 성공으로 부르지 않는다. representative_raw와 raw_full은 잘못된 상태에서 중단한다. repaired_full은 clipping＋원소 투영＋상압 엔탈피 온도 보정이 있는 별도 시스템이다.</p>
<p>검증으로 선택한 모델과 학습 규칙을 끝까지 수행한 최종 가중치를 둘 다 보존·평가한다. protocol_final 두 행과 대표 변수 PDF의 주황 선은 마지막 학습 가중치다. 다른 네 행 및 전체 변수 PDF는 검증 선택 모델이다. 후속 학습이 나빠졌을 때 이전 체크포인트로 고른 결과만 보여주어 이를 숨기지 않는다. 시간상한으로 업데이트 목표를 채우지 못하면 최종 가중치도 불완료 프로토콜로 표시한다.</p>
<p>NRMSE=RMSE(Cantera,NN)/RMS(Cantera), 백분율이 아닌 비율이다. 대표 변수는 T/H2/H/O2/H2O의 joint5, 전체 상태는 T＋8반응종의 joint9를 판정하며 각각 모두 0.1 이하가 정확도 통과다. N2와 AR은 별도다. 불완주에는 전체 기간 점수가 없다. 표의 마지막 열은 곡선 정확도와 상태 유효성을 둘 다 만족한 수다.</p>'''+table+f'''
<p>총 학습 {selection['total_updates']} / {selection['target_updates']} update, 선택 {selection['selected_phase']}의 global update {selection['selected_global_update']}. 종료: {selection['stop_reason']}. 시간상한을 평탄이라고 부르지 않는다.</p>
<p><a href="all_profiles.pdf">모든 변수 그래프</a> · <a href="ignition_zoom.pdf">초기 0.1 ms 확대</a> · <a href="paper_representatives.pdf">논문식 정규화 대표 변수 곡선</a> · <a href="../evaluations/NRMSE.csv">변수별 NRMSE</a> · <a href="../evaluations/rollouts.csv">조건별 유효성·정확도·음수 발생</a> · <a href="summary.csv">집계 표</a></p>
<img src="loss_and_lr.png" alt="학습 단계별 loss와 학습률"><p>주된 비교는 전체 미지 조건의 결과다. 온도 또는 좋은 한 조건만으로 재현 성공을 주장하지 않는다.</p></html>'''
    (out/'REPORT.html').write_text(content)
    for name in ['all_profiles.pdf','ignition_zoom.pdf','paper_representatives.pdf']:
        assert len(re.findall(rb'/Type\s*/Page\b',(out/name).read_bytes()))==count
    write(out/'complete.json',dict(status='passed',pages_per_comparison_pdf=count,output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
