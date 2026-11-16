"""Reports and independent arithmetic checks for actual feedback boundaries."""
from pathlib import Path
import csv
import json
import time
import hashlib
from html import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def report(root, cases, truth, predictions, variables, summary, deadline):
    target=root/'report'
    colors={'selected':'#c2342e','protocol_final':'#2869a0','stabilized':'#8c6995'}
    labels={'selected':'validation-selected pretrain','protocol_final':'original K5 protocol final',
            'stabilized':'LR/100 K5 selected'}
    pages={}
    for stride,zoom,filename in [(200,False,'all_profiles_20us.pdf'),(200,True,'ignition_zoom_20us.pdf'),
                                  (20,False,'all_profiles_2us.pdf')]:
        with PdfPages(target/filename) as pdf:
            for i,case in enumerate(cases):
                if time.perf_counter()>deadline: raise TimeoutError('Report budget exhausted')
                time_us=np.arange(truth.shape[1])*0.1
                if zoom:
                    peak=int(np.argmax(np.gradient(truth[i,:,0])))
                    lo=max(0,peak-200);hi=min(len(time_us),peak+401)
                else:lo,hi=0,len(time_us)
                fig,axes=plt.subplots(3,4,figsize=(19,12),layout='constrained')
                for j,ax in enumerate(axes.flat):
                    if j>=len(variables):ax.axis('off');continue
                    ax.plot(time_us[lo:hi],truth[i,lo:hi,j],color='black',lw=1.5,label='Cantera')
                    for model in ['selected','protocol_final','stabilized']:
                        pred,_=predictions[(model,stride)]
                        ax.plot(time_us[lo:hi],pred[i,lo:hi,j],color=colors[model],lw=1,ls='--',label=labels[model])
                    # These markers are the values actually forwarded to the next step.
                    marks=np.arange(0,len(time_us),stride)
                    marks=marks[(marks>=lo)&(marks<hi)]
                    ax.plot(time_us[marks],predictions[('selected',stride)][0][i,marks,j],
                        '.',color=colors['selected'],ms=3)
                    ax.set(title=variables[j]+(' [K]' if j==0 else ' mass fraction'),xlabel='time [microseconds]')
                    ax.grid(alpha=.2)
                handles,labs=axes.flat[0].get_legend_handles_labels()
                fig.legend(handles,labs,loc='lower right',fontsize=9)
                fig.suptitle(f"{case['split']} IC{case['IC']} | T0={case['T0']:.4f}K phi={case['phi']:.6f} | "
                    f"feedback every {stride/10:g}us; dots=executed endpoints; lines=dense diagnostics\n"
                    'No repair. Negative diagnostic curves retained; invalid raw boundaries are failures.',fontsize=13)
                pdf.savefig(fig);plt.close(fig)
        pages[filename]=len(cases)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,step in zip(axes,[20,2]):
        for j,key in enumerate(['raw_complete','raw_joint9','raw_full_early_joint9']):
            vals=[sum(r[key] for r in summary if r['model']==m and r['step_us']==step)
                  for m in ['selected','protocol_final','stabilized']]
            ax.bar(np.arange(3)+(j-1)*.22,vals,.22,label=key)
        ax.set(xticks=np.arange(3),xticklabels=['pretrain selected','protocol final','LR/100 selected'],
               ylim=(0,len(cases)+2),title=f'{step}us actual feedback boundaries',ylabel='independent IC count')
        ax.legend(fontsize=8)
    fig.savefig(target/'coverage.png',dpi=150);plt.close(fig)

    selected=json.loads((root/'selection_frozen.json').read_text())
    rows=''.join('<tr>'+''.join(f'<td>{escape(str(r[k]))}</td>' for k in ['model','step_us','split','total',
        'raw_complete','raw_joint9','raw_full_early_joint9','raw_dense_curve_full_early_joint9','dense_physical_valid'])+'</tr>' for r in summary)
    html='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>React-DeepONet: 실제 예측 상태를 되먹이는 0D 재검증</title>
<style>body{max-width:1250px;margin:36px auto;padding:0 20px;font:17px/1.8 system-ui;color:#183044}h2{margin-top:2em}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #bdcad6;padding:6px;vertical-align:top}th{background:#eaf0f6}img{max-width:100%}pre{white-space:pre-wrap;background:#edf2f7;padding:15px}a{color:#075b98}.note{padding:12px 18px;background:#fff2e5;border-left:4px solid #ad6732}</style></head><body>
<h1>React-DeepONet: 예측 상태를 되먹이는 0D 화학 솔버 재검증</h1>
<p>대상은 Kumar–Echekki의 React-DeepONet(Fuel 365, 2024, 131212)과 Kumar 학위논문 3장의 수소/공기 예제다.
이 실험은 초기조건과 절대시간으로 전역 궤적을 질의하는 flexDeepONet과 다르다.
이전에 학습한 React 가중치를 그대로 사용하고 실제 시간 진행과 보조 곡선 질의를 분리해 재검증했다. 새 학습과 물리 보정은 없다.</p>
<h2>1. 실제로 계산하는 절차</h2>
<pre>현재 대표 상태 xₙ → React-DeepONet(xₙ, Δt, 당량비) → 예측 상태 xₙ₊₁
                                                        ↓
                                       다음 입력은 이 예측 상태
참조 Cantera 상태는 처음 초기값과 오차 계산에만 사용한다.</pre>
<p>대표 변수는 T/H₂/H/O₂/H₂O의 5개다. 나머지 6개 변수는 같은 시점의 대표 예측값에서 별도 복원 ANN으로 계산한다.
복원 ANN의 정답을 넣지 않는다. 대표 상태가 다음 시간창에 되먹이고, 참조 해는 되먹이지 않는다.</p>
<p>문헌의 실제 진행 간격인 <strong>20 μs × 50회 = 1 ms</strong>를 주 평가로 사용했다.
기존 ANN의 용도와 연결하기 위해 <strong>2 μs × 500회 = 1 ms</strong>도 별도 평가했다.
20 μs 결과를 2 μs 솔버의 성공이라고 부르지 않는다.</p>
<h2>2. 이전 판정에서 무엇이 섞였나</h2>
<p>이전 raw 평가는 한 창 안의 0.1 μs 간격 보조 질의까지 모두 검사해, 그중 하나라도 음수이면 진행을 중단했다.
실제 20 μs 솔버가 다음 입력으로 사용하는 것은 20 μs 끝점이다. 창 안의 보조 질의는 다음 입력에 사용되지 않는다.
그 음수도 유효한 상태는 아니지만, 실제로 실행한 경계값의 실패와는 구분해야 한다.</p>
<p class="note">이번에 모델 가중치를 개선한 것은 아니다. 같은 가중치에서 실제 시간 진행 경계와 보조 곡선 질의의 실패를 분리했다.
경계값이 음수이거나 비유한 값이면 raw 솔버는 그 지점에서 실패로 처리한다. 그 이후의 수학적 곡선은 진단으로만 보존하며 전체기간 raw 점수를 주지 않는다.</p>
<h2>3. 검증 선택과 새 독립 평가</h2>
<p>기존 검증 12조건에서 20 μs 경계의 전체 9변수 정확도+상태 유효성 통과 수, 전체와 초기 0.1 ms 동시 통과 수,
worst9 중앙값 순으로 모델을 골랐다. 모델 선택과 새 초기조건 목록을 동결한 뒤에만 새 참조 해와 모델 예측을 계산했다.</p>
<p>선택 가중치: <strong>SELECTED_MODEL</strong>. 원래 K5 학습의 마지막 가중치와 학습률을 1/100로 줄였던 K5 선택 가중치는
사전 지정 대조군으로 함께 평가했다. 모든 중간 loss를 평균하는 all_steps 개발군은 원문 끝점 loss와 달라 선택 후보에서 제외했다.</p>
<p>새 보간 16조건은 학습 범위 안, 새 외삽 16조건은 학습 온도/당량비 범위 밖이다. 기존 검증·테스트 초기조건과 겹치지 않는다.
이 테스트를 보고 다시 가중치나 선택 기준을 고치지 않았다. seed11 한 실행의 가중치이며 여러 seed 평균이 아니다.</p>
<h2>4. 실제 시간 진행의 결과</h2>
<p>완주는 1 ms까지 실제 경계 상태가 모두 유한하고, 온도 200–6000 K, 모든 질량분율 비음수인 경우다.
정확도는 T+8반응종 각각 NRMSE≤0.1일 때 joint9 통과다. N₂와 AR은 별도로 기록한다.
질량/원소/엔탈피의 정확한 보존을 강제하는 보정은 없으며 이 상태 검사만으로 완전한 보존을 주장하지 않는다.</p>
<p>NRMSE=RMSE(Cantera,NN)/RMS(Cantera), 백분율이 아닌 무차원 비율이다. 전체 1 ms와 초기 0.1 ms에서 각각 계산한다.
각 시각의 오차를 모아 계산한 구간 지표이며, 모든 개별 시각에서 오차가 0.1 이하라는 뜻은 아니다.</p>
<table><tr><th>가중치</th><th>진행 μs</th><th>평가 영역</th><th>조건 수</th><th>경계 raw 완주</th><th>경계 raw joint9</th><th>경계 전체+초기 joint9</th><th>raw 완주+조밀 곡선 전체/초기 joint9</th><th>보조 곡선 전 시각 상태 유효</th></tr>RESULT_ROWS</table>
<p>오른쪽 보조 곡선 상태 유효 열은 0.1 μs의 모든 질의를 검사한 결과다. 경계 raw 완주 수와 다르다.
주요 5변수 곡선 정확도와 전체 9변수 정확도, 실제 끝점과 조밀한 점화 곡선 정확도를 CSV에 모두 분리했다.</p>
<img src="coverage.png" alt="두 진행 간격별 완주와 정확도 통과 수">
<h2>5. 그래프와 원자료</h2>
<p>조건별 CSV에는 실제 경계의 대표 5변수 정규화 MAE와 최대 |ΣY−1|도 기록했다.
정규화 MAE는 학습의 5제곱근/min-max 좌표에서의 평균 절대 차이이며, 원저자의 MAE 집계 정의가 완전히 확인되지 않아 동일 수치 재현으로 주장하지 않는다.
AR처럼 Cantera RMS가 0인 변수의 NRMSE는 정의되지 않아 빈칸으로 남겼다.</p>
<p><a href="all_profiles_20us.pdf">문헌 20 μs 전체 그래프</a> · <a href="ignition_zoom_20us.pdf">20 μs 점화 확대</a> ·
<a href="all_profiles_2us.pdf">2 μs 전체 그래프</a> — 각 PDF는 새 32조건을 모두 포함한다.
검정은 Cantera, 빨강은 검증 선택 모델, 파랑은 원래 K5 최종 모델, 보라는 LR/100 안정화 모델이다.
빨간 점은 실제 되먹인 경계이고 선은 추가 조밀 질의다. 점화 확대는 각 조건의 Cantera 점화 시각을 중심으로 잡은 가변창이다.</p>
<p><a href="summary.csv">집계표</a> · <a href="../evaluations/cases.csv">조건별 결과</a> ·
<a href="../evaluations/NRMSE.csv">변수별 NRMSE/물리단위 RMSE</a> · <a href="../selection_frozen.json">동결 선택 근거</a> ·
<a href="../prior_validation_audit.json">기존 검증 12조건의 진단</a> · <a href="../verification.json">독립 재계산 검증</a></p>
<h2>6. 학습·재현 범위의 한계</h2>
<p>이번 추가 학습 업데이트는 0회다. 이전 branch5-50-50-50-200, trunk1-50-50-50-200, Shift5-10-10-1,
ParamNet1-50-200의 42,236파라미터 모델과 train-only 5제곱근/min-max 변환을 그대로 사용했다.
선택 pretrain은 72,500 업데이트, 원래 최종 모델은 118,712 업데이트(사전학습 100,000+K5 18,712)다.
원래 최종 학습은 시간상한으로 끝났으며 평탄 종료나 목표 120,000회 완료가 아니다.
안정화 대조군은 100,000회 사전학습에서 LR/100 K5 500회 후의 검증 선택 가중치다.</p>
<p>원저자의 가중치와 비공개 융합 코드를 확보하지 못했다. ParamNet의 세 출력 곱 융합은 독립 재구현 가정이며 확정된 저자 코드가 아니다.
공개 Burke 축소본은 10종·25 Cantera 반응 항목으로, 문헌의 20반응 파일 자체와 동일하다고 주장할 수 없다.
문헌의 정규화 MAE와 물리단위 NRMSE도 다른 지표다. 실제 경계가 맞는 것과 보조 곡선의 모든 시각이 유효한 것은 별도다.</p>
<p><a href="https://doi.org/10.1016/j.fuel.2024.131212">Kumar–Echekki, Fuel 2024</a> ·
<a href="https://arxiv.org/html/2304.12188v1#S2.SS2">공개 초기 원문 §2.2의 되먹임 절차</a> ·
<a href="https://repository.lib.ncsu.edu/bitstreams/259ce066-870d-4034-8ec7-f140115facfd/download">Kumar 학위논문 3장</a> ·
<a href="../../flex_deeponet_reproduction/summary_001/REPORT.html">별도로 보존한 전역 flexDeepONet 결과</a></p>
</body></html>'''
    html=html.replace('SELECTED_MODEL',escape(selected['selected_source'])).replace('RESULT_ROWS',rows)
    (target/'REPORT.html').write_text(html)
    complete=dict(status='passed',PDF_pages=pages,
        output_sha256={p.name:sha(p) for p in target.iterdir() if p.is_file() and p.name!='complete.json'})
    (target/'complete.json').write_text(json.dumps(complete,indent=2)+'\n')

def verify(root,truth,predictions,variables,metrics):
    # Independent norm/sum formula, deliberately not the score() implementation.
    maximum_difference=0.;count=0;undefined=0
    for row in metrics:
        stride=int(round(row['step_us']*10));i=int(row['IC']);j=variables.index(row['variable'])
        index=np.arange(0,len(truth[i]),stride if row['grid']=='executed_boundaries' else 1)
        if row['window']=='early_0p1ms': index=index[index<=1000]
        ref=truth[i,index,j];pred=predictions[(row['model'],stride)][0][i,index,j]
        denominator=float(np.sum(ref*ref))
        if denominator==0:
            assert row['diagnostic_NRMSE']==''
            undefined+=1;continue
        ratio=float(np.sqrt(np.sum((pred-ref)**2)/denominator))
        recorded=float(row['diagnostic_NRMSE']);difference=abs(ratio-recorded)
        assert np.isclose(ratio,recorded,rtol=1e-10,atol=1e-12),(row,ratio)
        if not row['raw_complete']: assert row['raw_NRMSE']==''
        maximum_difference=max(maximum_difference,difference);count+=1
    with (root/'evaluations/cases.csv').open() as f: cases=list(csv.DictReader(f))
    for row in cases:
        stride=int(round(float(row['step_us'])*10));i=int(row['IC'])
        boundary=predictions[(row['model'],stride)][0][i,::stride]
        good=bool(np.isfinite(boundary).all() and np.all((boundary[:,0]>=200)&(boundary[:,0]<=6000))
                  and boundary[:,1:].min()>=0)
        assert good==(row['raw_complete']=='True')
        if not good: assert row['raw_joint9']=='False' and row['raw_full_early_joint9']=='False'
    return dict(status='passed',NRMSE_recomputations=count,undefined_zero_RMS_rows=undefined,
        maximum_NRMSE_difference=maximum_difference,actual_feedback_semantics_tested=True,
        no_full_horizon_raw_score_for_incomplete=True,
        no_training=True,no_repair=True,seed=11,
        selection_sha256=sha(root/'selection_frozen.json'),cases_sha256=sha(root/'cases_frozen.json'))
