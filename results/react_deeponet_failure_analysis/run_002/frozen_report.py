"""Record evidence and fidelity gaps; never equate executable code with accuracy."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/react-diagnosis-mpl')
from pathlib import Path
import json,csv,shutil,html,argparse
import numpy as np
from kinetics.react_deeponet_experiment import write,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('results/react_deeponet_failure_analysis/run_002'));root=parser.parse_args().root
    if root.exists():raise FileExistsError('Use a new diagnosis root')
    old=Path('results/react_deeponet_reconstruction/run_001')
    diag=Path('results/react_deeponet_diagnostics/run_001');refine=Path('results/react_deeponet_refine_probe/run_001')
    capacity=Path('results/react_deeponet_single_ic/run_001');sampling=Path('results/react_deeponet_sampling_probe/run_001')
    for p in [old,diag,refine,capacity,sampling]:assert json.loads((p/'complete.json').read_text())['status']=='passed'
    root.mkdir(parents=True);shutil.copyfile(__file__,root/'frozen_report.py')
    read=lambda p:list(csv.DictReader(p.open()))
    optimizer_rows=read(diag/'summary.csv');refinement_rows=read(refine/'summary.csv');sampling_rows=read(sampling/'summary.csv')
    history=read(old/'training.csv');operator=[r for r in history if r['stage']=='operator']
    differences=[
      ['Method source','2023 early-preprint approximation','Detailed thesis chapter 3'],
      ['Hidden networks','Plain dense MLP','Modified MLP with two encoders and gates in every sub-network'],
      ['Condition parameter','No ParamNet','Separate equivalence-ratio ParamNet'],
      ['Optimizer / LR','Adam, fixed 0.01','Adamax, 0.01 times 0.9^(update/5000)'],
      ['Training objective','Single future-state MSE only','100k single-step updates, then 20k final-5-step endpoint MSE'],
      ['Chemistry','Cantera h2o2.yaml (GRI-derived)','Pinned public Burke release; verified H2/air reduction'],
      ['Train / horizon / window','25 IC / 0.5ms / 20us','99 IC / 1ms / 40us; inference in 20us windows'],
      ['Exact fidelity limits','Several missing details were assumptions','Private fusion and exact 20-reaction mechanism remain unavailable']]
    write(root/'analysis.json',dict(status='analysis_complete_not_reproduction_success',
      thesis='https://repository.lib.ncsu.edu/server/api/core/bitstreams/259ce066-870d-4034-8ec7-f140115facfd/content',
      differences=differences,optimizer_diagnostics=optimizer_rows,append_refinement_probe=refinement_rows,seen_IC_sampling_probe=sampling_rows,
      old_loss=dict(minimum=min(float(r['loss']) for r in operator),maximum=max(float(r['loss']) for r in operator),last=float(operator[-1]['loss'])),
      observations=[
       'Original fixed-lr Adam run is unstable; 200k updates are not evidence of convergence.',
       'In matched short diagnostics, lr 0.001 raised strict validation completion from 2/12 to 12/12; both had zero joint5 passes.',
       'Adamax and modified-MLP gating can reduce fixed-window MSE without solving self-fed trajectory accuracy.',
       'Appending K5 endpoint refinement to an already poor short model did not yield a joint5 pass.',
       'The one-IC capacity and sampling probes use a seen trajectory. They cannot demonstrate generalization.',
       'Broad reconstruction changes are source-fidelity corrections, not controlled proof that any single omission caused all failure.'
      ],limits=[
       'Author exact JAX code/weights for this method are unavailable.',
       'The pinned Burke variant has 10 species and 25 Cantera reaction entries, while the thesis says 20 reactions.',
       'Three-way coefficient fusion and log(0) floor are declared reconstruction choices.',
       'A tiny negative species is reported as nonphysical under strict raw criteria; unconstrained curve agreement is a separate diagnostic.',
       'All future held-out evaluation must occur after freezing validation selection. No soft constraint or training repair is used.'
      ]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    axes[0].plot([int(r['update']) for r in operator],[float(r['loss']) for r in operator],lw=.8);axes[0].set_yscale('log');axes[0].set_xlabel('Optimizer updates');axes[0].set_ylabel('Normalized training MSE');axes[0].set_title('Failed first reconstruction: fixed Adam LR=0.01')
    labels=['Adam .01','Adam .001','Adamax .01','Adamax .001','Gated .01','Gated .001'];axes[1].bar(np.arange(6),[int(r['complete']) for r in optimizer_rows],color='#2563eb')
    axes[1].set_xticks(np.arange(6),labels,rotation=25,ha='right');axes[1].set_ylim(0,13);axes[1].set_ylabel('Strict validation completion / 12');axes[1].set_title('All six controls: 0/12 joint5 accuracy passes')
    fig.tight_layout();fig.savefig(root/'instability_and_controls.png',dpi=160);fig.savefig(root/'instability_and_controls.pdf');plt.close(fig)
    for arm in ['uniform','initial_transient_focused']:
      a=np.load(sampling/arm/'traces.npz');truth,pred=a['reference'],a['prediction'];fig,axes=plt.subplots(2,5,figsize=(16,6))
      for j,name in enumerate(['T','H2','H','O2','H2O']):
       for k,limit in enumerate([1,.1]):
        ax=axes[k,j];ax.plot(np.arange(len(truth))*1e-4,truth[:,j],c='black',label='Cantera');ax.plot(np.arange(len(pred))*1e-4,pred[:,j],c='#2563eb',ls='--',label='NN');ax.set_xlim(0,limit);ax.set_title(name);ax.set_xlabel('Time [ms]')
        if j==0:ax.legend(fontsize=8)
      fig.suptitle('Seen-trajectory learning diagnostic ONLY: '+arm);fig.tight_layout();fig.savefig(root/(arm+'.png'),dpi=140);plt.close(fig)
    def table(headers,rows):
      return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    content='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>DeepONet 실패 원인 분석</title><style>body{max-width:1150px;margin:40px auto;padding:0 20px;font:16px/1.8 system-ui;color:#172033}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border:1px solid #ccd5e1}img{max-width:100%}th{background:#edf2f9}</style>
<h1>DeepONet 실패 분석과 재구현 근거</h1>
<p><b>기존 결과는 문헌 성능 재현에 실패했다.</b> 끝까지 계산된 것과 Cantera 곡선을 정확히 맞춘 것은 다르다. 첫 모델의 전체 31개 미지 조건에서 raw joint9 정확도 통과는 0개였다. ANN 쪽은 저자 공개 가중치의 추론을 재현한 것이고, DeepONet은 독자 구현을 새로 학습한 것이므로 같은 수준의 재현 검증으로 비교할 수 없다.</p>
<p>기존 넓은 조건 실험은 10 ms 동안 2 μs씩 5,000번 자기 예측을 되먹였다. 상세 문헌은 더 좁은 조건에서 1 ms 동안 20 μs씩 50번 진행한다. 문헌은 대표 5변수의 5제곱근·min/max 정규화 곡선과 MAE를 주로 보여주지만, 우리의 전체 성공 기준은 원래 단위의 T＋8반응종 NRMSE다. 따라서 같은 난도와 같은 성공 기준의 비교는 아니다. 다만 실패한 첫 문헌 근사는 이미 0.5 ms로 줄인 실험이고 대표 5변수도 0/31 통과였으므로, 이 차이만으로 기존 실패를 설명할 수는 없다.</p>
<h2>무엇이 달랐는가</h2><p>이전 2023 초고 근사 대신 저자 <a href="https://repository.lib.ncsu.edu/server/api/core/bitstreams/259ce066-870d-4034-8ec7-f140115facfd/content">학위논문 3장</a>의 상세 방법을 다시 확인했다. 아래의 차이를 발견했다. 여러 항목을 함께 바꾸는 재구현만으로 특정 하나를 단독 원인이라고 단정하지 않는다.</p>'''+table(['항목','이전 구현','새 상세 구현'],differences)+'''
<h2>실제로 확인한 문제</h2><p>고정 학습률 0.01에서 loss가 크게 튀었다. 동일 초기화·데이터·배치 추출·6,000회 업데이트로 비교하면 Adam의 학습률을 0.001로 낮췄을 때 검증 완주는 2/12에서 12/12로 개선됐으나, 대표 5변수 정확도 통과는 둘 다 0/12였다. 따라서 학습 불안정은 실제 문제지만, 이것만 고치면 해결된다는 증거는 없다.</p>
<img src="instability_and_controls.png" alt="이전 loss 불안정과 통제 실험">
<p>Adamax와 수정 MLP 조합은 한 번의 예측 MSE를 더 낮췄지만, 자기 예측을 되먹이는 궤적 정확도는 여전히 실패했다. 이미 품질이 낮은 단기 가중치에 K5 학습만 덧붙이는 별도 실험에서도 통과가 없었다. 작은 단일 예측 오차가 장기 누적 오차를 보장하지 않는다는 점이 확인됐다. 서로 완주한 조건이 다르므로 각 방식의 중간값을 같은 조건인 것처럼 비교하지 않았다.</p>
<p><b>물리 보정이 반응속도를 정답으로 만들어 주지는 않는다.</b> 원소량·양수성·엔탈피를 맞추더라도 반응이 진행되는 속도와 점화 시각은 틀릴 수 있다. 첫 모델은 나머지 화학종을 복원하는 ANN을 거치기 전, 대표 5변수 예측 자체부터 실패했다. 따라서 복원 ANN이나 보정기만의 문제라고 볼 수 없다. 이미 작은 예측 오차를 가진 모델에 보정이 유용했던 문헌 결과를, 기본 반응 궤적부터 틀린 모델에 같은 효과가 난다는 뜻으로 해석하면 안 된다.</p>
<h2>초기·점화 구간 비중 진단</h2><p>아래는 같은 궤적을 이미 본 상태에서 시작점 추출 비중만 바꾼 통제 실험이다. 초기 가중치와 Adam 상태, 이후 업데이트 수, 배치 크기, 학습률 규칙을 같게 유지했다. 이는 학습 가능성 진단이며, 미지 조건에 대한 성능이 아니다. 사진 한 장이나 낮은 loss만으로 재현 성공이라고 하지 않는다.</p>'''+table(['조건','선택 update','수치 완주','곡선 joint5','상태 유효','최악 NRMSE'],[[r[k] for k in ['arm','selected_update','numerically_complete','joint5_curve','strict_state_valid','worst_NRMSE']] for r in sampling_rows])+'''
<img src="uniform.png" alt="균일 시작점, 학습 중 본 조건"><img src="initial_transient_focused.png" alt="초기·점화 시작점 강조, 학습 중 본 조건">
<h2>재구현에서 지킬 기준</h2><p>문헌의 수정 MLP·ParamNet·Adamax 감소 규칙·100k 사전학습＋20k 후속 K5 학습·99개 학습 조건을 구현한다. 후속 학습은 5번째 끝점의 MSE이며, 모든 중간 오차 평균을 쓰는 예전 실험과 다르다. 훈련 중 정답으로 중간 상태를 바꾸거나 물리 보정을 넣지 않는다. 대표 변수 자체의 예측 오차와 나머지 화학종 복원 ANN 오차를 별도로 측정한다.</p>
<p>미지 조건은 모델 선택 후 평가한다. NRMSE는 RMSE(Cantera,NN)/RMS(Cantera)인 무차원 비율이다. 전체 기간과 초기 0.1 ms를 모두 본다. 음수가 있어도 계속 그린 곡선 진단, 잘못된 상태에서 중단하는 raw, clipping＋원소 투영＋엔탈피 보정 결과를 구분한다. 물리적으로 유효하지 않은 곡선 진단을 raw 성공으로 세지 않는다.</p>
<p>남은 재현 한계: 공개 Burke 원본의 검증된 H2/air 축소본은 10종·25개 Cantera rate entry다. 저자가 사용한 20반응 파일 자체는 확보하지 못했다. 세 망의 정확한 비공개 융합 코드와 log(0) 처리도 없으므로 가정은 config에 명시했다. 새 모델의 검증·테스트가 끝나기 전에는 문헌 성능을 재현했다고 말할 수 없다.</p></html>'''
    (root/'REPORT.html').write_text(content)
    sources=[p/'complete.json' for p in [old,diag,refine,capacity,sampling]]
    write(root/'complete.json',dict(status='passed',meaning='diagnosis report verified, not successful model reproduction',input_sha256={str(p):sha(p) for p in sources},output_sha256={str(p.relative_to(root)):sha(p) for p in root.iterdir() if p.is_file()}))
    print(root/'REPORT.html')


if __name__=='__main__':main()
