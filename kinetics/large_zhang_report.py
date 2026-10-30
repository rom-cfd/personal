"""Render a completed full-data raw GELU evaluation; never claim preflight accuracy."""
import argparse,hashlib,json,os,zipfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .preprocess_report import table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/zhang_5p3m_gelu/g0.1/evaluation_seeds11'))
    parser.add_argument('--report',type=Path,default=Path('ZHANG_5P3M_GELU_REPORT.md'))
    args=parser.parse_args();root=args.root
    manifest=json.loads((root/'manifest.json').read_text());assert manifest['status']=='completed'
    protocol=json.loads((root/'protocol.json').read_text());selection=json.loads((root/'selection_frozen.json').read_text())
    enthalpy=protocol.get('enthalpy_closure',False)
    summary=pd.read_csv(root/'summary.csv');detail=pd.read_csv(root/'trajectories.csv');one=pd.read_csv(root/'one_step.csv')
    with np.load('results/zhang_multiscale_raw/reference20ms.npz') as d:
        refs={int(i):a for i,a in zip(d['ids'],d['states'])};dt=float(d['dt'])
    with np.load('results/long10ms_data/trajectories.npz') as d:ids=np.flatnonzero(d['split']=='test')
    methods=list(summary.method.unique());colors={m:c for m,c in zip(methods,['tab:red','tab:blue','tab:orange'])}
    figures=root/'figures';figures.mkdir(exist_ok=True)
    paths=json.loads((root/'methods'/f'{methods[0]}.json').read_text())['paths']
    with zipfile.ZipFile(paths[0]+'.keras') as z:layers=json.loads(z.read('config.json'))['config']['layers']
    dense=[l['config'] for l in layers if l['class_name']=='Dense']
    architecture=[11]+[l['units'] for l in dense]
    traces={}
    for name in methods:
        with np.load(root/'traces'/f'{name}_test.npz') as d:traces[name]={int(i):d[f'trajectory_{int(i)}'] for i in ids}
    fig,axes=plt.subplots(3,6,figsize=(22,11))
    for i,ax in zip(ids,axes.flat):
        truth=refs[int(i)];ax.plot(np.arange(1,len(truth))*dt*1e3,truth[1:,0],'k--',lw=1.6,label='Cantera')
        notes=[]
        for name,color in colors.items():
            a=traces[name][int(i)];t=np.arange(1,len(a))*dt*1e3
            if len(a)>1:ax.plot(t,a[1:,0],color=color,lw=.9,label=name);ax.plot(t[-1],a[-1,0],marker='x',color=color,ms=3)
            else:ax.plot([],[],color=color,label=name)
            notes.append(f'{name}: {(len(a)-1)*dt*1e3:.3g} ms')
        ax.text(.02,.97,'\n'.join(notes),transform=ax.transAxes,va='top',fontsize=5.5)
        ax.set(xscale='log',xlim=(dt*1e3,20),ylim=(200,6000),xlabel='Time (ms)',ylabel='T (K)',title=f'IC {i}')
        ax.grid(alpha=.2)
    axes.flat[-1].legend(fontsize=6);fig.suptitle(f'GELU, temperature '+('from enthalpy' if enthalpy else 'from ANN')+f', {manifest["training_examples"]:,} pairs: actual end time annotated')
    fig.tight_layout();fig.savefig(figures/'temperature20ms.png',dpi=150);plt.close(fig)
    keys=['method','split','horizon']
    extra=detail.groupby(keys).agg(mean_coverage=('coverage','mean'),median_end_ms=('completed_steps',lambda s:float(s.median())*dt*1e3)).reset_index()
    summary=summary.merge(extra,on=keys,validate='one_to_one');summary.to_csv(root/'summary_with_coverage.csv',index=False)
    heading='# '+('PRE-FLIGHT ONLY: tiny fixture' if manifest['preflight'] else '530만 쌍 GELU NN: '+('엔탈피 기반 온도 복원' if enthalpy else 'physics OFF'))
    text=[heading,'']
    selected=[r.get('selected_epoch') for r in manifest['model_records'] if 'selected_epoch' in r]
    if selected:text += ['모든 epoch의 validation MAE로 고정한 가중치를 사용했다. 선택된 epoch: '+str(selected)+
        '. 아래 epoch 수는 전체 학습 예산이며 선택된 가중치의 시점과 다를 수 있다.','']
    if manifest['preflight']:text+=['**이것은 작은 검증용 데이터 결과이며, 530만 쌍 모델의 예측 성능이 아니다.**','']
    text += [f'학습 데이터 **{manifest["training_examples"]:,}쌍**, epoch **{manifest["epochs"]}회**, '
        f'모델 **{manifest["models"]}개**, gamma={protocol["gamma"]:g}. 구조는 '+str(architecture)+'.', '',
        '입력은 train Z-score(T, linear P, literal BCT(Y)), 출력은 별도 train Z-score의 Δ(T,P,BCT(Y))다. '+
        ('NN의 온도 출력을 버리고, 현재 엔탈피와 갱신된 조성으로 다음 온도를 복원한다. '
         '열역학 물성 계산에만 Cantera의 정규화 조성 정의를 사용하고, 저장·self-feed하는 ANN 조성은 변경하지 않는다. '
         '원소 보정·decoder clipping·출력 질량합 재정규화는 없다. 온도 해가 200–6000 K에 없으면 실패로 기록한다.'
         if enthalpy else 'NN의 온도·압력도 그대로 self-feed한다. 원소 보정·엔탈피 복원·decoder clipping·질량합 재정규화는 없다.'), '',
        '## 최종 해석 시간: 10 ms / 20 ms', '',
        '수치 완주와 물리 허용 완주를 구분한다. 도중 실패한 궤적의 full-profile 오차는 NaN으로 제외하며 '
        '실제 도달 시간과 coverage를 함께 표시한다. 숫자가 유한하다는 사실만으로 안정적인 kinetics 예측이라고 판단하지 않는다.', '']
    for horizon in [.01,.02]:
        f=summary[(summary.split=='test')&np.isclose(summary.horizon,horizon)].copy()
        f['coverage_%']=100*f.mean_coverage
        text += [f'### Test {horizon*1000:g} ms','',table(f[['method','completed','physical_completed','coverage_%','median_end_ms',
            'ignition_classification','ignition_pairs','temperature_profile_mean','species_profile_mean']]),'']
    text += ['## 단일 모델 선택','',f'Validation one-step 90,000쌍으로 seed {selection["seed"]}를 고정했다. '
        f'선택 후보 풀은 {selection["selection_pool"]}개이며 usable validation model={selection["usable_validation_model"]}이다. '
        'Test/rollout 결과로 선택을 바꾸지 않았다. 후보가 하나면 다른 모델 중 최상위를 골랐다는 뜻이 아니다.', '',
        table(pd.read_csv(root/'validation_ranking.csv')),'','## One-step test','',table(one[one.split=='test']),'',
        '## 후기 구간과 실패','',
        table(detail[(detail.split=='test')&np.isclose(detail.horizon,.02)].groupby('method').agg(
            early_count=('early_0_2ms_complete','sum'),early_T_RMSE=('early_0_2ms_T_RMSE','mean'),
            late_count=('late_2_10ms_complete','sum'),late_T_RMSE=('late_2_10ms_T_RMSE','mean'),
            tail_count=('tail_10_20ms_complete','sum'),tail_T_RMSE=('tail_10_20ms_T_RMSE','mean')).reset_index()),'',
        '각 구간 전체를 끝낸 경우에만 RMSE를 정의한다. 구간에 도달하지 못한 결과를 작은 오차로 해석하지 않는다.', '',
        table(detail[(detail.split=='test')&np.isclose(detail.horizon,.02)].groupby(['method','status']).size().rename('ICs').reset_index()),'',
        '## 그래프','',f'![18 temperature profiles]({figures}/temperature20ms.png)','',
        '검정 점선은 Cantera, 나머지는 NN 방법별 결과다. 각 패널은 실제 숫자로 전진한 마지막 시간을 표시한다. '
        't=0은 로그축 표시에서만 제외했고, y축 200–6000 K 제한은 원자료를 clipping한 것이 아니다.', '',
        '## 범위와 한계','',
        '데이터는 Zhang §2.5의 독립 log sampling 규칙을 사용한다. 다만 기존 비교를 위한 9종/28반응, dt=2 μs, '
        'P=.5–5 atm 확장과 명시적인 N2-preserving 정규화 가정은 유지했다. 전체 표본 수를 논문 규모로 맞춘 것과 '
        '논문의 모든 구현/학습 조건을 재현했다는 주장은 다르다. GELU 1600/800/400은 저장소의 기존 논문 모델 설정을 따른다.', '',
        '이전 8,834쌍 pilot과는 데이터 규모뿐 아니라 네트워크 크기 및 학습량도 달라졌다. '
        '그 비교로 데이터 크기만의 인과 효과를 주장하지 않는다. '+
        ('이번 실험은 온도만 엔탈피로 복원하며, Wang식 원소 보존 projection은 사용하지 않는다.' if enthalpy else
         'Physics-off는 사용자 요청이며, 원문에서 온도를 엔탈피로 복원하는 부분과 의도적으로 다르다.'), '',
        f'- [원시 IC별 결과]({root}/trajectories.csv)',f'- [입출력/선택 프로토콜]({root}/protocol.json)',
        f'- [모델/epoch/검증 기록]({root}/manifest.json)','']
    args.report.write_text('\n'.join(text)+'\n')
    (root/'report_manifest.json').write_text(json.dumps(dict(renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evaluation_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()),indent=2))
    print('WROTE',args.report)


if __name__=='__main__':main()
