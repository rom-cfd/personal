"""Frozen matched/long endpoints, all observed losses, and full held-out profiles."""
import json
from pathlib import Path
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .ann_deeponet_study import check_deadline, verified_evaluation
from .p_only_training import sha256, write_json
from .data import SPECIES
from .deeponet_long_report import evaluation_summary, save, STYLE, common_completed

COLORS={'ann10':'tab:blue','phi_10k10':'tab:orange','ann_phi_10k20':'tab:green',
        'phi_long10':'tab:purple','ann_phi_long20':'tab:red'}
LABELS={'ann10':'ANN10','phi_10k10':'T/P/phi10 (10k)','ann_phi_10k20':'ANN+T/P/phi20 (10k)',
        'phi_long10':'T/P/phi10 (long)','ann_phi_long20':'ANN+T/P/phi20 (long)'}


def training_figures(root,destination,records,deadline):
    rows=[];full=[];mini=[]
    for r in records:
        state=json.loads(Path(r['checkpoint'],'state.json').read_text())
        rows.append({k:v for k,v in r.items() if k not in ('plateau_checks','timing')})
        for check in r['plateau_checks']:
            rows[-1]['plateau_check_count']=len(r['plateau_checks'])
        rows[-1].update(r['timing'])
        full.extend(dict(stage=r['stage'],**p) for p in state['full_train_history'])
        mini.extend(dict(stage=r['stage'],**p) for p in state['history'])
    table=pd.DataFrame(rows);history=pd.DataFrame(full);batch=pd.DataFrame(mini)
    table.to_csv(destination/'training_summary.csv',index=False)
    history.to_csv(destination/'observed_full_train_history.csv',index=False)
    batch.to_csv(destination/'observed_minibatch_history.csv',index=False)
    checks=[dict(seed=r['seed'],**c) for r in records if r['stage']=='long' for c in r['plateau_checks']]
    pd.DataFrame(checks).to_csv(destination/'plateau_checks.csv',index=False)
    fig,axes=plt.subplots(5,2,figsize=(13,15),squeeze=False)
    longs=table[table.stage=='long']
    for ax,r in zip(axes.flat,longs.itertuples()):
        check_deadline(deadline)
        h=history[(history.stage=='long')&(history.seed==r.seed)]
        ax.plot(h['update'],h.full_train_loss,lw=1,color='tab:purple')
        ax.axvline(10000,color='tab:orange',ls='--',lw=1,label='matched ANN budget')
        ax.set_yscale('log');ax.grid(True,which='both',ls=':',alpha=.25)
        ax.set_title(f'Seed {r.seed}: {r.updates:,} / {r.reason}')
        ax.set_xlabel('Optimizer update');ax.set_ylabel('Full-train residual MSE')
    fig.suptitle('T/P/initial-phi: all observed member loss curves, including long tails')
    fig.tight_layout(rect=(0,0,1,.98));save(fig,destination,'full_train_per_member')
    fig,axes=plt.subplots(3,1,figsize=(12,11),sharex=True)
    plot_rows=[]
    for ax,frame,key in [(axes[0],batch,'loss'),(axes[1],history,'full_train_loss')]:
        part=frame[frame.stage=='long'].pivot(index='seed',columns='update',values=key).dropna(axis=1)
        values=part.to_numpy();low=values.min(axis=0);high=values.max(axis=0)
        mean=np.exp(np.log(values).mean(axis=0))
        ax.fill_between(part.columns,low,high,alpha=.12,color='tab:purple')
        ax.plot(part.columns,mean,color='tab:purple',label='geometric mean, 10 members')
        ax.plot(part.columns,low,':',color='tab:purple',lw=.7)
        ax.plot(part.columns,high,'--',color='tab:purple',lw=.7)
        ax.set_yscale('log');ax.set_ylabel(key);ax.legend()
        plot_rows.extend(dict(metric=key,update=int(u),minimum=float(a),geometric_mean=float(b),maximum=float(c))
                         for u,a,b,c in zip(part.columns,low,mean,high))
    h=batch[(batch.stage=='long')&(batch.seed==longs.seed.iloc[0])]
    axes[2].step(h['update'],h.learning_rate,where='post',color='tab:purple')
    axes[2].set_ylabel('Learning rate');axes[2].set_xlabel('Optimizer update per member')
    for ax in axes:
        ax.axvline(10000,color='tab:orange',ls='--',lw=.8)
        ax.grid(True,which='both',ls=':',alpha=.25)
    fig.suptitle('Common observed updates only; min-max band is not a confidence interval')
    fig.tight_layout(rect=(0,0,1,.97));save(fig,destination,'training_loss_and_lr')
    pd.DataFrame(plot_rows).to_csv(destination/'training_plotted_statistics.csv',index=False)
    return table


def profile_figures(destination,entries,data,deadline):
    ids=np.flatnonzero(data['split']=='test');dt=float(data['dt'])
    for mode in ('raw','repaired'):
        loaded={}
        for directory,info in entries:
            if info['split']=='test' and info['mode']==mode:
                with np.load(directory/'traces.npz') as f:loaded[info['method']]={k:f[k] for k in f.files}
        if not loaded:continue
        for i in ids:
            check_deadline(deadline)
            fig,axes=plt.subplots(5,2,figsize=(13,15),squeeze=False)
            truth=data['states'][i];curves={'Cantera':truth}
            curves.update({name:value[f'trajectory_{int(i)}'] for name,value in loaded.items()})
            for component,(axis,label) in enumerate(zip(axes.flat,['T']+SPECIES)):
                for name,trace in curves.items():
                    axis.plot(np.arange(len(trace))*dt*1000,trace[:,component],
                        color='black' if name=='Cantera' else COLORS[name],
                        ls='--' if name=='Cantera' else '-',lw=1.,
                        label='Cantera' if name=='Cantera' else LABELS[name])
                axis.set_xlabel('Time [ms]');axis.set_ylabel('T [K]' if component==0 else f'Y {label}')
                axis.set_xlim(0,.01*1000);axis.minorticks_on();axis.grid(True,which='both',ls=':',alpha=.25)
            handles,labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.97))
            fig.suptitle(f'Test IC {int(i)} / {mode}: independent self-fed trajectories; failed lines end early')
            fig.tight_layout(rect=(0,0,1,.94));save(fig,destination,f'test_IC{int(i):02d}_{mode}_T_Y9')
            frames=[]
            for name,trace in curves.items():
                frame=pd.DataFrame(trace,columns=['T_K']+SPECIES)
                frame.insert(0,'time_s',np.arange(len(trace))*dt);frame.insert(0,'method',name)
                frames.append(frame)
            pd.concat(frames,ignore_index=True).to_csv(destination/f'test_IC{int(i):02d}_{mode}_profiles.csv',index=False)



def overview(destination,entries,data,deadline):
    loaded={}
    for directory,info in entries:
        if info['split']=='test' and info['mode']=='repaired':
            with np.load(directory/'traces.npz') as f:loaded[info['method']]={k:f[k] for k in f.files}
    ids=np.flatnonzero(data['split']=='test')
    fig,axes=plt.subplots(6,3,figsize=(16,21),squeeze=False)
    for ax,i in zip(axes.flat,ids):
        check_deadline(deadline)
        truth=data['states'][i]
        ax.plot(np.arange(len(truth))*2e-3,truth[:,0],color='black',ls='--',lw=1,label='Cantera')
        for name,traces in loaded.items():
            t=traces[f'trajectory_{int(i)}']
            ax.plot(np.arange(len(t))*2e-3,t[:,0],color=COLORS[name],lw=.9,label=LABELS[name])
        ax.set_title(f'IC {int(i)} | phi={data["initials"][i,2]:.3f}')
        ax.set_xlim(0,10);ax.set_xlabel('Time [ms]');ax.set_ylabel('T [K]')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.5,.98))
    fig.suptitle('Test: independent self-fed repaired trajectories, fixed initial phi',y=.995)
    fig.tight_layout(rect=(0,0,1,.955));save(fig,destination,'test_repaired_all18_temperature')


def report(root,cfg,snapshot,data,deadline):
    from .tp_phi_study import event
    started=time.perf_counter();check_deadline(deadline)
    destination=root/'reports'/f'report_{time.time_ns()}';destination.mkdir(parents=True,exist_ok=False)
    matched=json.loads((root/'matched_members.json').read_text())
    long=json.loads((root/'final_members.json').read_text())
    records=matched+long
    with plt.rc_context(STYLE):
        training=training_figures(root,destination,records,deadline)
        entries,summary=evaluation_summary(root,destination)
        profile_figures(destination,entries,data,deadline)
        overview(destination,entries,data,deadline)
    loss_markers=sorted((root/'loss_diagnostics').glob('*/complete.json'))
    if loss_markers:
        d=loss_markers[-1].parent;verified_evaluation(d)
        pd.read_csv(d/'residual_losses.csv').to_csv(destination/'final_train_validation_losses.csv',index=False)
    previous=json.loads((root/'original_results_sha256.json').read_text())
    for name,h in previous.items():
        if not Path(name).is_file() or sha256(name)!=h:raise ValueError(f'Preexisting artifact changed: {name}')
    # Independent recomputation of every recorded plateau check and endpoint integrity.
    from .deeponet_long_training import check_plateau,new_monitor
    checked=0
    for r in records:
        cp=Path(r['checkpoint']);marker=json.loads((cp/'complete.json').read_text())
        for name,h in marker['output_sha256'].items():
            if sha256(cp/name)!=h:raise ValueError('Checkpoint changed')
        state=json.loads((cp/'state.json').read_text())
        if state['update']!=r['updates'] or state['optimizer_iteration']!=r['updates']:
            raise ValueError('Endpoint iteration differs')
        monitor=new_monitor(cfg['plateau'])
        for expected in r['plateau_checks']:
            check_plateau(state['full_train_history'],expected['update'],monitor)
            if monitor['checks'][-1]!=expected:raise ValueError('Plateau replay mismatch')
            checked+=1
    plateau=sum(r['reason']=='plateau' for r in long)
    wall=(time.time()-json.loads((root/'budget.json').read_text())['started_unix'])
    lines=[
      'T/P/초기 phi trunk 실험을 종료했다. 아래 결과는 사전에 고정한 10k 및 장기 종료점의 비교다.',
      f'장기 모델 10개 중 평탄 도달 {plateau}개, 시간상한 종료 {10-plateau}개. 시간상한 종료는 수렴으로 표시하지 않았다.',
      f'장기 종료 update 범위 {min(r["updates"] for r in long):,}–{max(r["updates"] for r in long):,}.',
      f'실험 파이프라인 경과 {wall:.2f}초 / 승인 상한 21,600초. 구현·사전 테스트 및 사전 백업 시간은 학습 파이프라인 시간과 별도다.',
      '',
      '기존 코드·모델·Adam/RNG·결과 11,159개 파일의 사전 백업 커밋은 '+cfg['pre_experiment_backup_commit']+'이며 학습 전 GitHub bw 반영을 확인했다.',
      '새 실험은 기존 9종·28반응, 54/18/18 분할, 8,815 train 쌍, 2us target, 10ms horizon을 유지했다. 증강은 없다.',
      'Branch=현재 Y9, trunk=현재 T·log(P)·초기 당량비 phi. Phi는 데이터 initials에서 초기조건별 상수로 읽으며 미래 정답을 사용하지 않는다.',
      '폐쇄계의 원소비를 정확히 보존하면 phi는 조성에도 포함된 정보다. 이 실험은 보존된 혼합물 정보를 trunk에 명시하는 효과를 검토한다.',
      '파라미터 9,719개. 공통 T/Y/logP 전처리는 기존 ANN과 정확히 같고, phi 표준화는 선별 train 입력에서만 fit했다.',
      'ANN은 기존 첫10개(seed11,22,…,110), 10k 모델을 재사용했다. 새 10개는 같은 seed·batch512·복원추출·Adam·clip5 및 절대 update LR 경계 6000/9000을 사용했다.',
      '모든 모델을 10k까지 순차 학습해 저장한 뒤 Adam·RNG·history를 복원해 장기 학습했다. 재개는 LR/전체 6시간 시계를 초기화하지 않는다.',
      '장기 평탄 정책은 전체 train MSE를1k마다 관측하고 최소50k부터5k마다 두20k창의 q25/median/q75 상대변화1%/1%/3%를 연속3회(10k지속) 요구한다.',
      '모델별 순차 실행시간을 기록했고, 전체 예산 중40분은 두 종료점 평가·그림에 유보했다. 장기 모델들에는 남은 시간을 나누며 조기 평탄으로 남은 시간은 뒤 모델에 배분된다.',
      'TF intra/inter=2/1, OMP/OpenBLAS/MKL=2, oneDNN=0, CPU만 사용했다. 학습·평가·그림은 순차다.',
      '',
      f'평가 완료 {len(entries)}/20 arms: ANN10, phi10k10, ANN+phi10k20, philong10, ANN+philong20 × raw/repaired × val/test.',
      'Mixed 팀은 자기 현재 상태에서 20개 후보를 계산해 평균했다. Raw는 모든 후보가 유한해야 하며 repaired는 기존 동일 보정과 유효 후보 제외 규칙을 적용했다.',
      '초기 phi는 raw에서 원소량이 드리프트해도 고정된다. 물리 보정은 현재 예측 상태를 기준으로 동작하며 미래 참값을 사용하지 않는다.',
      'One-step은 참 현재 상태를 입력하고 rollout은 자기 예측을 재입력한다. 두 지표를 동일하게 해석하지 않는다.',
      'Test18 IC는 이전 연구에서도 관찰된 집합이다. Test를 종료·모델 선택이나 하이퍼파라미터 변경에 사용하지 않았다.',
      '미완주는 가용 prefix만 기록하고 전체 RMSE에서 제외했다. 공정 비교는 common_completed_IC_comparison.csv의 동일 IC 교집합을 사용한다.',
      '보존은 물리 보정으로 강제한 성질이며 학습된 보존성이 아니다. 엔트로피는 진단만 하고 후보 필터에는 사용하지 않았다.',
      '과거 ANN 병렬 그룹 학습 시간과 이번 순차 시간으로 공정한 속도비를 주장하지 않는다. NN/feature/decode/repair/rollout/one-step/로딩/warm-up 시간을 분리했다.',
      f'기존 결과 {len(previous)}개 SHA256 불변. 20개 checkpoint 해시·iteration과 평탄 검사 {checked}개를 재검증했다.',
      '',
      '|방법|test repaired10ms 완주|완주 IC 온도 RMSE 평균 K|',
      '|---|---:|---:|']
    selected=summary[(summary['split']=='test')&(summary['mode']=='repaired')&np.isclose(summary.horizon,.01)]
    for r in selected.itertuples():
        lines.append(f'|{LABELS[r.method]}|{r.completed}/{r.trajectories}|{r.temperature_RMSE_IC_mean_K:.6f}|')
    lines.extend(['','위 평균은 각 방법의 완주 집합 기준이다. 공통 완주 IC 수와 평균은 별도 CSV에서 확인한다.',
      '[공통 IC 비교](common_completed_IC_comparison.csv) · [학습 기록](training_summary.csv) · '
      '[모든 지표](summary.csv) · [one-step](one_step_summary.csv) · '
      '[loss](training_loss_and_lr.png) · [멤버별 loss](full_train_per_member.png) · '
      '[test18 온도](test_repaired_all18_temperature.png)',
      '각 test IC의 raw/repaired T+9종 PNG/PDF 및 profile CSV를 같은 폴더에 저장했다.',
      '이번 새 실험 파일은 아직 commit/push하지 않았다. 사전 백업 커밋과 구분한다.',''])
    (destination/'REPORT_KO.md').write_text('\n'.join(lines))
    write_json(destination/'verification.json',dict(endpoints=len(records),plateau_checks_recomputed=checked,
        preexisting_files_preserved=len(previous),evaluation_arms=len(entries)))
    write_json(destination/'complete.json',dict(output_sha256={p.name:sha256(p) for p in destination.iterdir() if p.is_file()}))
    event(root,phase='report_finished',path=str(destination),seconds=time.perf_counter()-started)
    write_json(root/'FINAL_ARTIFACTS.json',dict(report=str(destination),
        report_complete_sha256=sha256(destination/'complete.json'),
        pre_experiment_backup_commit=cfg['pre_experiment_backup_commit'],
        plateau_members=plateau,evaluation_arms=len(entries),pipeline_seconds=time.time()-json.loads((root/'budget.json').read_text())['started_unix'],
        finished_kst=datetime.now(timezone(timedelta(hours=9))).isoformat()))
    return destination
