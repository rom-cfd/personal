"""Immutable long-training diagnostics and common-IC, five-team comparison."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .ann_deeponet_study import check_deadline,verified_evaluation,write_json
from .p_only_training import sha256
from .data import SPECIES


COLORS={'ann10':'tab:blue','tp10':'tab:orange','p_only10':'tab:purple',
        'ann_tp20':'tab:green','ann_p_only20':'tab:red'}
LABELS={'ann10':'ANN10','tp10':'T/P10','p_only10':'P-only10',
        'ann_tp20':'ANN+T/P20','ann_p_only20':'ANN+P-only20'}
STYLE={'figure.facecolor':'white','axes.facecolor':'white','axes.grid':True,
       'grid.linestyle':':','grid.alpha':.3,'font.size':9,'savefig.dpi':140}


def common_completed(frames,horizon):
    """Full-horizon complete IC intersection, with no prefix substitution."""
    sets=[set(f.loc[np.isclose(f.horizon,horizon)&f.complete,'trajectory'].astype(int))
          for f in frames.values()]
    return sorted(set.intersection(*sets)) if sets else []


def save(fig,directory,name):
    fig.savefig(directory/f'{name}.png',bbox_inches='tight')
    fig.savefig(directory/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def training_figures(root,destination,records,deadline):
    rows,observed,mini=[],[],[]
    for r in records:
        if r is None:continue
        checkpoint=Path(r['checkpoint'])
        state=json.loads((checkpoint/'state.json').read_text())
        record={k:v for k,v in r.items() if k not in ('plateau_checks','timing')}
        record.update(**r['timing'],plateau_check_count=len(r['plateau_checks']))
        rows.append(record)
        observed.extend(dict(family=r['family'],**point) for point in state['full_train_history'])
        mini.extend(dict(family=r['family'],**point) for point in state['history'])
    table=pd.DataFrame(rows);history=pd.DataFrame(observed);batch=pd.DataFrame(mini)
    table.to_csv(destination/'training_summary.csv',index=False)
    history.to_csv(destination/'observed_full_train_history.csv',index=False)
    batch.to_csv(destination/'observed_minibatch_history.csv',index=False)
    for family,color in [('p_only','tab:purple'),('tp','tab:orange')]:
        check_deadline(deadline)
        selected=table[table.family==family]
        fig,axes=plt.subplots(5,2,figsize=(13,15),squeeze=False)
        for axis,(_,r) in zip(axes.flat,selected.iterrows()):
            h=history[(history.family==family)&(history.seed==r.seed)]
            axis.plot(h['update'],h.full_train_loss,color=color,lw=1.)
            axis.set_yscale('log');axis.minorticks_on();axis.grid(True,which='both',ls=':',alpha=.25)
            axis.set_title(f'Seed {int(r.seed)}: {int(r.updates):,} updates / {r.reason}')
            axis.set_xlabel('Optimizer update');axis.set_ylabel('Full-train residual MSE')
        for axis in list(axes.flat)[len(selected):]:axis.set_visible(False)
        fig.suptitle(f'{family}: complete observed loss curves; no test-based stopping')
        fig.tight_layout(rect=(0,0,1,.98));save(fig,destination,f'{family}_full_train_per_member')
    fig,axes=plt.subplots(3,1,figsize=(12,11),sharex=True,gridspec_kw={'height_ratios':[3,3,1]})
    plot_rows=[]
    for family,color in [('p_only','tab:purple'),('tp','tab:orange')]:
        for axis,frame,metric in [(axes[0],batch,'loss'),(axes[1],history,'full_train_loss')]:
            table=frame[frame.family==family].pivot(index='seed',columns='update',values=metric)
            # The family band uses only updates observed for every saved member.
            # Longer individual tails remain visible in the per-member figures.
            table=table.dropna(axis=1)
            if table.empty:continue
            values=table.to_numpy();low=values.min(axis=0);high=values.max(axis=0)
            mean=np.exp(np.log(values).mean(axis=0))
            axis.fill_between(table.columns,low,high,color=color,alpha=.1)
            axis.plot(table.columns,low,color=color,ls=':',lw=.8)
            axis.plot(table.columns,high,color=color,ls='--',lw=.8)
            axis.plot(table.columns,mean,color=color,lw=1.7,label=family)
            axis.set_yscale('log')
            plot_rows.extend(dict(family=family,metric=metric,update=int(u),minimum=float(a),
                geometric_mean=float(b),maximum=float(c),member_count=len(values))
                for u,a,b,c in zip(table.columns,low,mean,high))
    axes[0].set_ylabel('Observed mini-batch MSE');axes[1].set_ylabel('Observed full-train MSE')
    for family,color in [('p_only','tab:purple'),('tp','tab:orange')]:
        h=batch[(batch.family==family)&(batch.seed==batch[batch.family==family].seed.iloc[0])]
        axes[2].step(h['update'],h.learning_rate,where='post',color=color,label=family)
    axes[2].set_ylabel('Learning rate');axes[2].set_xlabel('Optimizer update per member')
    axes[0].legend();axes[1].legend()
    for a in axes:a.grid(True,which='both',ls=':',alpha=.25)
    axes[0].set_title('Both families: common observed updates within each family')
    fig.text(.5,.005,'Solid geometric mean; dashed max; dotted min; observed min-max shading, '
        'not a confidence interval. See individual plots for all tails.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.02,1,1));save(fig,destination,'training_loss_and_lr')
    pd.DataFrame(plot_rows).to_csv(destination/'training_plotted_statistics.csv',index=False)
    return pd.DataFrame(rows)


def evaluation_summary(root,destination):
    entries=[]
    for parent in sorted((root/'evaluations').glob('*')):
        markers=sorted(parent.glob('*/complete.json'))
        if markers:
            directory=markers[-1].parent;info=verified_evaluation(directory)
            entries.append((directory,info))
    rows,common_rows,one_rows=[],[],[]
    for directory,info in entries:
        frame=pd.read_csv(directory/'trajectories.csv')
        physics=pd.read_csv(directory/'physics.csv');events=pd.read_csv(directory/'events.csv')
        timing=json.loads((directory/'timing.json').read_text())
        for horizon,part in frame.groupby('horizon'):
            complete=part[part.complete];delays=part.tau400_relative_error_refined.dropna()
            rows.append(dict(method=info['method'],mode=info['mode'],split=info['split'],horizon=horizon,
                trajectories=len(part),completed=int(part.complete.sum()),
                completion_fraction=float(part.complete.mean()),
                temperature_RMSE_IC_mean_K=float(complete.full_temperature_rmse.mean()),
                temperature_RMSE_pooled_K=float(np.sqrt((complete.full_temperature_rmse**2).mean())),
                species_RMSE_IC_mean=float(complete.full_species_rmse.mean()),
                ignition_classification_fraction=float(part.ignition_classification_correct.mean()),
                ignition_pairs=len(delays),ignition_delay_relative_error_median=float(delays.median()),
                ignition_delay_relative_error_p95=float(delays.quantile(.95)),
                unresolved_before_ignition=int(part.unresolved_before_ignition.sum()),
                mass_drift_max=float(part.mass_sum_max.max()),element_drift_max=float(part.elemental_max.max()),
                full_trace_enthalpy_drift_max=float(physics.enthalpy_drift_max.max()),
                full_trace_entropy_decreasing_fraction=float(physics.entropy_decreasing_fraction.mean()),
                full_trace_minimum_entropy_delta=float(physics.minimum_entropy_delta.min()),
                full_trace_final_T_error_mean_K=float(physics.final_temperature_error.mean()),
                full_trace_invalid_candidate_fraction=1-float(events.valid_count.sum())/(len(events)*info['member_count']),
                full_trace_valid_ann_mean=float(events.valid_ann.mean()),
                full_trace_valid_deeponet_mean=float(events.valid_deeponet.mean()),
                rollout_seconds=timing['rollout_seconds'],one_step_seconds=timing['one_step_seconds'],
                warmup_seconds=timing['warmup_seconds'],diagnostics_seconds=timing['diagnostics_seconds'],
                **{f'rollout_{k}_seconds':v for k,v in timing['rollout_components'].items()},
                **{f'one_step_{k}_seconds':v for k,v in timing['one_step_components'].items()}))
        one_rows.append(dict(method=info['method'],mode=info['mode'],split=info['split'],
            **json.loads((directory/'one_step.json').read_text())))
    for split in ('val','test'):
        for mode in ('raw','repaired'):
            available={i['method']:pd.read_csv(d/'trajectories.csv') for d,i in entries
                       if i['split']==split and i['mode']==mode}
            if not available:continue
            groups=[('all_available_teams',available)]
            for names in [('ann10','tp10','ann_tp20'),('ann10','p_only10','ann_p_only20')]:
                if all(n in available for n in names):groups.append(('+'.join(names),{n:available[n] for n in names}))
            for group,frames in groups:
                for horizon in (.002,.01):
                    ids=common_completed(frames,horizon)
                    for method,frame in frames.items():
                        part=frame[np.isclose(frame.horizon,horizon)&frame.trajectory.isin(ids)]
                        common_rows.append(dict(group=group,method=method,split=split,mode=mode,horizon=horizon,
                            common_completed_IC_count=len(ids),ICs=json.dumps(ids),
                            temperature_RMSE_IC_mean_K=float(part.full_temperature_rmse.mean()),
                            temperature_RMSE_pooled_K=float(np.sqrt((part.full_temperature_rmse**2).mean())),
                            species_RMSE_IC_mean=float(part.full_species_rmse.mean())))
    pd.DataFrame(rows).to_csv(destination/'summary.csv',index=False)
    pd.DataFrame(common_rows).to_csv(destination/'common_completed_IC_comparison.csv',index=False)
    pd.DataFrame(one_rows).to_csv(destination/'one_step_summary.csv',index=False)
    return entries,pd.DataFrame(rows)


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


def report(root,cfg,snapshot,data,deadline):
    started=time.perf_counter();check_deadline(deadline)
    destination=root/'reports'/f'report_{time.time_ns()}';destination.mkdir(parents=True,exist_ok=False)
    records=json.loads((root/'final_members.json').read_text())
    with plt.rc_context(STYLE):
        training=training_figures(root,destination,records,deadline)
        entries,summary=evaluation_summary(root,destination)
        profile_figures(destination,entries,data,deadline)
    original=json.loads((root/'original_results_sha256.json').read_text())
    changed=[name for name,expected in original.items() if not Path(name).is_file() or sha256(name)!=expected]
    if changed:raise ValueError(f'Preexisting files changed: {changed[:5]}')
    markers=sorted((root/'loss_diagnostics').glob('*/complete.json'))
    if markers:
        directory=markers[-1].parent;verified_evaluation(directory)
        loss=pd.read_csv(directory/'residual_losses.csv');loss.to_csv(destination/'final_train_validation_losses.csv',index=False)
    lines=['# Sequential P-only / T/P long-training comparison','',
        'Endpoints were frozen using training loss and the declared plateau policy only. '
        'Test ICs were already observed in previous studies; no test-based stopping or tuning was used.','',
        'P-only continued the new 10k Adam/RNG states. T/P was newly reproduced from initialization '
        'because the historical optimizer/RNG states were not saved. Original files remain unchanged.','',
        f'Saved endpoints: {len(training)}; plateau: {int(training.reason.eq("plateau").sum())}; '
        f'time limit: {int(training.reason.eq("time_limit").sum())}; '
        f'update limit: {int(training.reason.eq("update_limit").sum())}.',
        f'Completed evaluation arms: {len(entries)}/20. Missing arms are unrun, not successful.','',
        'summary.csv reports full-horizon RMSE only on complete trajectories, alongside completion. '
        'common_completed_IC_comparison.csv compares identical completed IC sets and lists the ICs. '
        'Pooled temperature RMSE and the arithmetic mean of per-IC RMSE are separate columns. '
        'All test IC figures include T and nine species, with Cantera and the available five teams. '
        'Failed prefixes end without padding. Raw and repaired are separate experiments.','',
        'Repair imposes composition/element/enthalpy conditions; this is not learned conservation. '
        'Entropy is a diagnostic, not a filtering rule. Mean teams evaluate all members at their own '
        'current state. Invalid rows are excluded only in repaired mode, with family counts retained.','',
        'Timing separates model loading, warm-up, neural inference, decoding, candidate repair, '
        'mean repair, rollout, dense one-step evaluation and diagnostics. Training uses the same '
        'sequential engine and CPU thread settings, but plateau endpoints have different update counts. '
        'Historical grouped ANN time is not used as a direct speed comparison.','',
        f'Preexisting result files preserved: {len(original)}. Report seconds: {time.perf_counter()-started:.3f}.',
        'No commit/push or published snapshot change was performed.','']
    if not summary.empty:
        selected=summary[(summary['split']=='test')&(summary['mode']=='repaired')&np.isclose(summary.horizon,.01)]
        for r in selected.itertuples():
            lines.append(f'- {r.method}: {r.completed}/{r.trajectories} complete, '
                f'T RMSE mean on completed ICs={r.temperature_RMSE_IC_mean_K:.6g} K; '
                f'pooled={r.temperature_RMSE_pooled_K:.6g} K.')
    (destination/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(destination/'complete.json',dict(plateau_members=int(training.reason.eq('plateau').sum()),
        evaluation_arms=len(entries),preexisting_files_preserved=len(original),
        output_sha256={p.name:sha256(p) for p in destination.iterdir() if p.is_file()}))
    from .deeponet_long_study import event
    event(root,phase='report_finished',path=str(destination),report_seconds=time.perf_counter()-started)
    return destination
