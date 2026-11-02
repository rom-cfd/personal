"""Verify saved P-only training states and plot observed training losses only.

No optimizer updates, validation/test evaluation, or source-result rewrites.
Run after training, sequentially, with the same CPU/oneDNN settings.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kinetics.p_only_training import PressureTrainer, sha256, write_json, learning_rate
from kinetics.mixed_ensemble import require_shared_preprocessing
from kinetics.study import Predictor
from plot_ann_deeponet import STYLE, statistics, band, save


def report(root, output, before):
    began = time.perf_counter()
    run = json.loads((root/'run.json').read_text())
    cfg, fingerprint = run['config'], run['fingerprint']
    assert sha256(cfg['training_pairs']) == fingerprint['pair_sha256']
    assert sha256(cfg['source']) == fingerprint['dense_sha256']
    for name, expected in fingerprint['code_sha256'].items():
        assert sha256(name) == expected, name
    with np.load(cfg['training_pairs']) as data:
        x, y = data['x'], data['y']
    ann_record = fingerprint['ann_members'][0]
    assert sha256(ann_record['path']+'.keras') == ann_record['sha256']
    assert sha256(ann_record['path']+'.json') == ann_record['preprocessing_sha256']
    ann = Predictor.load(ann_record['path'])
    rows, minibatches, full_train = [], [], []
    verified_checkpoints = 0
    for seed in cfg['seeds']:
        ckpts = sorted(p.parent for p in (root/'checkpoints'/f'seed{seed}').glob('*/complete.json'))
        for checkpoint in ckpts:
            marker = json.loads((checkpoint/'complete.json').read_text())
            for name, expected in marker['output_sha256'].items():
                assert Path(name).name == name and sha256(checkpoint/name) == expected
            verified_checkpoints += 1
        record_path = root/'members'/f'seed{seed}.json'
        if not ckpts:
            rows.append(dict(seed=seed,status='not_started',updates=0))
            continue
        checkpoint = ckpts[-1]
        trainer = PressureTrainer.restore(checkpoint, x, y, cfg, seed)
        assert trainer.predictor.model.count_params() == 9719
        require_shared_preprocessing([ann, trainer.predictor])
        assert int(trainer.optimizer.iterations.numpy()) == trainer.update
        if trainer.update:
            np.testing.assert_allclose(float(trainer.optimizer.learning_rate.numpy()),
                                       learning_rate(trainer.update),rtol=1e-7,atol=0)
        state = json.loads((checkpoint/'state.json').read_text())
        status = 'partial'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            assert trainer.update == record['updates'] == cfg['updates_per_member']
            assert record['seed'] == seed and record['checkpoint'] == str(checkpoint)
            assert record['sha256'] == sha256(checkpoint/'model.keras')
            assert record['preprocessing_sha256'] == sha256(checkpoint/'model.json')
            assert record['train_data_sha256'] == fingerprint['pair_sha256']
            status = 'completed'
        history = pd.DataFrame(trainer.history)
        losses = pd.DataFrame(trainer.full_train_history)
        if status == 'completed':
            assert history['update'].iloc[-1] == 10000
            # A deadline may expire after the final optimizer update but before
            # its diagnostic sweep. Never substitute the preceding loss for it.
            np.testing.assert_array_equal(losses['update'],np.arange(0,len(losses)*1000,1000))
        if not history.empty:
            expected_lr = [learning_rate(int(u)) for u in history['update']]
            np.testing.assert_allclose(history.learning_rate,expected_lr,rtol=1e-12,atol=0)
            minibatches.append(history)
        if not losses.empty:
            full_train.append(losses)
        rows.append(dict(seed=seed,status=status,updates=trainer.update,parameters=9719,
            optimizer_iteration=int(trainer.optimizer.iterations.numpy()),
            final_learning_rate=float(trainer.optimizer.learning_rate.numpy()),
            checkpoint=str(checkpoint),model_path=str(checkpoint/'model'),
            model_sha256=sha256(checkpoint/'model.keras'),
            preprocessing_sha256=sha256(checkpoint/'model.json'),
            checkpoint_reason=state['reason'],saved_member_elapsed_seconds=state['elapsed_seconds'],
            first_full_train_loss=float(losses.full_train_loss.iloc[0]) if len(losses) else None,
            final_full_train_loss=(float(losses.full_train_loss.iloc[-1])
                if len(losses) and losses['update'].iloc[-1]==trainer.update else None),
            minibatch_observations=len(history),full_train_observations=len(losses)))
        del trainer
        gc.collect()
    summary = pd.DataFrame(rows)
    output.mkdir(parents=True,exist_ok=False)
    summary.to_csv(output/'member_summary.csv',index=False)
    mini = pd.concat(minibatches,ignore_index=True) if minibatches else pd.DataFrame()
    full = pd.concat(full_train,ignore_index=True) if full_train else pd.DataFrame()
    mini.to_csv(output/'observed_minibatch_history.csv',index=False)
    full.to_csv(output/'observed_full_train_history.csv',index=False)
    plot_rows = []
    if not mini.empty:
        with plt.rc_context(STYLE):
            fig,axes = plt.subplots(3,1,figsize=(11,11),sharex=True,
                                   gridspec_kw={'height_ratios':[3,3,1]})
            for axis,frame,metric,title in [
                (axes[0],mini,'loss','Observed mini-batch residual MSE'),
                (axes[1],full,'full_train_loss','Observed full-training-set residual MSE')]:
                if frame.empty:
                    continue
                table = frame.pivot(index='seed',columns='update',values=metric)
                stats = statistics(table.to_numpy())
                band(axis,table.columns,stats,'tab:purple')
                axis.set_ylabel(title)
                for j,update in enumerate(table.columns):
                    plot_rows.append(dict(metric=metric,update=int(update),
                        **{key:float(value[j]) for key,value in stats.items()}))
            first = mini[mini.seed==mini.seed.iloc[0]]
            axes[2].step(first['update'],first.learning_rate,where='post',color='black')
            axes[2].set_yticks([.001,.0003,.0001],['.001','.0003','.0001'])
            axes[2].set_ylabel('Learning rate')
            axes[2].set_xlabel('Optimizer update per member')
            axes[0].set_title('P-only DeepONet training; observed member statistics')
            from matplotlib.lines import Line2D
            axes[0].legend(handles=[Line2D([],[],color='tab:purple',ls=ls,label=label)
                for ls,label in [('-','Geometric mean'),('--','Max'),(':','Min')]],loc='upper right')
            for axis in axes:
                axis.grid(True,which='both',ls=':',alpha=.28)
                for u in (6000.5,9000.5):
                    axis.axvline(u,color='gray',ls=':',lw=.8)
            fig.text(.5,.002,'Shading: observed min-max, not a confidence interval. '
                     'Training diagnostics only; no validation/test evaluation.',ha='center',fontsize=8)
            save(fig,output,'training_losses_and_lr')
        pd.DataFrame(plot_rows).to_csv(output/'plotted_statistics.csv',index=False)
    preserved = None
    if before:
        originals = json.loads(before.read_text())
        changed = [name for name,expected in originals.items()
                   if not Path(name).is_file() or sha256(name)!=expected]
        assert not changed,changed
        preserved = len(originals)
    execution = [json.loads(line) for line in (root/'execution.jsonl').read_text().splitlines()]
    completed = int(summary.status.eq('completed').sum())
    final_losses = (summary.loc[summary.status=='completed','final_full_train_loss'].dropna()
                    if completed else pd.Series(dtype=float))
    result = dict(status='all_members_completed' if completed==len(cfg['seeds']) else 'incomplete',
        completed_members=completed,planned_members=len(cfg['seeds']),
        member_updates_total=int(summary.updates.sum()),parameters_per_member=9719,
        verified_checkpoint_count=verified_checkpoints,preexisting_files_preserved=preserved,
        execution=execution,report_seconds=time.perf_counter()-began,
        final_full_train_MSE_member_count=len(final_losses),
        final_full_train_MSE_member_mean=float(final_losses.mean()) if len(final_losses) else None,
        test_used_for_selection=False,
        note='Saved training diagnostics only; no physical one-step or rollout evaluation',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short','--branch'],text=True),
        script_sha256=sha256(__file__))
    write_json(output/'verification.json',result)
    text = (f'P-only training: {completed}/{len(cfg["seeds"])} completed; '
        f'{result["member_updates_total"]} total member-updates.\n\n'
        'Each complete member has 9,719 parameters and 10,000 Adam updates. '
        'This is a new reproduction run, not recovered historical weights.\n\n'
        f'Runner execution: {json.dumps(execution)}\n\n'
        f'Final full-train transformed residual MSE, arithmetic member mean: '
        f'{result["final_full_train_MSE_member_mean"]}. This is not validation/test '
        'loss or evidence of accurate 10 ms rollout.\n\n'
        'The CSV lists exact loader paths and model/preprocessing hashes. All '
        'registered checkpoint file hashes and completed Adam iterations were verified. '
        'Minibatch and full-train observations are separate. Plot shading is observed '
        'min-max, not a confidence interval. No existing results were rewritten.\n\n'
        f'Preexisting files preserved: {preserved}. No commit or push was performed.\n')
    (output/'REPORT.md').write_text(text)
    write_json(output/'complete.json',dict(output_sha256={
        p.name:sha256(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/p_only_rebuild/run_001'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--before',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Report output exists; use a new path')
    report(args.root,args.output,args.before)


if __name__=='__main__':
    main()
