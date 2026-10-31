"""Plot observed training histories; never substitute training loss for test error."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .ann_deeponet_study import registered_members, sha256, write_json, check_deadline
from .deeponet import ThermoDeepONet


def histories(root, cfg, records):
    if len(records) != len(cfg['seeds']) or {r['seed'] for r in records} != set(cfg['seeds']):
        raise ValueError('Training report requires every configured member to complete')
    frames = []
    for r in records:
        h = pd.read_csv(Path(root) / 'models' / f"history_seed{r['seed']}.csv")
        if (h.empty or not h.seed.eq(r['seed']).all() or h['update'].duplicated().any() or
                not h['update'].is_monotonic_increasing or h['update'].iloc[-1] != cfg['updates_per_member'] or
                h['update'].iloc[0] < 1 or not np.isfinite(h.loss).all() or (h.loss < 0).any()):
            raise ValueError('Missing, incomplete or invalid observed training history')
        u = h['update'].to_numpy()
        expected = cfg['learning_rate'] * np.where(u <= cfg['updates_per_member'] * .6, 1.,
                                np.where(u <= cfg['updates_per_member'] * .9, .3, .1))
        np.testing.assert_allclose(h.learning_rate, expected, rtol=1e-12, atol=0)
        frames.append(h)
    return pd.concat(frames, ignore_index=True)


def training_report(root, budget_seconds=120):
    began = time.perf_counter()
    deadline = began + budget_seconds
    root = Path(root)
    run = json.loads((root / 'run.json').read_text())
    cfg, fingerprint = run['config'], run['fingerprint']
    records = registered_members(root, cfg, fingerprint)
    observed = histories(root, cfg, records)
    if sha256(cfg['training_pairs']) != fingerprint['pair_sha256']:
        raise ValueError('Training data changed')
    with np.load(cfg['training_pairs']) as data:
        x, y = data['x'], data['y']
    rows = []
    for r in records:
        check_deadline(deadline)
        p = ThermoDeepONet.load(r['path'])
        a, b = p.encode_state(x[:, :10]), p.encode_state(y)
        target = p.norm.forward(b) - p.norm.forward(a)
        features = p.features(x).astype('float32')
        squared = 0.
        for start in range(0, len(x), 512):
            check_deadline(deadline)
            end = start + 512
            prediction = p.model(features[start:end], training=False).numpy().astype('float64')
            squared += float(np.sum((prediction - target[start:end]) ** 2))
        h = observed[observed.seed == r['seed']]
        rows.append(dict(seed=r['seed'], updates=r['updates'], parameters=r['parameters'],
                         train_seconds=r['train_seconds'], logged_points=len(h),
                         first_logged_minibatch_loss=float(h.loss.iloc[0]),
                         final_logged_minibatch_loss=float(h.loss.iloc[-1]),
                         zero_residual_full_train_mse=float(np.mean(target ** 2)),
                         final_full_train_mse=squared / target.size))
    summary = pd.DataFrame(rows)
    if not np.isfinite(summary.final_full_train_mse).all():
        raise ValueError('Nonfinite final training-set loss')
    destination = root / 'training_reports' / f'report_{time.time_ns()}'
    destination.mkdir(parents=True)
    summary.to_csv(destination / 'summary.csv', index=False)
    observed.to_csv(destination / 'observed_history.csv', index=False)
    colors = plt.get_cmap('tab10')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    for i, seed in enumerate(cfg['seeds']):
        h = observed[observed.seed == seed]
        axes[0].semilogy(h['update'], np.maximum(h.loss, np.finfo(float).tiny),
                         color=colors(i), lw=1., alpha=.85, label=f'Seed {seed}')
    axes[0].set_ylabel('Mini-batch training MSE (transformed residual)')
    axes[0].set_title('DeepONet: 10 independent members, 10,000 updates each, batch 512')
    axes[0].legend(ncol=5, fontsize=8)
    common = observed[observed.seed == cfg['seeds'][0]]
    axes[1].step(common['update'], common.learning_rate, where='post', color='black', lw=1.7)
    axes[1].set_ylabel('Learning rate')
    axes[1].set_xlabel('Optimizer update per member')
    axes[1].set_yticks([.001, .0003, .0001], ['0.001', '0.0003', '0.0001'])
    for axis in axes:
        for boundary in (6000.5, 9000.5):
            axis.axvline(boundary, color='gray', ls='--', lw=.8)
        axis.grid(alpha=.2)
        axis.set_xlim(1, cfg['updates_per_member'])
    fig.tight_layout()
    fig.savefig(destination / 'training_loss_and_lr.png', dpi=160)
    plt.close(fig)
    check_deadline(deadline)
    fig, axes = plt.subplots(5, 2, figsize=(12, 14), sharex=True)
    for i, (axis, seed) in enumerate(zip(axes.flat, cfg['seeds'])):
        h = observed[observed.seed == seed]
        row = summary[summary.seed == seed].iloc[0]
        axis.semilogy(h['update'], np.maximum(h.loss, np.finfo(float).tiny), color=colors(i), lw=1)
        axis.axhline(max(row.final_full_train_mse, np.finfo(float).tiny), color='black',
                     ls=':', label='Final full-training-set MSE')
        axis.set_title(f'Seed {seed} | final full-train MSE = {row.final_full_train_mse:.3e}')
        axis.set_ylabel('Training MSE')
        axis.grid(alpha=.2)
        for boundary in (6000.5, 9000.5):
            axis.axvline(boundary, color='gray', ls='--', lw=.8)
        axis.legend(fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel('Optimizer update')
    fig.suptitle('Observed mini-batch loss; dotted level is a separate final full-train measurement')
    fig.tight_layout()
    fig.savefig(destination / 'training_loss_per_member.png', dpi=140)
    plt.close(fig)
    check_deadline(deadline)
    note = ['# DeepONet training only', '',
            'Branch: current Y9; trunk: current temperature and log pressure.',
            '10 members, 10000 Adam updates/member, batch 512, fixed final checkpoint.',
            'Learning rate: updates 1-6000 = .001; 6001-9000 = .0003; 9001-10000 = .0001.',
            'Curves show actually observed mini-batch training loss, logged at update 1,',
            'every 50 updates and the first update after each learning-rate change.',
            'The final full-training-set MSE is measured separately over all 8815 pairs.',
            'Neither quantity is validation/test error or evidence of rollout accuracy.',
            'No Cantera comparison evaluation or test-based model selection was run.', '',
            f'Total member training wall seconds: {summary.train_seconds.sum():.3f}',
            f'Final full-train MSE range: {summary.final_full_train_mse.min():.6e} to '
            f'{summary.final_full_train_mse.max():.6e}', '']
    (destination / 'REPORT.md').write_text('\n'.join(note))
    outputs = {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}
    write_json(destination / 'complete.json', dict(output_sha256=outputs,
               elapsed_seconds=time.perf_counter() - began, members=len(records),
               represented_updates=sum(r['updates'] for r in records), evaluation_run=False))
    print('TRAINING_REPORT_COMPLETE', destination, flush=True)
    print(summary.to_string(index=False), flush=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/ann_deeponet_mean10/run_001')
    parser.add_argument('--budget-seconds', type=float, default=120)
    args = parser.parse_args()
    if not np.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error('A positive finite budget is required')
    training_report(args.root, args.budget_seconds)
