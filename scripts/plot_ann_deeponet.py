"""Matplotlib figures with separate case/statistic legends and explicit ranges.

Run from the repository root. Ranges are observed min/max, not confidence bands.
No interpolation, retraining, incomplete-trajectory padding or model selection.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


STYLE = {
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'axes.linewidth': 1., 'axes.grid': True,
    'grid.linestyle': ':', 'grid.alpha': .35,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'legend.frameon': True, 'legend.framealpha': .92,
    'savefig.dpi': 160, 'savefig.bbox': 'tight',
}
COLORS = {'ann10': 'tab:blue', 'deeponet10': 'tab:orange', 'mixed20': 'tab:green'}
LABELS = {'ann10': 'ANN (10)', 'deeponet10': 'DeepONet (10)', 'mixed20': 'Mixed (20)'}


def statistics(values):
    """Rows are members/ICs. Missing values stay missing; exact zeros stay zero."""
    valid = np.isfinite(values) & (values >= 0)
    count = valid.sum(axis=0)
    safe = np.where(valid & (values > 0), values, 1.)
    geometric = np.exp(np.log(safe).sum(axis=0) / np.maximum(count, 1))
    geometric[np.any(valid & (values == 0), axis=0)] = 0.
    minimum = np.min(np.where(valid, values, np.inf), axis=0)
    maximum = np.max(np.where(valid, values, -np.inf), axis=0)
    for array in (minimum, geometric, maximum):
        array[count == 0] = np.nan
    return dict(minimum=minimum, geometric_mean=geometric, maximum=maximum, count=count)


def positive(values):
    return np.where(np.asarray(values) > 0, values, np.nan)


def band(axis, x, stats, color):
    axis.fill_between(x, positive(stats['minimum']), positive(stats['maximum']), color=color, alpha=.09)
    axis.plot(x, positive(stats['maximum']), color=color, ls='--', lw=1.)
    axis.plot(x, positive(stats['minimum']), color=color, ls=':', lw=1.)
    axis.plot(x, positive(stats['geometric_mean']), color=color, ls='-', lw=1.8)
    axis.set_yscale('log')
    axis.minorticks_on()
    axis.grid(True, which='both', ls=':', alpha=.28)


def legends(axis, methods, population):
    cases = [Line2D([], [], color=COLORS[m], lw=2, label=LABELS[m]) for m in methods]
    first = axis.legend(handles=cases, title='Model family', loc='upper right')
    axis.add_artist(first)
    stats = [Line2D([], [], color='gray', ls=ls, label=label) for ls, label in
             [('--', 'max'), ('-', 'mean (geometric)'), (':', 'min')]]
    axis.legend(handles=stats, title=f'Statistic across {population}', loc='upper center')


def save(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / f'{name}.png')
    fig.savefig(output / f'{name}.pdf')
    plt.close(fig)


def training(root, output):
    run = json.loads((root / 'run.json').read_text())
    cfg = run['config']
    ann = pd.read_csv(Path(cfg['ann_root']) / 'history.csv')
    ann = ann[ann.seed.isin(cfg['seeds'])]
    deep = pd.concat([pd.read_csv(root / 'models' / f'history_seed{s}.csv') for s in cfg['seeds']])
    # The old ANN recorded only every 500 updates. Compare observed common
    # updates; do not invent an initial point or interpolate missing histories.
    updates = sorted(set(ann['update']) & set(deep['update']))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    rows = []
    for method, frame in [('ann10', ann), ('deeponet10', deep)]:
        table = frame.pivot(index='seed', columns='update', values='loss').reindex(
            index=cfg['seeds'], columns=updates)
        if table.isna().any().any():
            raise ValueError('Each plotted update requires all ten observed member losses')
        stats = statistics(table.to_numpy())
        band(axes[0], updates, stats, COLORS[method])
        rows.extend(dict(method=method, update=u, **{k: v[j] for k, v in stats.items()})
                    for j, u in enumerate(updates))
    axes[0].set_title('Training convergence: ANN vs DeepONet (10 members each)')
    axes[0].set_ylabel('Recorded mini-batch residual MSE')
    legends(axes[0], ['ann10', 'deeponet10'], '10 members')
    u = np.arange(1, cfg['updates_per_member'] + 1)
    lr = cfg['learning_rate'] * np.where(u <= .6 * len(u), 1., np.where(u <= .9 * len(u), .3, .1))
    for _, h in deep.groupby('seed'):
        np.testing.assert_allclose(h.learning_rate, lr[h['update'].to_numpy() - 1], rtol=1e-12)
    axes[1].step(u, lr, where='post', color='black', lw=1.8)
    axes[1].set_ylabel('Learning rate')
    axes[1].set_xlabel('Optimizer update per member')
    axes[1].set_yticks([.001, .0003, .0001], ['0.001', '0.0003', '0.0001'])
    axes[1].set_xlim(1, cfg['updates_per_member'])
    for axis in axes:
        for boundary in (6000.5, 9000.5):
            axis.axvline(boundary, color='gray', ls=':', lw=.9)
    save(fig, output, 'training_convergence')
    pd.DataFrame(rows).to_csv(output / 'training_statistics.csv', index=False)


def comparison(root, output):
    cfg = json.loads((root / 'run.json').read_text())['config']
    with np.load(cfg['source']) as f:
        data = {k: f[k] for k in f.files}
    for split in ('val', 'test'):
        ids = np.flatnonzero(data['split'] == split)
        truth = data['states'][ids]
        t = np.arange(truth.shape[1]) * float(data['dt']) * 1000
        for mode in cfg['modes']:
            traces = {}
            for method in COLORS:
                directory = root / 'evaluations' / f'{method}_{mode}_{split}'
                if not (directory / 'complete.json').exists():
                    continue
                marker = json.loads((directory / 'complete.json').read_text())
                for name, digest in marker['output_sha256'].items():
                    if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                        raise ValueError(f'Evaluation output changed: {directory / name}')
                with np.load(directory / 'traces.npz') as f:
                    traces[method] = [f[f'trajectory_{int(i)}'] for i in ids]
            if not traces:
                continue
            fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                                     gridspec_kw={'height_ratios': [2, 2, 1]})
            rows = []
            for method, histories in traces.items():
                errors = [np.full(truth.shape[:2], np.nan) for _ in range(2)]
                for j, trace in enumerate(histories):
                    n = len(trace)
                    errors[0][j, :n] = np.abs(trace[:, 0] - truth[j, :n, 0])
                    errors[1][j, :n] = np.sqrt(np.mean((trace[:, 1:] - truth[j, :n, 1:]) ** 2, axis=1))
                for metric, values, axis in zip(('temperature_absolute_error', 'species_rmse'), errors, axes):
                    stats = statistics(values)
                    band(axis, t, stats, COLORS[method])
                    rows.extend(dict(method=method, metric=metric, time_ms=float(t[k]),
                                     **{key: val[k] for key, val in stats.items()}) for k in range(len(t)))
                count = statistics(errors[0])['count']
                axes[2].step(t, count, where='post', color=COLORS[method], lw=1.7, label=LABELS[method])
            axes[0].set_title(f'{split} / {mode}: self-fed error against Cantera')
            axes[0].set_ylabel('Absolute temperature error [K]')
            axes[1].set_ylabel('Species RMSE per IC [mass fraction]')
            legends(axes[0], list(traces), f'{len(ids)} initial conditions')
            axes[2].set_ylabel('Available ICs')
            axes[2].set_xlabel('Time [ms]')
            axes[2].set_ylim(-.5, len(ids) + .5)
            axes[2].legend(loc='lower right', ncol=3, fontsize=8)
            axes[2].set_xlim(0, t[-1])
            fig.text(.5, .002, 'Observed min/max bands; time-wise surviving prefixes. '
                     'Read coverage with error; these curves are not full-horizon RMSE.', ha='center', fontsize=8)
            save(fig, output, f'{split}_{mode}_error_convergence')
            pd.DataFrame(rows).to_csv(output / f'{split}_{mode}_error_statistics.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/ann_deeponet_mean10/run_001')
    parser.add_argument('--kind', choices=('training', 'comparison'), required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = root / 'plots' / f'{args.kind}_{time.time_ns()}'
    output.mkdir(parents=True)
    with plt.rc_context(STYLE):
        (training if args.kind == 'training' else comparison)(root, output)
    metadata = dict(kind=args.kind, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    note='Shading is observed min/max, not confidence intervals. Zeros omitted on log axes.',
                    output_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in output.iterdir() if p.is_file()})
    (output / 'complete.json').write_text(json.dumps(metadata, indent=2))
    print('PLOTS_COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
