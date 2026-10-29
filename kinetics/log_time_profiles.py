"""Redraw saved 10 ms profiles on log time axes without rerunning chemistry.

Run: python -m kinetics.log_time_profiles
The t=0 sample is omitted from display only; the first plotted time is dt.
"""
import argparse
import os
from pathlib import Path
import json

os.environ.setdefault('MPLCONFIGDIR', '/tmp/kinetics-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/committee100_long10ms'))
    args = parser.parse_args()
    root = args.root
    output = root / 'figures_log_time'
    output.mkdir(exist_ok=True)
    config = json.loads((root / 'config.json').read_text())
    with np.load(config['source']) as data:
        states, split = data['states'], data['split']
        species, dt = data['species'], float(data['dt'])
    times = np.arange(states.shape[1]) * dt * 1e3
    ids = np.flatnonzero(split == 'test')
    colors = dict(mean10='tab:blue', medoid10='tab:orange',
                  mean100='tab:green', medoid100='tab:red')
    traces = {}
    for name in colors:
        with np.load(root / 'traces' / f'{name}_test.npz') as archive:
            traces[name] = {int(i): archive[f'trajectory_{int(i)}'] for i in ids}

    def profile(ax, i, column, species_plot=False):
        def draw(values, **kwargs):
            y = values[1:, column]
            if species_plot:
                y = np.maximum(y, 1e-20)
            ax.plot(times[1:len(values)], y, **kwargs)
        draw(states[i], color='black', ls='--', lw=1.8, label='Cantera', zorder=5)
        for name, color in colors.items():
            draw(traces[name][i], color=color, lw=1, label=name)
        ax.set(xscale='log', xlim=(times[1], times[-1]),
               xlabel='Time (ms, log scale)', ylabel='Y' if species_plot else 'T (K)')
        ax.grid(True, which='major', alpha=.25)
        if species_plot:
            ax.set(yscale='log', ylim=(1e-20, 1.1))

    def save(fig, filename):
        fig.tight_layout(rect=(0, .025, 1, .96))
        fig.text(.5, .008, 't = 0 omitted on log axis; first sample = 0.002 ms (2 us).',
                 ha='center', fontsize=9)
        fig.savefig(output / filename, dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(3, 6, figsize=(22, 11), sharex=True)
    for i, ax in zip(ids, axes.flat):
        profile(ax, int(i), 0)
        ax.set(title=f'IC {i}: T0={states[i, 0, 0]:.0f} K', ylim=(150, 3500))
    axes.flat[-1].legend(fontsize=6)
    fig.suptitle('10 ms self-fed temperature profiles: logarithmic time')
    save(fig, 'all_temperature_profiles.png')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for i, ax in zip([44, 5, 64, 73], axes.flat):
        profile(ax, i, 0)
        ax.set_title(f'Previously difficult IC {i}')
    axes.flat[-1].legend(fontsize=8)
    fig.suptitle('Difficult initial conditions: logarithmic time')
    save(fig, 'difficult_initial_conditions.png')

    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
    i = int(ids[0])
    for j, ax in enumerate(axes.flat):
        profile(ax, i, j + 1, species_plot=True)
        ax.set_title(str(species[j]))
    axes.flat[-1].legend(fontsize=6)
    fig.suptitle(f'Nine species, held-out IC {i}: logarithmic time; display floor Y=1e-20')
    save(fig, 'species_profiles.png')
    print(f'Wrote three log-time figures to {output}')


if __name__ == '__main__':
    main()
