"""One-page paired test summaries using only validation-frozen gamma choices."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('results/gamma_long/run_001')
VARIABLES = ['T', 'H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'N2']
TEAMS = [('ann10', 'ANN'), ('p_only10', 'P-only'), ('ann_p_only20', 'ANN + P-only'),
         ('tp10', 'T/P'), ('ann_tp20', 'ANN + T/P'), ('tp_phi10', 'T/P/phi'),
         ('ann_tp_phi20', 'ANN + T/P/phi')]
report = Path(json.loads((ROOT / 'FINAL_ARTIFACTS.json').read_text())['report'])

for mode in ('repaired', 'raw'):
    source = report / 'test' / mode / 'validation_selected_same_gamma_short_long.csv'
    with source.open() as f:
        rows = list(csv.DictReader(f))
    lookup = {(r['team'], r['variable']): r for r in rows}
    values, labels, notes = [], [], []
    for team, label in TEAMS:
        if (team, 'T') not in lookup:
            continue
        t = lookup[team, 'T']
        gamma = float(t['gamma'])
        for stage in ('short', 'long'):
            labels.append(f'{label}, g={gamma:g} | {stage}')
            values.append([f"{float(lookup[team, v][stage + '_median_NRMSE']):.4g}"
                           if lookup[team, v][stage + '_median_NRMSE'] != '' else 'N/A' for v in VARIABLES])
        notes.append(f"{label}: common n={t['common_IC_count']}, complete short={t['short_completed']}/18, "
                     f"long={t['long_completed']}/18; ICs={t['common_ICs']}")
    fig = plt.figure(figsize=(20, 11.5))
    ax = fig.add_axes((.20, .30, .77, .55))
    ax.axis('off')
    table = ax.table(cellText=values, rowLabels=labels, colLabels=VARIABLES,
                     cellLoc='center', loc='center', bbox=(0, 0, 1, 1))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#ccd1d9')
        cell.set_linewidth(.45)
        if r == 0:
            cell.set_facecolor('#24344b')
            cell.set_text_props(color='white', weight='bold')
        elif c == -1:
            cell.set_facecolor('#eaf0f7' if r % 2 else '#dce8f3')
        else:
            cell.set_facecolor('white' if r % 2 else '#f5f7fa')
    for pair in range(len(values) // 2):
        for c in range(10):
            short, long = values[2 * pair][c], values[2 * pair + 1][c]
            if short != 'N/A' and long != 'N/A':
                table[2 * pair + 2, c].set_text_props(color='#14734b' if float(long) < float(short) else '#b13c36' if float(long) > float(short) else '#333333')
    fig.suptitle(f'Test / {mode}: Cantera versus NN | paired short and long', fontsize=22, weight='bold', y=.96)
    fig.text(.03, .89, 'Gamma chosen on long-run validation by completion count, then temperature NRMSE. Each pair uses the SAME gamma and common completed test ICs.\n'
             'NRMSE = RMSE(Cantera, NN) / RMS(Cantera); dimensionless ratio, median across the listed ICs. Lower is better. Green/red: long improved/worsened.', fontsize=12)
    fig.text(.03, .25, '\n'.join(notes), fontsize=9, va='top', linespacing=1.55)
    flat = json.loads((ROOT / 'FINAL_ARTIFACTS.json').read_text())['plateau_members']
    fig.text(.03, .045, f'Short: 10,000 updates. Long: scalar training-MSE plateau or 150 s of cumulative new active work/model; plateau {flat}/240.\n'
             'Gamma=0.1 ANN/T-P origins were rebuilt because saved Adam states were unavailable; bootstrap counts within 150 s. Other 220 origins preserve exact Adam/RNG.\n'
             'Incomplete profiles have no full-horizon score. Pairwise cohorts may differ between teams; these rows do not establish an overall architecture ranking or all-species optimum.', fontsize=10)
    fig.savefig(ROOT / f'summary_test_{mode}.pdf')
    fig.savefig(ROOT / f'summary_test_{mode}.png', dpi=150)
    plt.close(fig)
print('COMPACT_SUMMARIES_CREATED')
