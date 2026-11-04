"""Three-page matched-IC short/long NRMSE overview, all ten variables."""
import csv
import json
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/weight_long_summary_mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/weight_long_summary_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

root = Path('results/weight_long_study/run_001')
deadline = json.loads((root / 'budget.json').read_text())['expires_unix']
assert time.time() < deadline
analysis = json.loads((root / 'analysis_summary.json').read_text())
destination = root / 'comparison_summary'
destination.mkdir(exist_ok=True)
variables = ['T', 'H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'N2']
labels = {'ann10': 'ANN10', 'p_only10': 'P-only DeepONet10', 'ann_p_only20': 'ANN10 + P-only10',
          'tp10': 'T/P DeepONet10', 'ann_tp20': 'ANN10 + T/P10', 'tp_phi10': 'T/P/phi DeepONet10',
          'ann_tp_phi20': 'ANN10 + T/P/phi10'}


def number(x):
    if x is None:
        return 'N/A'
    if x == 0:
        return '0'
    return f'{x:.2e}' if abs(x) < .001 or abs(x) >= 1000 else f'{x:.3g}'


flat = []
with PdfPages(destination / 'NRMSE_short_vs_long.pdf') as pdf:
    for weight in (1, 100, 1000):
        assert time.time() < deadline
        comparisons = analysis['per_weight_within_team_comparisons'][str(weight)]['repaired']
        lookup = {(r['team'], r['variable']): r for r in comparisons}
        rows = []
        for team, label in labels.items():
            info = lookup[(team, 'T')]
            row = [label, str(info['common_IC_count'])]
            row += [f"{number(lookup[(team, v)]['median_NRMSE']['short'])}\n→ {number(lookup[(team, v)]['median_NRMSE']['long'])}" for v in variables]
            row += [f"{info['completed_of_18']['short']}/18 → {info['completed_of_18']['long']}/18"]
            rows.append(row)
            for v in variables:
                item = lookup[(team, v)]
                flat.append({'lambda': weight, 'model': label, 'variable': v, 'common_IC_count': item['common_IC_count'],
                             'common_ICs': json.dumps(item['common_ICs']),
                             **{f'{s}_median_NRMSE': item['median_NRMSE'][s] for s in ('baseline', 'short', 'long')},
                             **{f'{s}_complete_of_18': item['completed_of_18'][s] for s in ('baseline', 'short', 'long')}})
        fig, ax = plt.subplots(figsize=(27, 8.5)); ax.axis('off')
        fig.suptitle(f'Lambda {weight} | Cantera vs NN NRMSE | 10,000 updates → long continuation',
                     fontsize=21, fontweight='bold', y=.94)
        fig.text(.5, .865, 'Repaired 10 ms rollout; cell top: short, bottom: long. Medians use the SAME initial conditions within each model row.',
                 ha='center', fontsize=13, color='#475569')
        header = ['Model', 'Common ICs'] + variables + ['Complete rollouts']
        table = ax.table(cellText=rows, colLabels=header, cellLoc='center',
                         colWidths=[.155, .055] + [.068] * 10 + [.11], bbox=[0, .20, 1, .61])
        table.auto_set_font_size(False); table.set_fontsize(10.5)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#CBD5E1'); cell.set_linewidth(.5)
            if row == 0:
                cell.set_facecolor('#20344C'); cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor('#F1F5F9' if row % 2 == 0 else 'white')
                if col == 0:
                    cell.set_text_props(ha='left', weight='bold')
        fig.text(.04, .12, 'NRMSE = RMSE(Cantera, NN) / RMS(Cantera). Exact user util.py definition; dimensionless ratio, without ×100.', fontsize=12)
        fig.text(.04, .085, 'Common ICs are the intersection of baseline10k, soft10k and soft-long for each row. Row cohorts differ; use this table for within-model changes.', fontsize=12)
        fig.text(.04, .05, 'Long models ended at the time limit (joint plateau 0/120); actual endpoints: lambda1 41k–81k, lambda100 35k–84k, lambda1000 29k–62k.', fontsize=12)
        fig.subplots_adjust(left=.035, right=.985, top=.84, bottom=.12)
        pdf.savefig(fig)
        fig.savefig(destination / f'lambda_{weight}_NRMSE_short_vs_long.png', dpi=140)
        plt.close(fig)
with (destination / 'NRMSE_short_vs_long.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
print(json.dumps({'PDF_pages': 3, 'PNG_tables': 3, 'CSV_rows': len(flat), 'directory': str(destination)}))
