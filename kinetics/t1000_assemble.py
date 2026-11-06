"""Assemble the finished T >= 1000 K parents into one root (results/t1000_gelu/g0.1).

Each seed{S} folder holds the BEST-VALIDATION weights of ../seed{S}/stage1000/g0.1/seed{S} (copied from its best/,
see kinetics.t1000_best_roots), not the last epoch; the UR fine-tunes and the parent evaluations both start from
it. Shared preprocessing.json and metric.json come from the stage-1 preprocessing. Refuses a seed whose stage 2
is not completed at 1000 epochs.
"""
import json,shutil
from pathlib import Path
from .multiscale_large_data import atomic_json
from .t1000_best_roots import best_seed

ROOT=Path('results/t1000_gelu');OUT=ROOT/'g0.1';PREP=ROOT/'prep/g0.1/preprocessing'


def main(seeds=(11,22,33,44,55)):
    (OUT/'preprocessing').mkdir(parents=True,exist_ok=True)
    for name in ('preprocessing.json','metric.json'):shutil.copyfile(PREP/name,OUT/'preprocessing'/name)
    for s in seeds:best_seed(ROOT/f'seed{s}/stage1000/g0.1/seed{s}',OUT/f'seed{s}',require_epochs=1000)
    shutil.copyfile(ROOT/f'seed{seeds[0]}/stage1000/g0.1/config.json',OUT/'config.json')
    atomic_json(OUT/'README.json',dict(seeds=list(seeds),selected_model='best validation',
        layout='seed{S} = best/ of ../seed{S}/stage1000/g0.1/seed{S}',
        data='results/t1000_data (T 1000-3100 K)',preprocessing='results/t1000_gelu/prep/g0.1/preprocessing'))
    print('ASSEMBLED',list(seeds),flush=True)


if __name__=='__main__':main()
