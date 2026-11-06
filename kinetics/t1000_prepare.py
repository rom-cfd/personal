"""Shared preprocessing for the T >= 1000 K pipeline (run once after the data generator).

1. Z-score cache (features/targets float32 + preprocessing.json) with the stage-1 trainer's own
   prepare_cache, fitted on the full new training set, at results/t1000_gelu/prep/g0.1/preprocessing.
2. metric.json next to it, with the same definition large_zhang_evaluate uses
   (temperature std floored at 100 K; per-species 95th percentiles of training x).
Every parent seed links to this cache, so all five see identical normalizers.
"""
import json
from pathlib import Path
import numpy as np
from .multiscale_large_data import atomic_json,sha
from .large_zhang_training import prepare_cache

DATA=Path('results/t1000_data');PREP=Path('results/t1000_gelu/prep/g0.1/preprocessing')


def main():
    cfg=json.loads(Path('configs/t1000/stage10.json').read_text());assert cfg['data']==str(DATA)
    info=prepare_cache(DATA,PREP,cfg['gamma'])
    print('CACHE_READY',info['examples'],flush=True)
    if not (PREP/'metric.json').exists():
        m=json.loads((DATA/'manifest.json').read_text());x=np.load(DATA/m['arrays']['x']['path'],mmap_mode='r')
        metric=dict(temperature_scale=max(float(info['input_scale'][0]),100.),
            species_scale=[max(float(np.quantile(x[:,j],.95)),1e-8) for j in range(1,10)],
            fit_split='train',fit_examples=len(x),derivation='temperature std and species 95th percentiles, training only')
        atomic_json(PREP/'metric.json',dict(data_manifest_sha256=sha(DATA/'manifest.json'),metric=metric))
    print('METRIC_READY',flush=True)


if __name__=='__main__':main()
