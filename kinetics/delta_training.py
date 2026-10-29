"""Fixed-budget 100-member signed physical-increment training; resumable."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .delta_model import DeltaPredictor
from .batched_training import train_group
from .committee import fit_metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/delta100_raw.json')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text()); root = Path(cfg['output'])
    if root.exists() and not args.resume:
        raise FileExistsError('Use --resume or a new output; preserve original data/models')
    root.mkdir(exist_ok=True); (root/'models').mkdir(exist_ok=True)
    config_path = root/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError('Changed config during resume')
    config_path.write_text(json.dumps(cfg, indent=2))
    with np.load(cfg['training_pairs']) as data:
        x, y, dt, parent = data['x'], data['y'], float(data['dt']), data['parent']
    with np.load(cfg['source']) as data:
        training_ids = np.flatnonzero(data['split']=='train')
    assert dt == cfg['dt'] and set(parent[:, 0]) <= set(training_ids)
    pair_hash = hashlib.sha256(Path(cfg['training_pairs']).read_bytes()).hexdigest()
    progress_path = root/'training_progress.json'
    previous = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if previous and previous['training_pair_sha256'] != pair_hash:
        raise ValueError('Source training pairs changed')
    started = time.perf_counter(); elapsed = previous.get('execution_seconds', 0.)
    deadline = started + max(0, cfg['max_train_seconds']-elapsed)
    records = json.loads((root/'members.json').read_text()) if (root/'members.json').exists() else []
    groups = pd.read_csv(root/'training_groups.csv').to_dict('records') if (root/'training_groups.csv').exists() else []
    history = pd.read_csv(root/'history.csv').to_dict('records') if (root/'history.csv').exists() else []
    for r in records:
        assert hashlib.sha256(Path(r['path']+'.keras').read_bytes()).hexdigest()==r['sha256']
    metric = fit_metric(x[:, :10], 1e-8)
    (root/'distance_metric.json').write_text(json.dumps(metric, indent=2))
    spec = dict(kind='signed_boxcox_physical_delta', gamma=.1)
    def persist(status):
        progress_path.write_text(json.dumps(dict(status=status, completed_members=len(records),
            training_pair_sha256=pair_hash, execution_seconds=elapsed+time.perf_counter()-started), indent=2))
    persist('training')
    try:
        complete = {r['seed'] for r in records}
        pending = [s for s in cfg['seeds'] if s not in complete]
        for offset in range(0, len(pending), cfg['group_size']):
            seeds = pending[offset:offset+cfg['group_size']]
            members, seconds, updates = train_group(spec, seeds, cfg['updates_per_member'], x, y, cfg,
                deadline=deadline, progress=True, predictor_class=DeltaPredictor)
            for seed, p in zip(seeds, members):
                path = root/'models'/f'physical_delta_seed{seed}'; p.save(path)
                q = DeltaPredictor.load(path)
                np.testing.assert_array_equal(p.predict(x[:8]), q.predict(x[:8]))
                records.append(dict(seed=seed, path=str(path), updates=cfg['updates_per_member'],
                    sha256=hashlib.sha256(Path(str(path)+'.keras').read_bytes()).hexdigest(),
                    train_pair_sha256=pair_hash))
            (root/'members.json').write_text(json.dumps(records, indent=2))
            groups.append(dict(seeds=','.join(map(str, seeds)), seconds=seconds, members=len(seeds)))
            history.extend(updates)
            pd.DataFrame(groups).to_csv(root/'training_groups.csv', index=False)
            pd.DataFrame(history).to_csv(root/'history.csv', index=False)
            persist('training'); print('DELTA_GROUP_COMPLETE', len(records), seconds, flush=True)
            del members, p, q; gc.collect()
        assert len(records)==100
        persist('completed')
        print('DELTA_TRAINING_COMPLETE', elapsed+time.perf_counter()-started, flush=True)
    except BaseException:
        persist('interrupted'); raise


if __name__ == '__main__':
    main()
