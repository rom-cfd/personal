"""Verify serialized Wang committees against recorded autoregressive first steps."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .committee import Committee


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/wang_hard10ms'))
    root = parser.parse_args().root
    config = json.loads((root / 'config.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert hashlib.sha256(Path(config['source']).read_bytes()).hexdigest() == manifest['source_sha256']
    with np.load(config['source']) as data:
        ids = np.flatnonzero(data['split'] == 'test')
        x = np.c_[data['states'][ids, 0], data['pressures'][ids]]
    rows = []
    for size in (10, 100):
        p = Committee.load(root / 'methods' / f'medoid{size}.json')
        assert p.physics_backend == 'wang_boxcox'
        for prefix, rule in [('mean', 'mean_after_repair'), ('medoid', 'medoid')]:
            descriptor = json.loads((root / 'methods' / f'{prefix}{size}.json').read_text())
            assert descriptor['rule'] == rule
            assert descriptor['paths'] == p.paths
            assert descriptor['physics_backend'] == 'wang_boxcox'
            p.rule = rule
            actual = p.predict(x)
            with np.load(root / 'traces' / f'{prefix}{size}_test.npz') as archive:
                expected = np.array([archive[f'trajectory_{int(i)}'][1] for i in ids])
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-10)
            seed_identity = True
            if rule == 'medoid':
                selected = p.last_diagnostics['selected_seed'].copy()
                events = pd.read_csv(root / f'selection_{prefix}{size}_test.csv')
                first = events[events.step == 1].set_index('trajectory').loc[ids]
                np.testing.assert_array_equal(selected, first.selected_seed)
                candidates = p.repair(p.proposals(x), x)
                for b, seed in enumerate(selected):
                    j = list(p.seeds).index(seed)
                    np.testing.assert_array_equal(actual[b], candidates[j, b])
            rows.append(dict(method=f'{prefix}{size}', serialized_reload=True,
                             max_absolute_error=float(np.max(np.abs(actual-expected))),
                             representative_identity=seed_identity if rule == 'medoid' else 'not_applicable'))
        del p
    pd.DataFrame(rows).to_csv(root / 'reload_checks.csv', index=False)
    (root / 'source_hashes.json').write_text(json.dumps({str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path('kinetics').glob('*.py'))}, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
