"""Build model roots that hold the best-validation weights of finished runs (T >= 1000 K pipeline).

Every later stage (UR fine-tune parents, rollout evaluation) uses the best-validation model, not the last epoch.
A best root has the layout the loaders expect: <dst>/seed{S}/model.keras, model.json and completed.json, plus
<dst>/preprocessing/{preprocessing,metric}.json. completed.json keeps the run record, with model_sha256 pointing
at the best weights and the last-epoch hash and selection record kept alongside.

  python -m kinetics.t1000_best_roots ur     # UR=10 roots for evaluation (after t1k_ur)
"""
import json,shutil,sys
from pathlib import Path
from .multiscale_large_data import atomic_json,sha


def best_seed(run,dst,require_epochs=None):
    """Copy run/best/model.* into dst (a seed folder) with a completed.json that names the best weights."""
    run=Path(run).resolve();dst=Path(dst);r=json.loads((run/'completed.json').read_text())
    assert r['status']=='completed' and r['model_sha256']==sha(run/'model.keras'),run
    if require_epochs is not None:assert r['epochs']==require_epochs,(run,r['epochs'])
    sel=json.loads((run/'best/selection.json').read_text())
    assert sel['model_sha256']==sha(run/'best/model.keras') and sel['metadata_sha256']==sha(run/'best/model.json'),run
    tracking=run/'best/tracking.json'
    dst.mkdir(parents=True,exist_ok=True)
    for name in ('model.keras','model.json'):shutil.copyfile(run/'best'/name,dst/name)
    assert sha(dst/'model.keras')==sel['model_sha256']
    atomic_json(dst/'completed.json',{**r,'model_sha256':sel['model_sha256'],'metadata_sha256':sel['metadata_sha256'],
        'selected_model':'best validation','selection':sel,'best_tracking':json.loads(tracking.read_text()) if tracking.exists() else None,
        'last_epoch_model_sha256':r['model_sha256'],'run_folder':str(run)})
    print('BEST',dst,'epoch',sel.get('epoch'),'validation',sel.get('validation_mae',sel.get('validation_unrolled')),flush=True)


def best_root(src_root,dst_root,seeds=(11,22,33,44,55)):
    """Best-validation copy of a whole root whose seed folders are src_root/seed{S}."""
    src_root,dst_root=Path(src_root),Path(dst_root);(dst_root/'preprocessing').mkdir(parents=True,exist_ok=True)
    for name in ('preprocessing.json','metric.json'):shutil.copyfile(src_root/'preprocessing'/name,dst_root/'preprocessing'/name)
    if (src_root/'config.json').exists():shutil.copyfile(src_root/'config.json',dst_root/'config.json')
    for s in seeds:best_seed(src_root/f'seed{s}',dst_root/f'seed{s}')
    atomic_json(dst_root/'README.json',dict(source=str(src_root),seeds=list(seeds),selected_model='best validation (best/ of each seed)'))


def main():
    if sys.argv[1:]!=['ur']:raise SystemExit(__doc__)
    for name in ('t1000_ur10','t1000_ur10_phys'):best_root(f'results/{name}/g0.1',f'results/{name}_best/g0.1')
    print('BEST_ROOTS_READY',flush=True)


if __name__=='__main__':main()
