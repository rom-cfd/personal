"""Copy finished UR models with the pressure-increment output pinned to exactly zero.

Constant-pressure reactor: every delta-P label is exactly 0 and the parent networks' P column is
exactly 0 (zero-initialized output layer, zero MAE subgradient). Unrolled training fed the P output
back as input, so later loss terms gave that column a small nonzero gradient; over 5000-step rollouts
the resulting ~1e-5 weights drift pressure by up to ~17 %. This writes copies with W[:,1]=b[1]=0 to a
new root (originals untouched); nothing else in the network changes.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from .multiscale_large_data import atomic_json,sha
from .zhang_model import ZhangPredictor


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('results/zhang_5p3m_unroll10/g0.1'))
    ap.add_argument('--output',type=Path,default=Path('results/zhang_5p3m_unroll10_pzero/g0.1'))
    ap.add_argument('--seeds',type=int,nargs='+',required=True)
    a=ap.parse_args()
    if not (a.output/'preprocessing').exists():shutil.copytree(a.root/'preprocessing',a.output/'preprocessing')
    for s in a.seeds:
        src=a.root/f'seed{s}';dst=a.output/f'seed{s}'
        r=json.loads((src/'completed.json').read_text());assert r['status']=='completed' and r['model_sha256']==sha(src/'model.keras')
        if dst.exists():raise FileExistsError(dst)
        dst.mkdir(parents=True);p=ZhangPredictor.load(src/'model');layer=p.model.layers[-1];W,b=layer.get_weights()
        before=dict(p_col_abs_max=float(np.abs(W[:,1]).max()),p_bias=float(b[1]));W[:,1]=0.;b[1]=0.;layer.set_weights([W,b])
        p.model.save(dst/'model.keras');shutil.copy2(src/'model.json',dst/'model.json')
        atomic_json(dst/'completed.json',dict(r,status='completed',model_sha256=sha(dst/'model.keras'),source_model_sha256=r['model_sha256'],
            source=str(src),modification='output-layer pressure column and bias set to 0',removed=before))
        print('PZERO',s,before,flush=True)


if __name__=='__main__':main()
