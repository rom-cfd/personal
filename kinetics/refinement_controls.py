"""Equal-update, unaugmented three-seed control for augmentation attribution."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .data import samples
from .refinement import SPECS,train_updates,posteriori


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/refinement.json');parser.add_argument('--root',default='results/refinement');parser.add_argument('--output',default='results/refinement_controls');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(args.output)
    if out.exists():raise FileExistsError('Use a new controls directory')
    out.mkdir();(out/'models').mkdir();(out/'traces').mkdir()
    start=time.perf_counter();source=np.load(cfg['source']);x,y,_=samples(source,'train')
    ref=np.load(Path(args.root)/'validation_reference.npz');states=ref['states'];ids=ref['ids'];pressures=ref['pressures']
    rows=[];details=[];models=[]
    for spec in SPECS:
        for seed in cfg['seeds']:
            p,seconds,h=train_updates(spec,seed,cfg['final_updates'],x,y,cfg,start+900)
            name='original_'+spec['name'];path=out/'models'/f'final_{name}_seed{seed}'
            p.save(path);models.append(dict(name=name,seed=seed,path=str(path)))
            for mode in ('raw','physics'):
                result,rr,sp,traces=posteriori(p,states,pressures,ids,source['element_matrix'],cfg['dt'],mode)
                rows.append(dict(name=name,seed=seed,phase='control',mode=mode,train_seconds=seconds,**result))
                details.extend(dict(name=name,seed=seed,**r) for r in rr)
                np.savez_compressed(out/'traces'/f'{name}_seed{seed}_{mode}.npz',**{f'trajectory_{int(ids[k])}':t for k,t in traces})
                pd.DataFrame(rows).to_csv(out/'validation_summary.csv',index=False)
                pd.DataFrame(details).to_csv(out/'validation_trajectories.csv',index=False)
                print('CONTROL',name,seed,mode,result,flush=True)
    (out/'final_models.json').write_text(json.dumps(models,indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-start,updates=cfg['final_updates'],seeds=cfg['seeds'],augmentation=False,test_accessed=False),indent=2))


if __name__=='__main__':main()
