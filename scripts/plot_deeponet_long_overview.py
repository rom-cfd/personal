"""All 18 test temperatures, read-only source traces, new PNG/PDF and provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/deeponet_long_study/run_001'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output; preserve all prior figures')
    run=json.loads((args.root/'run.json').read_text());source=Path(run['base_config']['source'])
    assert sha(source)==run['fingerprint']['dense_sha256']
    with np.load(source) as data:
        ids=np.flatnonzero(data['split']=='test');truth=data['states'][ids];dt=float(data['dt'])
    definitions=[('ann10','ANN10','tab:blue'),('tp10','T/P10','tab:orange'),
        ('p_only10','P-only10','tab:purple'),('ann_tp20','ANN+T/P20','tab:green'),
        ('ann_p_only20','ANN+P-only20','tab:red')]
    traces={};source_hashes={str(source):sha(source)}
    for method,_,_ in definitions:
        marker=sorted((args.root/'evaluations'/f'{method}_repaired_test').glob('*/complete.json'))[-1]
        info=json.loads(marker.read_text());path=marker.parent/'traces.npz'
        assert sha(path)==info['output_sha256']['traces.npz']
        source_hashes[str(path)]=sha(path)
        with np.load(path) as f:traces[method]={k:f[k] for k in f.files}
    args.output.mkdir(parents=True,exist_ok=False)
    with plt.rc_context({'figure.facecolor':'white','axes.facecolor':'white','font.size':8,
                         'axes.grid':True,'grid.linestyle':':','grid.alpha':.3,'savefig.dpi':150}):
        fig,axes=plt.subplots(6,3,figsize=(14,16),sharex=True,squeeze=False)
        for axis,i,reference in zip(axes.flat,ids,truth):
            axis.plot(np.arange(len(reference))*dt*1000,reference[:,0],color='black',ls='--',lw=1.2,label='Cantera')
            failures=[]
            for method,label,color in definitions:
                trace=traces[method][f'trajectory_{int(i)}']
                axis.plot(np.arange(len(trace))*dt*1000,trace[:,0],color=color,lw=.9,label=label)
                if len(trace)<len(reference):failures.append(f'{label} failed at {len(trace)*dt*1000:.3f} ms')
            axis.set_title(f'IC {int(i)}'+('\n'+', '.join(failures) if failures else ''))
            axis.set_xlim(0,10);axis.set_xlabel('Time [ms]');axis.set_ylabel('T [K]')
            axis.minorticks_on();axis.grid(True,which='both',ls=':',alpha=.25)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=6,bbox_to_anchor=(.5,.976))
        fig.suptitle('Repaired: all 18 test ICs, 10 ms independent self-fed temperatures',y=.995)
        fig.tight_layout(rect=(0,0,1,.958))
        fig.savefig(args.output/'test_repaired_all18_temperature.png',bbox_inches='tight')
        fig.savefig(args.output/'test_repaired_all18_temperature.pdf',bbox_inches='tight');plt.close(fig)
    (args.output/'complete.json').write_text(json.dumps(dict(script_sha256=sha(__file__),
        source_sha256=source_hashes,output_sha256={p.name:sha(p) for p in args.output.iterdir() if p.is_file()},
        note='No trace padding, teacher feedback, training or model selection; failed lines end early'),indent=2)+'\n')
    print('OVERVIEW_COMPLETE',args.output)


if __name__=='__main__':main()
