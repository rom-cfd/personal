"""Seven requested model rows x T/Y9 columns, one test IC per PDF page.

Reuse verified frozen traces. Only the four missing 10k P-only/T/P test
rollouts are computed, without training, dense one-step evaluation or selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/kinetics-cache')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter


OLD = Path('results/deeponet_long_study/run_001')
PHI = Path('results/tp_phi_study/run_001')
ROWS = [
    ('ANN10\n10k updates','ann10',None),
    ('P-only\nDeepONet10','p_short','p_long'),
    ('ANN10 +\nP-only DeepONet10','ann_p_short','ann_p_long'),
    ('T/P\nDeepONet10','tp_short','tp_long'),
    ('ANN10 +\nT/P DeepONet10','ann_tp_short','ann_tp_long'),
    ('T/P/'+r'$\phi$'+'\nDeepONet10','phi_short','phi_long'),
    ('ANN10 +\nT/P/'+r'$\phi$'+' DeepONet10','ann_phi_short','ann_phi_long')]
COLUMNS = [r'$T$ [K]',r'$Y_{\mathrm{H_2}}$',r'$Y_{\mathrm{H}}$',r'$Y_{\mathrm{O}}$',
    r'$Y_{\mathrm{O_2}}$',r'$Y_{\mathrm{OH}}$',r'$Y_{\mathrm{H_2O}}$',
    r'$Y_{\mathrm{HO_2}}$',r'$Y_{\mathrm{H_2O_2}}$',r'$Y_{\mathrm{N_2}}$']
BLUE, ORANGE, BLACK = '#0072B2','#D55E00','#262626'
STYLE = {'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,
    'axes.titlesize':12,'axes.linewidth':.6,'axes.spines.top':False,
    'axes.spines.right':False,'xtick.labelsize':8,'ytick.labelsize':8,
    'xtick.direction':'out','ytick.direction':'out','xtick.major.size':3,
    'ytick.major.size':3,'axes.facecolor':'white','figure.facecolor':'white',
    'pdf.fonttype':42,'pdf.compression':9,'savefig.facecolor':'white',
    'path.simplify':True,'path.simplify_threshold':.03}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def checked(path,expected,sources):
    actual=sha(path)
    if expected is not None and actual!=expected:raise ValueError(f'Source changed: {path}')
    sources[str(path)]=actual
    return path


def cached(root,method,sources):
    markers=sorted((root/'evaluations'/f'{method}_repaired_test').glob('*/complete.json'))
    if not markers:raise FileNotFoundError(f'Missing saved repaired test traces: {method}')
    marker=markers[-1];info=json.loads(marker.read_text())
    if (info['method']!=method or info['mode']!='repaired' or info['split']!='test'):
        raise ValueError('Trace arm identity differs')
    checked(marker,None,sources)
    path=marker.parent/'traces.npz'
    checked(path,info['output_sha256']['traces.npz'],sources)
    return str(path)


def prepare(output,resume):
    sources={}
    snapshot=json.loads((OLD/'run.json').read_text())
    phi_snapshot=json.loads((PHI/'run.json').read_text())
    for p in (OLD/'run.json',OLD/'final_members.json',PHI/'run.json',PHI/'matched_members.json',PHI/'final_members.json'):
        checked(p,None,sources)
    if snapshot['fingerprint']['dense_sha256']!=phi_snapshot['fingerprint']['dense_sha256']:
        raise ValueError('Different Cantera datasets')
    source=Path(snapshot['base_config']['source'])
    checked(source,snapshot['fingerprint']['dense_sha256'],sources)
    with np.load(source,allow_pickle=False) as f:data={k:f[k] for k in f.files}
    ids=np.flatnonzero(data['split']=='test').astype(int).tolist()
    if len(ids)!=18 or list(data['species'])!=['H2','H','O','O2','OH','H2O','HO2','H2O2','N2']:
        raise ValueError('Expected eighteen test ICs in the established species order')
    mapping={
        'ann10':cached(PHI,'ann10',sources),
        'p_long':cached(OLD,'p_only10',sources),
        'ann_p_long':cached(OLD,'ann_p_only20',sources),
        'tp_long':cached(OLD,'tp10',sources),
        'ann_tp_long':cached(OLD,'ann_tp20',sources),
        'phi_short':cached(PHI,'phi_10k10',sources),
        'phi_long':cached(PHI,'phi_long10',sources),
        'ann_phi_short':cached(PHI,'ann_phi_10k20',sources),
        'ann_phi_long':cached(PHI,'ann_phi_long20',sources)}
    # The P-only long run records its exact10k origins. T/P originals were
    # independently replayed at10k with array-equal weights in the saved audit.
    members={'ann':snapshot['fingerprint']['ann_members'],
        'p_short':[snapshot['p_only_origins'][str(seed)] for seed in snapshot['base_config']['seeds']],
        'tp_short':snapshot['original_tp']}
    for family,records in members.items():
        if len(records)!=10 or [r['seed'] for r in records]!=snapshot['base_config']['seeds']:
            raise ValueError('Incorrect seed/member order')
        for r in records:
            if r['updates']!=10000:raise ValueError('Short checkpoint is not10k')
            checked(r['path']+'.keras',r['sha256'],sources)
            checked(r['path']+'.json',r['preprocessing_sha256'],sources)
    record=dict(script_sha256=sha(__file__),source_sha256=sources,source=str(source),
        ids=ids,rows=ROWS,columns=COLUMNS,cached_traces=mapping,members=members,
        dt=float(data['dt']),mode='repaired',split='test',points_per_reference=len(data['states'][0]),
        note='No new training. New inference is restricted to missing frozen10k rollouts.')
    if output.exists():
        if not resume:raise FileExistsError('Use a new output or identical --resume')
        if json.loads((output/'run.json').read_text())!=json.loads(json.dumps(record)):
            raise ValueError('Source or plotting configuration changed')
    else:
        output.mkdir(parents=True,exist_ok=False);write(output/'run.json',record)
    return data,record


def missing_profiles(output,data,record):
    from kinetics.study import Predictor
    from kinetics.p_only_deeponet import PressureDeepONet
    from kinetics.deeponet import ThermoDeepONet
    from kinetics.mixed_ensemble import rollout,require_shared_preprocessing
    from kinetics.deeponet_long_evaluation import TimedMean
    markers=list((output/'generated_rollouts').glob('*/complete.json'))
    if len(markers)==4:return
    ids=np.array(record['ids']);states=data['states'][ids];pressures=data['pressures'][ids]
    before=time.perf_counter()
    ann=[Predictor.load(r['path']) for r in record['members']['ann']]
    p=[PressureDeepONet.load(r['path']) for r in record['members']['p_short']]
    tp=[ThermoDeepONet.load(r['path']) for r in record['members']['tp_short']]
    require_shared_preprocessing(ann+p+tp)
    print('MODELS_LOADED_SECONDS',round(time.perf_counter()-before,3),flush=True)
    definitions=[('p_short',p,['deeponet']*10),('ann_p_short',ann+p,['ann']*10+['deeponet']*10),
                 ('tp_short',tp,['deeponet']*10),('ann_tp_short',ann+tp,['ann']*10+['deeponet']*10)]
    for name,models,families in definitions:
        destination=output/'generated_rollouts'/name
        if (destination/'complete.json').exists():continue
        if destination.exists():raise FileExistsError('Preserve incomplete rollout; use new output')
        print('ROLLOUT_START',name,flush=True)
        team=TimedMean(models,families,'repaired');before=time.perf_counter()
        team.predict(np.c_[states[:,0],pressures]);warmup=time.perf_counter()-before
        team.reset_timing();before=time.perf_counter()
        traces,events=rollout(team,states[:,0],pressures,states.shape[1]-1,record['dt'],ids,
                             deadline=time.perf_counter()+1200)
        seconds=time.perf_counter()-before
        destination.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(destination/'traces.npz',**{f'trajectory_{i}':t for i,t in zip(ids,traces)})
        pd.DataFrame(events).to_csv(destination/'events.csv',index=False)
        write(destination/'timing.json',dict(rollout_seconds=seconds,warmup_seconds=warmup,
            components=team.timing,member_count=len(models),updates_each=10000,
            completed=sum(len(t)==states.shape[1] for t in traces),trajectory_count=len(ids)))
        write(destination/'complete.json',dict(method=name,mode='repaired',split='test',
            output_sha256={p.name:sha(p) for p in destination.iterdir() if p.is_file()}))
        print('ROLLOUT_FINISHED',name,'seconds',round(seconds,2),
              'completed',sum(len(t)==states.shape[1] for t in traces),'/18',flush=True)


def load_traces(output,data,record):
    mapping=dict(record['cached_traces'])
    for name in ('p_short','ann_p_short','tp_short','ann_tp_short'):
        directory=output/'generated_rollouts'/name
        marker=json.loads((directory/'complete.json').read_text())
        for filename,h in marker['output_sha256'].items():
            if sha(directory/filename)!=h:raise ValueError('Generated rollout changed')
        mapping[name]=str(directory/'traces.npz')
    traces={};coverage=[]
    expected={f'trajectory_{i}' for i in record['ids']}
    for name,path in mapping.items():
        with np.load(path,allow_pickle=False) as f:
            if set(f.files)!=expected:raise ValueError('Different ICs across plotted models')
            traces[name]={i:f[f'trajectory_{i}'] for i in record['ids']}
        for i,t in traces[name].items():
            if t.ndim!=2 or t.shape[1]!=10 or not np.isfinite(t).all():
                raise ValueError('Invalid trace shape/values')
            np.testing.assert_array_equal(t[0],data['states'][i,0])
            coverage.append(dict(method=name,trajectory=i,points=len(t),
                last_accepted_ms=(len(t)-1)*record['dt']*1000,
                complete=len(t)==record['points_per_reference']))
    return traces,coverage


def limits(reference,traces,i):
    ranges=[]
    for k in range(10):
        values=[reference[:,k]]+[t[i][:,k] for t in traces.values()]
        lo=min(float(a.min()) for a in values);hi=max(float(a.max()) for a in values)
        if k==0:
            pad=max((hi-lo)*.045,20.);ranges.append((lo-pad,hi+pad))
        elif k==9 and hi-lo < .02*max(hi,1e-12):
            # Do not amplify machine-precision changes of inert N2.
            pad=max(.02*hi,1e-6);ranges.append((max(0,lo-pad),hi+pad))
        else:ranges.append((0.,max(hi*1.08,1e-16)))
    return ranges


def figure(data,record,traces,i):
    reference=data['states'][i];dt=record['dt'];horizon=(len(reference)-1)*dt*1000
    ylimits=limits(reference,traces,i)
    fig,axes=plt.subplots(7,10,figsize=(31,20.2),sharex=True,squeeze=False)
    fig.subplots_adjust(left=.087,right=.990,bottom=.062,top=.900,wspace=.36,hspace=.38)
    status=[]
    for row,(label,short,long) in enumerate(ROWS):
        models=[(short,BLUE)]+([(long,ORANGE)] if long else [])
        failures=[]
        for name,color in models:
            trace=traces[name][i]
            if len(trace)<len(reference):
                failures.append(('10k' if name==short else 'long',(len(trace)-1)*dt*1000))
        for k,ax in enumerate(axes[row]):
            for name,color in models:
                trace=traces[name][i];tt=np.arange(len(trace))*dt*1000
                ax.plot(tt,trace[:,k],color=color,lw=1.05,alpha=.95,zorder=2)
                if len(trace)<len(reference):
                    ax.plot(tt[-1],trace[-1,k],marker='x',color=color,ms=3.5,mew=.9,zorder=5)
            ax.plot(np.arange(len(reference))*dt*1000,reference[:,k],color=BLACK,
                    ls=(0,(4,2.5)),lw=1.1,zorder=3)
            ax.set_xlim(0,horizon);ax.set_ylim(*ylimits[k])
            ax.set_xticks([0,2,4,6,8,10]);ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
            formatter=ScalarFormatter(useOffset=False,useMathText=True);formatter.set_powerlimits((-3,3))
            ax.yaxis.set_major_formatter(formatter);ax.yaxis.get_offset_text().set_size(7.5)
            ax.grid(True,which='major',ls=':',lw=.45,color='#b7c1cc',alpha=.65)
            ax.tick_params(pad=2)
            if row==0:ax.set_title(COLUMNS[k],pad=12,fontweight='semibold')
            if row==6:ax.set_xlabel('Time [ms]',labelpad=5)
        position=axes[row,0].get_position()
        fig.text(.009,(position.y0+position.y1)/2,f'{row+1}  {label}',
                 ha='left',va='center',fontsize=11,fontweight='semibold',linespacing=1.5)
        if failures:
            text='\n'.join(f'{which} stopped: {ms:.3f} ms' for which,ms in failures)
            axes[row,0].text(.985,.97,text,transform=axes[row,0].transAxes,
                ha='right',va='top',fontsize=6.8,color='#555555',
                bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=1.8))
        status.append(dict(row=row+1,variables=10,line_counts=[len(ax.lines) for ax in axes[row]],
            short_method=short,long_method=long,failures=failures))
    t0,p0,phi=data['initials'][i]
    fig.suptitle(f'Cantera vs neural networks  |  Test IC {i:02d}',x=.54,y=.980,
                 fontsize=21,fontweight='semibold')
    fig.text(.54,.954,f'$T_0$ = {t0:.1f} K     $P_0$ = {p0/101325:.3f} atm     '
        +r'$\phi_0$'+f' = {phi:.3f}     |     Repaired self-fed rollout, '+r'$\Delta t = 2\,\mu$s',
        ha='center',va='center',fontsize=12,color='#414b57')
    handles=[Line2D([0],[0],color=BLACK,ls=(0,(4,2.5)),lw=1.6,label='Cantera reference'),
        Line2D([0],[0],color=BLUE,lw=1.6,label='NN: 10,000 updates'),
        Line2D([0],[0],color=ORANGE,lw=1.6,label='NN: long training (time-limited)')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.941),ncol=3,
               frameon=False,fontsize=11.5,handlelength=3.5,columnspacing=3)
    fig.text(.54,.021,'Shared linear y scale within each variable column. '
        'All curves use saved 2 us samples. Failed predictions stop at their last accepted state (x).\n'
        'ANN always uses its 10k checkpoint; mixed teams use ANN10 + DeepONet10 at the same current state. '
        'Long endpoints reached their wall-time caps, without verified loss plateaus.',
        ha='center',va='center',fontsize=9.5,color='#4b5563',linespacing=1.5)
    if len(fig.axes)!=70:raise AssertionError('Figure must contain exactly7x10 axes')
    return fig,status


def render(output,data,record,preview=False):
    traces,coverage=load_traces(output,data,record)
    if preview:
        destination=output/'preview';destination.mkdir(exist_ok=False)
        with plt.rc_context(STYLE):
            fig,status=figure(data,record,traces,record['ids'][0])
            fig.savefig(destination/'first_IC_7x10.png',dpi=110)
            fig.savefig(destination/'first_IC_7x10.pdf')
            plt.close(fig)
        print('PREVIEW_COMPLETE',destination,flush=True);return
    if (output/'all_test_ICs_7x10.pdf').exists():raise FileExistsError('Do not overwrite final plots')
    pngs=output/'png';pngs.mkdir(exist_ok=False)
    pages=[];started=time.perf_counter()
    with plt.rc_context(STYLE), PdfPages(output/'all_test_ICs_7x10.pdf',metadata={
            'Title':'Cantera vs ANN and DeepONet: 7 x 10 comparison, 18 test initial conditions',
            'Author':'Hydrogen kinetics study','Subject':'Frozen10k and time-limited long repaired rollouts'}) as pdf:
        for page,i in enumerate(record['ids'],1):
            before=time.perf_counter();fig,status=figure(data,record,traces,i)
            pdf.savefig(fig)
            path=pngs/f'test_IC{i:02d}_7x10.png';fig.savefig(path,dpi=180)
            plt.close(fig)
            pages.append(dict(page=page,trajectory=i,png=str(path.relative_to(output)),
                              axes=70,rows=status))
            print('PAGE_FINISHED',page,'/18','IC',i,'seconds',round(time.perf_counter()-before,2),flush=True)
    pd.DataFrame(coverage).to_csv(output/'trace_coverage.csv',index=False)
    long=json.loads((OLD/'final_members.json').read_text())
    phi=json.loads((PHI/'final_members.json').read_text())
    ranges={name:[min(r['updates'] for r in group),max(r['updates'] for r in group)] for name,group in [
        ('P-only',[r for r in long if r['family']=='p_only']),('T/P',[r for r in long if r['family']=='tp']),('T/P/phi',phi)]}
    write(output/'pages.json',dict(page_count=18,axes_per_page=70,pages=pages,
        source_trace_sha256={str(p):sha(p) for p in sorted((output/'generated_rollouts').glob('*/traces.npz'))},
        long_updates_per_member=ranges,plot_seconds=time.perf_counter()-started))
    (output/'README_KO.txt').write_text(
        'test 18개 초기조건, IC별 한 페이지의7행×10열 비교 그림입니다.\n'
        '열: T,H2,H,O,O2,OH,H2O,HO2,H2O2,N2. T는 K, 종 변수는 질량분율 Y입니다.\n'
        '행1 ANN10 / 행2 P-only / 행3 ANN+P-only / 행4 T/P / 행5 ANN+T/P / 행6 T/P/phi / 행7 ANN+T/P/phi.\n'
        '검정 점선=Cantera, 파랑=10,000 update, 주황=장기 학습. 첫 행은 Cantera+ANN의2개 선입니다.\n'
        'ANN은 모든 혼합에서 기존10k를 유지하고 DeepONet만 단기/장기를 비교합니다.\n'
        f'장기 모델당 update 범위: {ranges}. 모두 평탄 미도달 시간상한 종료입니다.\n'
        '모두 repaired 자기 예측이며, 기존 동일 물리 보정 및 후보 평균 규칙을 사용했습니다.\n'
        '동일 IC 내 같은 열은 동일한 선형 y축을 씁니다. 열 사이 단위/범위는 다릅니다.\n'
        '중간 실패 trace는 마지막 정상점에서 종료하며 x로 표시합니다. 미래 값을 채우지 않았습니다.\n'
        'N2가 거의 일정할 때 약±2% 여백을 주어 기계정밀도 변화의 과도한 확대를 막았습니다.\n'
        '기존 source/model/result는 수정하지 않았고, 누락된4개10k 팀의 rollout만 순차 추론했습니다. 새 학습은 없습니다.\n'
        'PDF는18페이지 벡터 그림, png 폴더는IC별5580×3636 고해상도 이미지입니다.\n'
        '재현: scripts/plot_kinetics_7x10_comparison.py --phase plot --output 새경로\n')
    # Recheck every reused dataset, weight, preprocessing and saved trace after plotting.
    for p,h in record['source_sha256'].items():
        if sha(p)!=h:raise ValueError(f'Original source changed: {p}')
    archive=output/'all_test_ICs_7x10_bundle.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [output/'all_test_ICs_7x10.pdf',output/'README_KO.txt',output/'pages.json',
                  output/'trace_coverage.csv']+sorted(pngs.glob('*.png')):
            z.write(p,arcname=p.relative_to(output))
    write(output/'complete.json',dict(page_count=18,axes_per_page=70,png_count=18,
        source_files_preserved=len(record['source_sha256']),script_sha256=sha(__file__),
        output_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p.name!='complete.json'}))
    print('GRID_COMPLETE',output,'seconds',round(time.perf_counter()-started,2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/comparison_grid_7x10/run_001'))
    parser.add_argument('--phase',choices=('profiles','preview','plot'),default='plot')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();data,record=prepare(args.output,args.resume)
    if args.phase in ('profiles','plot'):missing_profiles(args.output,data,record)
    if args.phase in ('preview','plot'):render(args.output,data,record,preview=args.phase=='preview')


if __name__=='__main__':main()
