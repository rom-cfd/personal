"""Pair the frozen7x10 profile pages with case-wise Cantera-range NRMSE tables."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/kinetics-cache')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scripts.plot_kinetics_7x10_comparison import (load_traces,sha,write,ROWS,COLUMNS,BLUE,ORANGE)

VARIABLES=['T','H2','H','O','O2','OH','H2O','HO2','H2O2','N2']
ABSOLUTE_FLOORS=np.array([1e-8]+[1e-12]*9)  # K, then mass fractions
RELATIVE_RANGE_TOLERANCE=1e-8
METHODS=[(short,label.replace('\n',' '),'10k',row) for row,(label,short,long) in enumerate(ROWS,1)]
METHODS=[]
for row,(label,short,long) in enumerate(ROWS,1):
    label=label.replace('\n',' ').replace(' 10k updates','')
    METHODS.append((short,label,'10k',row))
    if long:METHODS.append((long,label,'long',row))


def profile_errors(reference,prediction,dt):
    """Uniform-time trapezoidal RMSE, whole-reference normalization, no padding."""
    reference=np.asarray(reference,dtype=float);prediction=np.asarray(prediction,dtype=float)
    if (reference.ndim!=2 or reference.shape[1]!=10 or len(reference)<2
            or prediction.ndim!=2 or prediction.shape[1]!=10
            or not 1<=len(prediction)<=len(reference) or dt<=0
            or not np.isfinite(reference).all() or not np.isfinite(prediction).all()):
        raise ValueError('Finite aligned10-variable traces and positive dt required')
    n=len(prediction);error=prediction-reference[:n]
    prefix=np.sqrt(np.trapezoid(error**2,dx=dt,axis=0)/((n-1)*dt)) if n>1 else np.abs(error[0])
    complete=n==len(reference)
    full=prefix.copy() if complete else np.full(10,np.nan)
    low=reference.min(axis=0);high=reference.max(axis=0);span=high-low
    tolerance=np.maximum(ABSOLUTE_FLOORS,RELATIVE_RANGE_TOLERANCE*np.abs(reference).max(axis=0))
    constant=span<=tolerance
    valid=~constant if complete else np.zeros(10,dtype=bool)
    normalized=np.full(10,np.nan);normalized[valid]=100*full[valid]/span[valid]
    status=np.full(10,'ok',dtype=object)
    status[constant]='near_constant_reference'
    if not complete:status[:]='incomplete_prediction'
    return dict(full_rmse=full,prefix_rmse=prefix,nrmse_percent=normalized,
        reference_min=low,reference_max=high,reference_range=span,range_tolerance=tolerance,
        status=status,complete=complete,points=n)


def metrics(data,traces,ids,dt):
    rows=[]
    for i in ids:
        for method,label,training,group in METHODS:
            trace=traces[method][i];result=profile_errors(data['states'][i],trace,dt)
            for k,variable in enumerate(VARIABLES):
                rows.append(dict(trajectory=i,method=method,model=label,training=training,
                    group=group,variable=variable,mode='repaired',complete=result['complete'],
                    points=result['points'],last_accepted_ms=(len(trace)-1)*dt*1000,
                    nrmse_status=result['status'][k],**{key:float(result[key][k]) for key in
                    ('full_rmse','prefix_rmse','nrmse_percent','reference_min','reference_max',
                     'reference_range','range_tolerance')}))
    frame=pd.DataFrame(rows)
    sets=[set(part.loc[part.complete,'trajectory']) for _,part in frame.groupby('method')]
    common=sorted(set.intersection(*sets))
    summary=[]
    for method,label,training,group in METHODS:
        for variable in VARIABLES:
            part=frame[(frame.method==method)&(frame.variable==variable)&frame.trajectory.isin(common)]
            values=part.nrmse_percent.dropna()
            summary.append(dict(method=method,model=label,training=training,group=group,
                variable=variable,common_ICs=json.dumps(common),common_IC_count=len(common),
                valid_nrmse_IC_count=len(values),median_nrmse_percent=values.median(),
                mean_nrmse_percent=values.mean(),std_nrmse_percent=values.std(ddof=0),
                q25_nrmse_percent=values.quantile(.25),q75_nrmse_percent=values.quantile(.75),
                minimum_nrmse_percent=values.min(),maximum_nrmse_percent=values.max(),
                completed_all_test_ICs=int(frame[(frame.method==method)&(frame.variable=='T')].complete.sum())))
    return frame,pd.DataFrame(summary),common


def number(value):
    if pd.isna(value):return 'N/A'
    if value==0:return '0'
    if abs(value)>=10000 or abs(value)<.01:return f'{value:.2e}'
    return f'{value:.3g}'


def table_figure(frame,common,*,trajectory=None,initial=None):
    summary=trajectory is None
    fig,ax=plt.subplots(figsize=(22,11.5));ax.axis('off')
    fig.subplots_adjust(left=.02,right=.98,bottom=.16,top=.83)
    rows=[];row_colors=[];row_training=[]
    for method,label,training,group in METHODS:
        part=frame[frame.method==method].set_index('variable')
        values=[]
        for variable in VARIABLES:
            r=part.loc[variable]
            values.append(number(r['median_nrmse_percent'] if summary else r['nrmse_percent'])
                if summary or r['complete'] else '—')
        status=f"{int(part.loc['T','completed_all_test_ICs'])}/18" if summary else (
            'Complete' if part.loc['T','complete'] else f"Stop {part.loc['T','last_accepted_ms']:.3f} ms")
        rows.append([f'{group}  {label}',training]+values+[status])
        color='#f1f5f9' if group%2==0 else '#ffffff'
        row_colors.append([color]*13);row_training.append(training)
    table=ax.table(cellText=rows,colLabels=['Model / same-state mean','Training']+COLUMNS+[
        'Complete ICs' if summary else '10 ms status'],cellColours=row_colors,
        cellLoc='center',colLoc='center',bbox=[0,0,1,1],
        colWidths=[.207,.06]+[.064]*10+[.093])
    table.auto_set_font_size(False);table.set_fontsize(10.5)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#d8dee6');cell.set_linewidth(.45)
        if row==0:
            cell.set_facecolor('#213248');cell.set_text_props(color='white',fontweight='semibold',fontsize=11)
            cell.set_height(cell.get_height()*1.12)
        elif col==0:
            cell.set_text_props(ha='left',fontweight='semibold');cell.PAD=.035
        elif col==1:cell.set_text_props(color=BLUE if row_training[row-1]=='10k' else ORANGE,fontweight='bold')
        elif col==12 and not summary and str(rows[row-1][-1]).startswith('Stop'):
            cell.set_text_props(color='#aa3126',fontweight='bold')
    if summary:
        title='Cantera vs NN | NRMSE summary [%]'
        subtitle=f'Median of per-IC NRMSE on the same {len(common)} complete test ICs for every model'
        extra='Common ICs: '+', '.join(map(str,common))+'.'
    else:
        t,p,phi=initial
        title=f'Cantera vs NN | Test IC {trajectory:02d} | NRMSE [%]'
        subtitle=f'T0 = {t:.1f} K    P0 = {p/101325:.3f} atm    phi0 = {phi:.3f}    |    Repaired self-fed, 10 ms'
        extra='All columns compare the NN curve with the same IC Cantera curve. First graph row has only the ANN10k result.'
    fig.suptitle(title,x=.5,y=.96,fontsize=22,fontweight='semibold')
    fig.text(.5,.913,subtitle,ha='center',va='center',fontsize=12,color='#475569')
    fig.text(.5,.871,r'$\mathrm{NRMSE}(\%)=100\,\mathrm{RMSE}_{time}/(\max_t y_{Cantera}-\min_t y_{Cantera})$',
             ha='center',va='center',fontsize=14)
    fig.text(.024,.11,'RMSE uses trapezoidal time integration on the common 2 us grid. '
        'The Cantera range uses the entire 10 ms reference and never NN values.\n'
        'N/A: constant or nearly constant Cantera reference (N2 is constant here). '
        '—: prediction stopped early; full-horizon NRMSE is unavailable.\n'
        'ANN stays at 10k in every mixed team. Long DeepONets stopped at their wall-time caps; no verified loss plateau.\n'
        +extra,ha='left',va='top',fontsize=10,color='#475569',linespacing=1.55)
    return fig


def validate_numerics():
    reference=np.repeat(np.arange(5.)[:,None],10,axis=1)
    prediction=reference+.5
    result=profile_errors(reference,prediction,1.)
    np.testing.assert_allclose(result['full_rmse'],.5,rtol=0,atol=1e-14)
    np.testing.assert_allclose(result['nrmse_percent'],12.5,rtol=0,atol=1e-13)
    constant=np.ones((5,10))*2
    result=profile_errors(constant,constant+.5,1.)
    assert np.isnan(result['nrmse_percent']).all()
    np.testing.assert_allclose(result['full_rmse'],.5)
    result=profile_errors(reference,prediction[:3],1.)
    assert np.isnan(result['full_rmse']).all() and np.isnan(result['nrmse_percent']).all()
    np.testing.assert_allclose(result['prefix_rmse'],.5)
    reference=np.zeros((3,10));reference[1]=1
    result=profile_errors(reference,reference+np.array([0.,1.,0.])[:,None],1.)
    np.testing.assert_allclose(result['full_rmse'],np.sqrt(.5))
    return ['known_constant_error','constant_reference_undefined','incomplete_no_full_score','trapezoidal_endpoint_weights']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-grid',type=Path,default=Path('results/comparison_grid_7x10/run_001'))
    parser.add_argument('--output',type=Path,default=Path('results/comparison_nrmse/run_001'))
    args=parser.parse_args();source=args.source_grid;output=args.output
    if output.exists():raise FileExistsError('Use a new result directory')
    checks=validate_numerics()
    marker=json.loads((source/'complete.json').read_text())
    if sha(source/'all_test_ICs_7x10.pdf')!=marker['output_sha256']['all_test_ICs_7x10.pdf']:
        raise ValueError('Original graph PDF changed')
    record=json.loads((source/'run.json').read_text())
    for path,h in record['source_sha256'].items():
        if sha(path)!=h:raise ValueError(f'Original input changed: {path}')
    with np.load(record['source'],allow_pickle=False) as f:data={k:f[k] for k in f.files}
    traces,coverage=load_traces(source,data,record)
    frame,summary,common=metrics(data,traces,record['ids'],record['dt'])
    # Compare the reused absolute temperature RMSE to the previously reported ANN.
    old=next(Path('results/tp_phi_study/run_001/evaluations/ann10_repaired_test').glob('*/trajectories.csv'))
    previous=pd.read_csv(old);previous=previous[np.isclose(previous.horizon,.01)].set_index('trajectory')
    current=frame[(frame.method=='ann10')&(frame.variable=='T')].set_index('trajectory')
    np.testing.assert_allclose(current.loc[previous.index,'full_rmse'],previous.full_temperature_rmse,rtol=1e-12,atol=1e-12)
    checks.append('ANN_absolute_RMSE_matches_previous_report')
    output.mkdir(parents=True,exist_ok=False);tables=output/'tables';tables.mkdir()
    frame.to_csv(output/'per_IC_variable_NRMSE.csv',index=False)
    summary.to_csv(output/'common_IC_NRMSE_summary.csv',index=False)
    pd.DataFrame(coverage).to_csv(output/'trace_coverage.csv',index=False)
    sources={str(source/'run.json'):sha(source/'run.json'),str(source/'complete.json'):sha(source/'complete.json'),
        str(source/'all_test_ICs_7x10.pdf'):sha(source/'all_test_ICs_7x10.pdf'),str(old):sha(old),
        **record['source_sha256']}
    write(output/'run.json',dict(script_sha256=sha(__file__),source_sha256=sources,source_grid=str(source),
        formula='100*time_integrated_RMSE/(full_IC_Cantera_max-full_IC_Cantera_min)',
        relative_range_tolerance=RELATIVE_RANGE_TOLERANCE,absolute_range_floors=ABSOLUTE_FLOORS.tolist(),
        summary_statistic='median of per_IC NRMSE_percent on identical complete ICs',
        common_completed_ICs=common,methods=METHODS,variables=VARIABLES,validation_checks=checks,
        note='Read-only frozen trajectories. No training, inference, model or test-based selection.'))
    style={'font.family':'DejaVu Sans','pdf.fonttype':42,'pdf.compression':9,
           'figure.facecolor':'white','savefig.facecolor':'white'}
    with plt.rc_context(style):
        fig=table_figure(summary,common)
        fig.savefig(output/'NRMSE_summary.pdf');fig.savefig(output/'NRMSE_summary.png',dpi=180)
        plt.close(fig)
        with PdfPages(output/'all_IC_NRMSE_tables.pdf') as pdf:
            for page,i in enumerate(record['ids'],1):
                fig=table_figure(frame[frame.trajectory==i],common,trajectory=i,initial=data['initials'][i])
                pdf.savefig(fig);fig.savefig(tables/f'test_IC{i:02d}_NRMSE.pdf')
                fig.savefig(tables/f'test_IC{i:02d}_NRMSE.png',dpi=180);plt.close(fig)
                print('NRMSE_TABLE_FINISHED',page,'/18','IC',i,flush=True)
    with tempfile.TemporaryDirectory(prefix='kinetics-nrmse-',dir='/tmp') as temp:
        prefix=Path(temp)/'graph_%d.pdf'
        subprocess.run(['pdfseparate',str(source/'all_test_ICs_7x10.pdf'),str(prefix)],check=True)
        alternating=[]
        for page,i in enumerate(record['ids'],1):
            alternating.extend([str(Path(temp)/f'graph_{page}.pdf'),str(tables/f'test_IC{i:02d}_NRMSE.pdf')])
        subprocess.run(['pdfunite',*alternating,str(output/'plots_and_NRMSE_tables.pdf')],check=True)
    (output/'README_KO.md').write_text(
        'Cantera vs NN NRMSE 표와 기존7×10 그림을 같은 IC 순서로 묶었다.\n\n'
        'plots_and_NRMSE_tables.pdf는 총36페이지로, 홀수 페이지가 그림, 바로 다음 짝수 페이지가 해당 IC NRMSE 표다.\n'
        'NRMSE_summary.pdf/png는 모든 모델이 완주한 동일 IC 집합에서 계산한 IC별 NRMSE의 중앙값이다.\n'
        f'공통 IC {len(common)}개: {common}. 전체 test는18개이며 미완주 수를 별도 표시했다.\n\n'
        'NRMSE(%)=100×시간적분RMSE/(해당 IC·변수의 전체 Cantera 최댓값−최솟값). '
        '시간적분은 기존 보고서와 같은 사다리꼴 규칙, 2us, 전체10ms다.\n'
        '미완주는 전체 RMSE/NRMSE를 비우고 가용 prefix 절대RMSE를 별도 기록했다. 미래 값을 채우지 않았다.\n'
        'Cantera 변화폭이 max(절대하한,1e-8×Cantera 최대절댓값) 이하이면 NRMSE는 N/A다. '
        '절대하한은 T=1e-8 K, Y=1e-12다. 이번 데이터의 N2는 정확히 일정하다. '
        '절대RMSE·분모·허용치는 CSV에 남겼으며 임의 분모를 넣어 N2의 정규화 점수를 만들지 않았다.\n'
        '상대 변화폭이 작은 비점화 IC는 NRMSE가 매우 커질 수 있다. '
        'NRMSE는 초기조건·변수별 Cantera 변화폭에 대한 비율이며 절대오차와 함께 읽어야 한다.\n'
        '전체 요약은 같은15개 IC의 중앙값이다. 평균·표준편차·사분위수·유효 개수도 CSV에 저장했다. '
        '종 변수들을 임의로 평균해 단일 순위 점수로 만들지 않았다.\n'
        '모델과 입력, repaired 보정, 단기10k·장기 종료점은 기존 그림과 동일하다. '
        'ANN은 혼합에서도10k를 유지하며 장기는 모두 평탄 미도달 시간상한 종료다.\n'
        '원본 코드·데이터·모델·결과를 수정하지 않았고 새 학습이나 추론은 수행하지 않았다.\n')
    for path,h in sources.items():
        if sha(path)!=h:raise ValueError('Original source changed')
    archive=output/'plots_and_NRMSE_bundle.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [output/'plots_and_NRMSE_tables.pdf',output/'NRMSE_summary.pdf',output/'NRMSE_summary.png',
            output/'all_IC_NRMSE_tables.pdf',output/'README_KO.md',output/'per_IC_variable_NRMSE.csv',
            output/'common_IC_NRMSE_summary.csv',output/'trace_coverage.csv',output/'run.json']+sorted(tables.glob('*.png')):
            z.write(p,arcname=p.relative_to(output))
    write(output/'complete.json',dict(paired_pdf_pages=36,IC_table_count=18,model_variant_count=13,
        variable_count=10,per_IC_variable_rows=len(frame),common_completed_ICs=common,
        numerical_checks=checks,source_files_preserved=len(sources),
        output_sha256={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}))
    print('NRMSE_REPORT_COMPLETE',output,'common_ICs',common,flush=True)


if __name__=='__main__':main()
