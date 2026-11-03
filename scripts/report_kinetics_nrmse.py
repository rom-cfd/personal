"""Pair frozen7x10 profiles with the user's util.py RMS-normalized NRMSE tables."""
import ast
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
from scripts.nrmse_util_reference import _nrmse_1d

VARIABLES=['T','H2','H','O','O2','OH','H2O','HO2','H2O2','N2']
METHODS=[]
for row,(label,short,long) in enumerate(ROWS,1):
    label=label.replace('\n',' ').replace(' 10k updates','')
    METHODS.append((short,label,'10k',row))
    if long:METHODS.append((long,label,'long',row))


def profile_errors(reference,prediction,dt):
    """Exact util.py norm='rms', Cantera=A, equal sample weights; no padding."""
    reference=np.asarray(reference,dtype=float);prediction=np.asarray(prediction,dtype=float)
    if (reference.ndim!=2 or reference.shape[1]!=10 or len(reference)<2
            or prediction.ndim!=2 or prediction.shape[1]!=10
            or not 1<=len(prediction)<=len(reference) or dt<=0
            or not np.isfinite(reference).all() or not np.isfinite(prediction).all()):
        raise ValueError('Finite aligned10-variable traces and positive dt required')
    n=len(prediction)
    # Preserve the uploaded1D function's reduction order, also in diagnostics.
    prefix=np.array([float(np.sqrt(np.mean((reference[:n,k]-prediction[:,k])**2)))
                     for k in range(10)])
    complete=n==len(reference)
    full=prefix.copy() if complete else np.full(10,np.nan)
    low=reference.min(axis=0);high=reference.max(axis=0);span=high-low
    rms=np.array([float(np.sqrt(np.mean(reference[:,k]**2))) for k in range(10)])
    normalized=np.full(10,np.nan)
    status=np.full(10,'ok',dtype=object)
    if complete:
        for k in range(10):
            try:normalized[k]=_nrmse_1d(reference[:,k],prediction[:,k],norm='rms')
            except ValueError as error:
                if str(error)!='ref data value is 0':raise
                status[k]='zero_reference_rms'
        np.testing.assert_array_equal(normalized[rms>0],full[rms>0]/rms[rms>0])
    else:status[:]='incomplete_prediction'
    return dict(full_rmse=full,prefix_rmse=prefix,
        nrmse=normalized,reference_min=low,reference_max=high,reference_range=span,reference_rms=rms,
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
                    ('full_rmse','prefix_rmse','nrmse','reference_min','reference_max',
                     'reference_range','reference_rms')}))
    frame=pd.DataFrame(rows)
    sets=[set(part.loc[part.complete,'trajectory']) for _,part in frame.groupby('method')]
    common=sorted(set.intersection(*sets))
    summary=[]
    for method,label,training,group in METHODS:
        for variable in VARIABLES:
            part=frame[(frame.method==method)&(frame.variable==variable)&frame.trajectory.isin(common)]
            values=part.nrmse.dropna()
            summary.append(dict(method=method,model=label,training=training,group=group,
                variable=variable,common_ICs=json.dumps(common),common_IC_count=len(common),
                valid_nrmse_IC_count=len(values),median_nrmse=values.median(),
                mean_nrmse=values.mean(),std_nrmse=values.std(ddof=0),
                q25_nrmse=values.quantile(.25),q75_nrmse=values.quantile(.75),
                minimum_nrmse=values.min(),maximum_nrmse=values.max(),
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
            values.append(number(r['median_nrmse'] if summary else r['nrmse'])
                if summary or r['complete'] else '—')
        status=f"{int(part.loc['T','completed_all_test_ICs'])}/18" if summary else (
            'Complete' if part.loc['T','complete'] else f"Stop {part.loc['T','last_accepted_ms']:.3f} ms")
        rows.append([f'{group}  {label}',training]+values+[status])
        color='#f1f5f9' if group%2==0 else '#ffffff'
        row_colors.append([color]*13);row_training.append(training)
    table=ax.table(cellText=rows,colLabels=['Model / same-state mean','Training']+[r'$T$']+COLUMNS[1:]+[
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
        title='Cantera vs NN | NRMSE summary (util.py: RMS normalization)'
        subtitle=f'Median of per-IC NRMSE on the same {len(common)} complete test ICs for every model'
        extra='Common ICs: '+', '.join(map(str,common))+'.'
    else:
        t,p,phi=initial
        title=f'Cantera vs NN | Test IC {trajectory:02d} | NRMSE (util.py)'
        subtitle=f'T0 = {t:.1f} K    P0 = {p/101325:.3f} atm    phi0 = {phi:.3f}    |    Repaired self-fed, 10 ms'
        extra='All columns compare the NN curve with the same IC Cantera curve. First graph row has only the ANN10k result.'
    fig.suptitle(title,x=.5,y=.96,fontsize=22,fontweight='semibold')
    fig.text(.5,.913,subtitle,ha='center',va='center',fontsize=12,color='#475569')
    fig.text(.5,.871,r'$\mathrm{NRMSE}=\sqrt{\mathrm{mean}[(y_{Cantera}-y_{NN})^2]}/\sqrt{\mathrm{mean}[y_{Cantera}^{2}]}$',
             ha='center',va='center',fontsize=14)
    fig.text(.024,.11,'Exact util.py defaults: norm="rms", reference A=Cantera. '
        'Every time sample has equal weight; entries are dimensionless ratios, without x100.\n'
        'N/A: zero Cantera RMS with a nonzero prediction. Constant nonzero references, including N2, are valid. '
        '—: prediction stopped early; full-horizon NRMSE is unavailable.\n'
        'ANN stays at 10k in every mixed team. Long DeepONets stopped at their wall-time caps; no verified loss plateau.\n'
        +extra,ha='left',va='top',fontsize=10,color='#475569',linespacing=1.55)
    return fig


def validate_numerics():
    reference=np.repeat(np.arange(1.,6.)[:,None],10,axis=1)
    prediction=reference+.5
    result=profile_errors(reference,prediction,1.)
    np.testing.assert_allclose(result['full_rmse'],.5,rtol=0,atol=1e-14)
    np.testing.assert_allclose(result['nrmse'],.5/np.sqrt(11.),rtol=0,atol=1e-14)
    constant=np.ones((5,10))*2
    result=profile_errors(constant,constant+.5,1.)
    np.testing.assert_allclose(result['nrmse'],.25,rtol=0,atol=1e-14)
    np.testing.assert_allclose(result['full_rmse'],.5)
    np.testing.assert_array_equal(profile_errors(constant,constant,1.)['nrmse'],np.zeros(10))
    zero=np.zeros((5,10))
    np.testing.assert_array_equal(profile_errors(zero,zero,1.)['nrmse'],np.zeros(10))
    invalid=profile_errors(zero,np.ones((5,10)),1.)
    assert np.isnan(invalid['nrmse']).all() and set(invalid['status'])=={'zero_reference_rms'}
    result=profile_errors(reference,prediction[:3],1.)
    assert np.isnan(result['full_rmse']).all() and np.isnan(result['nrmse']).all()
    np.testing.assert_allclose(result['prefix_rmse'],.5)
    reference=np.ones((3,10))
    result=profile_errors(reference,reference+np.array([0.,1.,0.])[:,None],1.)
    np.testing.assert_allclose(result['full_rmse'],np.sqrt(1/3))
    np.testing.assert_allclose(result['nrmse'],np.sqrt(1/3))
    def function_tree(path):
        return next(n for n in ast.parse(Path(path).read_text()).body
                    if isinstance(n,ast.FunctionDef) and n.name=='_nrmse_1d')
    assert ast.dump(function_tree('references/user_nrmse/util.py'))==ast.dump(function_tree('scripts/nrmse_util_reference.py'))
    return ['known_RMSE_over_reference_RMS','constant_nonzero_reference_valid','identical_constants_zero',
        'zero_reference_zero_prediction_zero','zero_reference_nonzero_prediction_undefined',
        'incomplete_no_full_score','equal_time_sample_weights','verbatim_user_function_AST_match']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-grid',type=Path,default=Path('results/comparison_grid_7x10/run_001'))
    parser.add_argument('--output',type=Path,default=Path('results/comparison_nrmse_util/run_001'))
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
    output.mkdir(parents=True,exist_ok=False);tables=output/'tables';tables.mkdir()
    frame.to_csv(output/'per_IC_variable_NRMSE.csv',index=False)
    summary.to_csv(output/'common_IC_NRMSE_summary.csv',index=False)
    pd.DataFrame(coverage).to_csv(output/'trace_coverage.csv',index=False)
    sources={str(source/'run.json'):sha(source/'run.json'),str(source/'complete.json'):sha(source/'complete.json'),
        str(source/'all_test_ICs_7x10.pdf'):sha(source/'all_test_ICs_7x10.pdf'),
        'references/user_nrmse/util.py':sha('references/user_nrmse/util.py'),
        'scripts/nrmse_util_reference.py':sha('scripts/nrmse_util_reference.py'),
        **record['source_sha256']}
    write(output/'run.json',dict(script_sha256=sha(__file__),source_sha256=sources,source_grid=str(source),
        formula='sqrt(mean((Cantera-NN)**2))/sqrt(mean(Cantera**2))',
        user_function='_nrmse_1d(A=Cantera,B=NN,norm="rms")',sample_weighting='equal',table_scale='ratio',
        denominator_floor=None,alignment='as-is; all saved trajectories have the same2us time grid',
        summary_statistic='median of per_IC NRMSE on identical complete ICs',
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
        'NRMSE=sqrt(mean((Cantera−NN)^2))/sqrt(mean(Cantera^2)). '
        '사용자 첨부 util.py의 _nrmse_1d(A=Cantera,B=NN,norm="rms")를 그대로 사용했다.\n'
        '표는 무차원 비율이며 %를 곱하지 않았다. 모든 시간점은 같은 가중치다. '
        '기존 오차 보고서의 사다리꼴 시간적분 대신 첨부 util.py와 같은 단순 평균을 적용했다.\n'
        '미완주는 전체 RMSE/NRMSE를 비우고 가용 prefix 절대RMSE를 별도 기록했다. 미래 값을 채우지 않았다.\n'
        '0이 아닌 일정한 Cantera 신호도 정상적으로 계산한다. N2를 상수라는 이유로 제외하지 않았다. '
        '두 신호가 동일한 상수이면 원본 함수처럼0을 반환한다. '
        'Cantera RMS=0인데 NN에 오차가 있으면 원본 함수가 ValueError를 내므로 표에서는 N/A로 기록한다. '
        '임의 분모나 epsilon 보정은 넣지 않았다. 절대RMSE·Cantera RMS·상태 코드를 CSV에 저장했다.\n'
        '모든 모델이 완주한 공통 IC의 중앙값을 전체 요약에 표시했다. 평균·표준편차·사분위수·유효 개수도 CSV에 저장했다. '
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
