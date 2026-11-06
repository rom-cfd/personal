"""Page data: online learning study (per-case online fine-tune, gym, EWC and UR10 consolidation, round 2) vs strategy 5 and the gated hybrid.

The EWC strength c is selected on the VALIDATION rollouts only (lowest mean 20-ms T RMSE, ties by classification);
test numbers are reported for the selected c and, for transparency, for every c.
"""
import glob,json
from pathlib import Path
import numpy as np,pandas as pd

OUT=Path('results/t1000_learning_report');SEEDS=[11,22,33,44,55];SLOW=[7,9,52,69,78];CS=['c0','c0.01','c0.1','c1','c10']
dt=2e-6


def members(root):
    fs=sorted(glob.glob(f'{root}/evaluation_wang_members/seed*/trajectories.csv'))
    return pd.concat([pd.read_csv(f).assign(seed=int(f.split('/seed')[-1].split('/')[0])) for f in fs]) if len(fs)==5 else None


def load():
    M={'s5':pd.read_csv('results/t1000_gated_hybrid/gate_off/trajectories.csv'),
       'hybrid':pd.read_csv('results/t1000_gated_hybrid/gate0.3_handover50K/trajectories.csv')}
    on=pd.concat([pd.read_csv(f) for f in glob.glob('results/t1000_online_ft/online_ft_l2sp1e-3_lr1e-5/case*/trajectories.csv')]);on['trajectory']=on.case;M['online']=on
    for p in (1,2):M[f'gym{p}']=members(f'results/t1000_gym/pass{p}/g0.1')
    for c in CS:
        m=members(f'results/t1000_ewc/{c}/g0.1')
        if m is not None:M[f'ewc_{c}']=m
    for r in (1,3):
        m=members(f'results/t1000_ur10c_r{r}_best/g0.1')
        if m is not None:M[f'urc_r{r}']=m
    for R in rounds()[1:]:   # iterated rounds >= 2: gym with the previous round's members, then UR10 1:1 from them
        m=members(R['best'])
        if m is not None:M[R['key']]=m
    return M


def rounds():
    """Iterated gym -> UR10 1:1 consolidation rounds. Round N's new data comes from a gym run with round N-1's model
    (round 0 = strategy 5); round 1 is the original consolidation urc_r1."""
    out=[dict(n=1,key='urc_r1',gym='results/t1000_gym',data='results/t1000_gymur_r1',train='results/t1000_ur10c_r1',best='results/t1000_ur10c_r1_best/g0.1')]
    n=2
    stopped=Path('results/t1000_rounds.STOP').exists()  # loop stopped by the user: drop a cancelled, untrained round
    while Path(f'results/t1000_ur10c{n}_r1').exists() or (not stopped and Path(f'results/t1000_gym{n}').exists()):
        out.append(dict(n=n,key=f'round{n}',gym=f'results/t1000_gym{n}',data=f'results/t1000_gymur{n}_r1',train=f'results/t1000_ur10c{n}_r1',best=f'results/t1000_ur10c{n}_r1_best/g0.1'));n+=1
    return out


def train_info(tr,dd):
    cs=[json.load(open(f)) for f in sorted(glob.glob(f'{tr}/g0.1/seed*/completed.json'))]
    sel=[json.load(open(f)) for f in sorted(glob.glob(f'{tr}/g0.1/seed*/best/selection.json'))]
    par=[json.load(open(f)) for f in sorted(glob.glob(f'{tr}/g0.1/seed*/parent_validation.json'))]
    if len(cs)<5:return None
    man=json.load(open(f'{dd}/manifest.json'))
    return dict(seconds=float(sum(c['elapsed_seconds'] for c in cs)),epochs=[x['epoch'] for x in sel],
        val_unrolled_parent=float(np.mean([x['unrolled_validation'] for x in par])),val_unrolled_best=float(np.mean([x['validation_unrolled'] for x in sel])),
        val_one_parent=float(np.mean([x['one_step_validation_mae'] for x in par])),val_one_best=float(np.mean([x['validation_one_step_mae'] for x in sel])),
        windows=int(man['initial_states']),gym_windows=int(man['gym_windows']))


def gym_info(g):
    f=Path(g)/'state.json'
    if not f.exists():return None
    st=json.load(open(f));L=pd.DataFrame(st['log'])
    if 'forced' not in L:L['forced']=False
    L['triggered']=L.rounds-L.forced.astype(int)
    by=lambda col:{int(k):int(v) for k,v in L.groupby('pass_')[col].sum().items()}
    return dict(complete=(Path(g)/'manifest.json').exists(),episodes=int(len(L)),triggered=int(L.triggered.sum()),forced=int(L.forced.sum()),
        triggered_by_pass=by('triggered'),forced_by_pass=by('forced'),pairs=int(len(glob.glob(f'{g}/rounds/*.npz'))*1000),
        seconds=float(json.load(open(Path(g)/'manifest.json'))['seconds']) if (Path(g)/'manifest.json').exists() else None,
        triggered_cases={int(c):int(v) for c,v in L.groupby('case').triggered.sum().items() if v>0},
        max_R=float(L.max_R_smoothed_pre.max()))


def summary(df):
    out={}
    for sp,g in df[df.horizon==.02].groupby('split'):
        per=g.groupby('trajectory').full_temperature_rmse.mean()
        out[sp]=dict(valid=int(g.physically_admissible_complete.sum()),runs=len(g),cls=float(g.ignition_classification_correct.mean()),
            tau_med=float(g.tau400_relative_error_refined.median()),tau_mean=float(g.tau400_relative_error_refined.mean()),tau_max=float(g.tau400_relative_error_refined.max()),
            T_mean=float(g.full_temperature_rmse.mean()),T_med_case=float(per.median()),
            T_reg_med=float(per.drop([i for i in SLOW if i in per.index]).median()),T_slow_mean=float(per[[i for i in SLOW if i in per.index]].mean()))
    return out


def per_case(df):
    g=df[df.horizon==.02].groupby('trajectory')
    return {int(i):dict(tau=None if np.isnan(x.tau400_relative_error_refined.median()) else float(x.tau400_relative_error_refined.median()),
        T=float(x.full_temperature_rmse.mean()),ok=int(x.ignition_classification_correct.sum()),
        seeds=[None if np.isnan(v) else float(v) for v in x.sort_values('seed').tau400_relative_error_refined]) for i,x in g}


def traces(M,best,best_urc=None):
    with np.load('results/t1000_reference/reference20ms.npz') as d:refs={int(i):s for i,s in zip(d['ids'],d['states'])};taus={int(i):t for i,t in zip(d['ids'],d['refined_tau400'])}
    split=np.load('results/t1000_long10ms_data/trajectories.npz')['split']
    hyb=np.load('results/t1000_gated_hybrid/gate0.3_handover50K/traces.npz')
    out={}
    for i in SLOW:
        sp=str(split[i]);tau=taus[i]
        pts=np.unique(np.r_[np.round(np.geomspace(1,10000,320)).astype(int),np.clip(np.round(np.linspace(.5*tau,1.5*tau,200)/dt).astype(int),1,10000)])
        def mem(root):return [np.load(f'{root}/evaluation_wang_members/seed{s}/traces/seed{s}_{sp}.npz')[f'trajectory_{i}'][:,0] for s in SEEDS]
        series=dict(s5=mem('results/t1000_ur10_phys_best/g0.1'),hybrid=[hyb['T'][n] for n in np.flatnonzero(hyb['case']==i)],
            online=list(np.load(f'results/t1000_online_ft/online_ft_l2sp1e-3_lr1e-5/case{i}/traces.npz')['T']),
            gym2=mem('results/t1000_gym/pass2/g0.1'),ewc=mem(f'results/t1000_ewc/{best}/g0.1'))
        if best_urc:series['urc']=mem(f'results/t1000_ur10c_{best_urc}_best/g0.1')
        series['urc1']=mem('results/t1000_ur10c_r1_best/g0.1')
        for k in M:
            if k.startswith('round'):series[k]=mem(next(R['best'] for R in rounds() if R['key']==k))
        out[str(i)]=dict(split=sp,tau=float(tau),T0=float(refs[i][0,0]),idx=pts.tolist(),ref=np.round(refs[i][pts,0],2).tolist(),
            **{k:[np.round(np.asarray(v[k_])[np.minimum(pts,len(v[k_])-1)],2).tolist() for k_ in range(5)] for k,v in series.items()})
    return out


def main():
    OUT.mkdir(exist_ok=True);M=load()
    S={k:summary(v) for k,v in M.items()};C={k:per_case(v) for k,v in M.items()}
    ew=[k for k in M if k.startswith('ewc_')]
    best=min(ew,key=lambda k:(S[k]['val']['T_mean'],-S[k]['val']['cls']))[4:]
    uc=[k for k in M if k.startswith('urc_')]
    best_urc=min(uc,key=lambda k:(S[k]['val']['T_mean'],-S[k]['val']['cls']))[4:] if uc else None
    # online / gym costs
    rounds_on={int(Path(f).parent.name[4:]):len(json.load(open(f))) for f in glob.glob('results/t1000_online_ft/online_ft_l2sp1e-3_lr1e-5/case*/rounds.json')}
    ft_on=sum(r['finetune_seconds'] for f in glob.glob('results/t1000_online_ft/online_ft_l2sp1e-3_lr1e-5/case*/rounds.json') for r in json.load(open(f)))
    gym=pd.DataFrame(json.load(open('results/t1000_gym/state.json'))['log'])
    gym_ep=gym[gym.rounds>0][['pass_','case','split','T0','rounds','seconds']].to_dict('records')
    ewc_train={}
    for c in CS:
        recs=[json.load(open(f)) for f in sorted(glob.glob(f'results/t1000_ewc/{c}/g0.1/seed*/training.json'))]
        if recs:ewc_train[c]=dict(new0=float(np.mean([r['new_val_pretrained'] for r in recs])),old0=float(np.mean([r['old_val_pretrained'] for r in recs])),
            new=float(np.mean([r['best']['new_val'] if r['best'] else r['new_val_pretrained'] for r in recs])),
            old=float(np.mean([r['best']['old_val'] if r['best'] else r['old_val_pretrained'] for r in recs])),
            epochs=[r['best_epoch'] for r in recs],seconds=float(sum(r['seconds'] for r in recs)))
    hm=json.load(open('results/t1000_gated_hybrid/gate0.3_handover50K/manifest.json'))
    urc_train={}
    for key,tr,dd in (('r1','results/t1000_ur10c_r1','results/t1000_gymur_r1'),('r3','results/t1000_ur10c_r3','results/t1000_gymur_r3')):
        t=train_info(tr,dd)
        if t:urc_train[key]=t
    RD=[dict(R,gym_stats=gym_info(R['gym']),train_stats=train_info(R['train'],R['data']) if Path(R['data']).exists() else None,evaluated=R['key'] in M) for R in rounds()]
    cost=dict(online_rounds=rounds_on,online_rounds_total=int(sum(rounds_on.values())),online_ode_label_steps=int(sum(rounds_on.values())*10000),
        online_finetune_cpu_s=ft_on,gym_rounds={int(k):int(v) for k,v in gym.groupby('pass_').rounds.sum().items()},
        gym_ode_label_steps=int(gym.rounds.sum()*10000),gym_pairs=int(len(glob.glob('results/t1000_gym/rounds/*.npz'))*1000),
        gym_seconds=float(json.load(open('results/t1000_gym/manifest.json'))['seconds']),hybrid_ode_steps=int(hm['ode_calls']),ewc_train=ewc_train,urc_train=urc_train,
)
    # test without case 7 (strategy 5 never ignites it, so it has no tau error and dominates the mean T RMSE)
    ex7=[k for k in ('s5',f'ewc_{best}','urc_r1','urc_r3') if k in M]+[k for k in M if k.startswith('round')]
    excl7={k:summary(M[k][(M[k].split=='test')&(M[k].trajectory!=7)])['test'] for k in ex7}
    json.dump(dict(summary=S,cases=C,best_c=best,traces=traces(M,best,best_urc),best_urc=best_urc,cost=cost,gym_episodes=gym_ep,rounds=RD,slow=SLOW,excl7=excl7,
        test_ids_excl7=sorted(int(i) for i in M['s5'][M['s5'].split=='test'].trajectory.unique() if i!=7)),
        open(OUT/'report_data.json','w'),separators=(',',':'),default=lambda o:None)
    rows=[dict(method=k,split=sp,**v) for k,s in S.items() for sp,v in s.items()]
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    pd.set_option('display.width',250);print(pd.DataFrame(rows).sort_values(['split','method']).round(4).to_string());print('BEST c (val):',best,' BEST replay ratio (val):',best_urc)
    print({i:{k:(C[k][i]['tau'],round(C[k][i]['T'],1),C[k][i]['ok']) for k in C if i in C[k]} for i in SLOW})
    print('TEST WITHOUT #7');print(pd.DataFrame(excl7).T.round(4).to_string())


if __name__=='__main__':main()
