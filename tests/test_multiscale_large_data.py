import hashlib,json
from pathlib import Path
import numpy as np
from kinetics.multiscale_large_data import tasks_for,generate_shard
from kinetics.multiscale_data import integrate_labels
from kinetics.data import mechanism


def cfg():return json.loads(Path('configs/zhang_5p3m_data.json').read_text())


def test_full_counts_are_exact_and_group_ids_do_not_overlap():
    c=cfg();tasks=tasks_for(c)
    assert sum(t['initial_states']*(2 if t['hot'] else 1) for t in tasks)==5300000
    assert sum(t['initial_states'] for t in tasks)==4900000
    intervals=sorted((t['group_offset'],t['group_offset']+t['initial_states']) for t in tasks)
    assert intervals[0][0]==0 and intervals[-1][1]==4900000
    assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))


def test_shards_reproducible_and_reset_solver_matches_fresh_solver(tmp_path):
    c=cfg();(tmp_path/'parts').mkdir()
    t=dict(name='test',hot=True,k=4,ordinal=0,initial_states=12,group_offset=17)
    r=generate_shard(t,c,tmp_path,'test')
    with np.load(tmp_path/r['file']) as d:x=d['x'].copy();y=d['y'].copy();groups=d['group_id'].copy()
    r2=generate_shard(t,c,tmp_path,'test')
    assert r['sha256']==r2['sha256']
    np.testing.assert_array_equal(x[1::2,:10],y[::2])
    assert len(np.unique(groups))==12
    fresh=[];gas=mechanism()
    for a in x[::2]:fresh.extend(integrate_labels(a,2,c['dt'],gas)[1])
    fresh=np.asarray(fresh)
    np.testing.assert_allclose(y[:,0],fresh[:,0],rtol=0,atol=1e-4)
    np.testing.assert_allclose(y[:,1:],fresh[:,1:],rtol=0,atol=1e-8)
