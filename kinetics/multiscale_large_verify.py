"""Verify all full-scale shard hashes/content and sample fresh Cantera labels."""
import argparse,json
from pathlib import Path
import numpy as np
from .multiscale_large_data import sha,atomic_json,tasks_for,LEGACY_SOURCE_SHA256
from .multiscale_data import integrate_labels
from .data import mechanism


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/zhang_5p3m_data'))
    parser.add_argument('--preflight',action='store_true',help='Allow tiny fixture counts; never labels them full-scale')
    args=parser.parse_args();root=args.root
    m=json.loads((root/'manifest.json').read_text());cfg=json.loads((root/'config.json').read_text())
    assert m['status']=='completed'
    accepted_sources=set(m.get('accepted_generator_sources',[m['source_sha256']]))
    assert accepted_sources <= {m['source_sha256'],LEGACY_SOURCE_SHA256}
    assert accepted_sources == {r['source_sha256'] for r in m['parts']}
    if not args.preflight:
        assert m['pairs']==5300000 and m['basic_pairs']==4500000 and m['hot_pairs']==800000
        assert m['independent_initial_states']==4900000 and len(m['parts'])==2450
    expected_pairs=len(cfg['scales'])*cfg['per_scale']+2*cfg['hot_initial_states']
    expected_groups=len(cfg['scales'])*cfg['per_scale']+cfg['hot_initial_states']
    assert m['pairs']==expected_pairs and len(m['parts'])==len(tasks_for(cfg))
    x=np.load(root/m['arrays']['x']['path'],mmap_mode='r');y=np.load(root/m['arrays']['y']['path'],mmap_mode='r')
    for key in ['x','y']:assert sha(root/m['arrays'][key]['path'])==m['arrays'][key]['sha256']
    gas=mechanism();offset=0;group_end=0;negative=0;max_t_error=0.;max_y_error=0.;fresh_cases=0;per_scale={};hot_rows=0
    for number,r in enumerate(m['parts']):
        assert sha(root/r['file'])==r['sha256'],r['file']
        assert r['config_sha256']==m['config_sha256'] and r['source_sha256'] in accepted_sources
        assert r['max_element_residual']<1e-9
        with np.load(root/r['file']) as d:
            xx=d['x'];yy=d['y'];groups=d['group_id'];positions=d['group_step']
            np.testing.assert_array_equal(x[offset:offset+len(xx)],xx)
            np.testing.assert_array_equal(y[offset:offset+len(yy)],yy)
            assert np.isfinite(xx).all() and np.isfinite(yy).all() and np.isfinite(d['initial_hrr_W_m3']).all()
            assert xx[:,1:10].min()>=0 and yy[:,1:].min()>=0
            np.testing.assert_allclose(xx[:,1:10].sum(axis=1),1,atol=2e-14)
            np.testing.assert_allclose(yy[:,1:].sum(axis=1),1,atol=2e-10)
            assert np.min(groups)==group_end and np.max(groups)==group_end+r['initial_states']-1
            assert np.min(xx[:,-1])>=cfg['pressure_atm'][0]*101325. and np.max(xx[:,-1])<=cfg['pressure_atm'][1]*101325.
            assert np.min(xx[:,9])>=cfg['nitrogen_mass_fraction'][0]-1e-14 and np.max(xx[:,9])<=cfg['nitrogen_mass_fraction'][1]+1e-14
            if r['hot']:
                np.testing.assert_array_equal(xx[1::2,:10],yy[::2])
                np.testing.assert_array_equal(groups[::2],groups[1::2])
                np.testing.assert_array_equal(positions,np.tile([0,1],r['initial_states']))
                hot_rows+=len(xx)
            else:
                assert np.all(positions==0)
                per_scale[r['k']]=per_scale.get(r['k'],0)+len(xx)
            negative+=int((d['initial_hrr_W_m3']<0).sum())
            if number%10==0:
                steps=2 if r['hot'] else 1;start=min(7,r['initial_states']-1)*steps
                _,fresh,_=integrate_labels(xx[start],steps,cfg['dt'],gas)
                expected=yy[start:start+steps]
                t_error=float(np.max(np.abs(expected[:,0]-fresh[:,0])))
                y_error=float(np.max(np.abs(expected[:,1:]-fresh[:,1:])))
                assert t_error<=1e-4 and y_error<=1e-8,(r['file'],t_error,y_error)
                max_t_error=max(max_t_error,t_error);max_y_error=max(max_y_error,y_error);fresh_cases+=1
            offset+=len(xx);group_end+=r['initial_states']
        if (number+1)%100==0:print('VERIFIED_SHARDS',number+1,'/',len(m['parts']),flush=True)
    assert offset==expected_pairs and group_end==expected_groups and hot_rows==2*cfg['hot_initial_states']
    assert len(per_scale)==len(cfg['scales']) and all(v==cfg['per_scale'] for v in per_scale.values())
    assert negative==m['negative_initial_hrr_pairs']
    result=dict(status='verified',preflight=args.preflight,pairs=offset,independent_initial_states=group_end,shards=len(m['parts']),
        all_shard_and_array_hashes_verified=True,all_array_rows_match_shards=True,all_hot_pairs_adjacent=True,
        per_scale_pairs=per_scale,hot_pairs=hot_rows,fresh_Cantera_cases=fresh_cases,
        fresh_reintegration_max_T_error_K=max_t_error,fresh_reintegration_max_Y_error=max_y_error,
        fresh_reintegration_tolerance_T_K=1e-4,fresh_reintegration_tolerance_Y=1e-8,
        negative_HRR_pairs_retained=negative,manifest_sha256=sha(root/'manifest.json'),verifier_sha256=sha(__file__))
    atomic_json(root/'verification.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
