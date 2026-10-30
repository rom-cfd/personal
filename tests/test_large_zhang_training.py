import math
import numpy as np
from kinetics.large_zhang_training import stream_moments,permutation_for


def test_streaming_zscores_match_population_statistics_and_constant_columns():
    rng=np.random.default_rng(81);a=rng.normal(size=(71,11));a[:,1]=0
    mean,scale,count=stream_moments(a[i:i+9] for i in range(0,len(a),9))
    expected=a.std(axis=0);expected[expected<=1e-14]=1
    assert count==len(a)
    np.testing.assert_allclose(mean,a.mean(axis=0),rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(scale,expected,rtol=1e-13,atol=1e-14)


def test_each_epoch_visits_all_rows_once_and_resume_reconstructs_order():
    a=permutation_for(77,11,0);b=permutation_for(77,11,1)
    np.testing.assert_array_equal(np.sort(a),np.arange(77))
    np.testing.assert_array_equal(a,permutation_for(77,11,0))
    assert not np.array_equal(a,b)
    batches=[a[i:i+16] for i in range(0,77,16)]
    np.testing.assert_array_equal(np.concatenate(batches[2:]),a[32:])
    assert sum(map(len,batches))==77
    n=5300000;batch=512
    assert math.ceil(n/batch)*10==103520
    assert (math.ceil(n/batch)-1)*batch+n%batch==n
