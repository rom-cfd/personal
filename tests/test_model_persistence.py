import numpy as np
from kinetics.study import Predictor


def test_decimal_names_and_seed_models_are_distinct(tmp_path):
    rng=np.random.default_rng(1)
    y=rng.dirichlet(np.ones(9),size=8)
    state=np.c_[rng.uniform(1000,1500,8),y]
    x=np.c_[state,np.full(8,101325.)]
    cfg=dict(hidden_layers=[4],activation='tanh',learning_rate=.001)
    paths=[]
    for gamma,seed in ((.1,11),(.1,22),(.3,11)):
        p=Predictor(dict(kind='boxcox',gamma=gamma),x,state,cfg,seed)
        # Nonzero weights ensure a reload really tests learned output persistence.
        weights=p.model.get_weights()
        weights[-1]=np.full_like(weights[-1],.001*seed)
        p.model.set_weights(weights)
        path=tmp_path/f'boxcox_g{gamma}_seed{seed}'
        p.save(path)
        loaded=Predictor.load(path)
        np.testing.assert_allclose(p.predict(x),loaded.predict(x),rtol=1e-12,atol=1e-12,equal_nan=True)
        paths.append(str(path)+'.keras')
    assert len(set(paths))==3
    assert len(list(tmp_path.glob('*.keras')))==3
