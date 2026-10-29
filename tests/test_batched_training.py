import numpy as np
from kinetics.batched_training import train_group,clip_member_gradients
from kinetics.refinement import train_updates
from kinetics.study import tf


def test_clipping_is_per_member_not_across_the_committee():
    g=tf.constant([[3.,4.],[0.,10.]])
    np.testing.assert_allclose(clip_member_gradients([g])[0].numpy(),[[3,4],[0,5]],atol=1e-6)


def test_grouped_training_matches_independent_updates_and_rng():
    rng=np.random.default_rng(18)
    states=np.c_[rng.uniform(1000,1500,20),rng.dirichlet(np.ones(9),20)]
    x=np.c_[states,np.full(20,101325.)]
    y=states.copy();y[:,0]+=20;y[:,1:]=rng.dirichlet(np.ones(9),20)
    cfg=dict(hidden_layers=[4,4],activation='tanh',learning_rate=.001,batch_size=8)
    spec=dict(kind='boxcox',gamma=.1)
    grouped,_,_=train_group(spec,[11,22],30,x,y,cfg)
    for seed,group_member in zip([11,22],grouped):
        independent,_,_=train_updates(spec,seed,30,x,y,cfg,float('inf'))
        for a,b in zip(group_member.model.get_weights(),independent.model.get_weights()):
            np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-5)
