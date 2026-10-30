import numpy as np
from kinetics.large_zhang_continue import plateau_step


def test_plateau_reduce_resume_cooldown_and_lr_floor():
    cfg=dict(patience=2,factor=.5,min_lr=1e-6,min_delta=1e-5,cooldown=1)
    start=dict(best=np.inf,wait=0,cooldown=0)
    values=[.1,.1,.1,.1,.09,.09,.09]+[.09]*40
    state=start;lr=1e-4;rates=[]
    for value in values:
        state,lr=plateau_step(state,value,lr,cfg);rates.append(lr)
    assert rates[:4]==[1e-4,1e-4,5e-5,5e-5]
    assert rates[4]==5e-5 and rates[6]==2.5e-5
    assert lr==1e-6 and state['best']==.09
    # Serializing scheduler state at an epoch boundary gives identical decisions.
    import json
    resumed=start;rate=1e-4
    for k,value in enumerate(values):
        resumed,rate=plateau_step(resumed,value,rate,cfg)
        if k==5:resumed=json.loads(json.dumps(resumed))
        assert rate==rates[k]
