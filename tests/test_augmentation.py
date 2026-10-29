import hashlib
import cantera as ct
import numpy as np
from kinetics.data import generate,mechanism
from kinetics.augmentation import augment
from kinetics.physics import element_matrix


def test_augmented_labels_are_reintegrated_and_training_only(tmp_path):
    source=tmp_path/'original.npz';generate(source,trajectories=10,steps=4,seed=17)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    target=tmp_path/'aug.npz';meta=augment(source,target,count=6,seed=23,level='mild')
    original=np.load(source);data=np.load(target)
    assert set(data['parent'][:,0]) <= set(np.flatnonzero(original['split']=='train'))
    assert float(data['dt'])==float(original['dt'])==2e-6
    assert meta['labels'].startswith('Cantera independently')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    gas=mechanism();a=element_matrix(gas)
    for x,y in zip(data['x'],data['y']):
        assert x[1:10].min()>=0 and y[1:].min()>=0
        np.testing.assert_allclose(a@x[1:10],a@y[1:],atol=1e-9)
        gas.TPY=x[0],x[-1],x[1:10]
        reactor=ct.IdealGasConstPressureReactor(gas,clone=True)
        sim=ct.ReactorNet([reactor]);sim.rtol=1e-10;sim.atol=1e-20
        sim.advance(float(data['dt']))
        np.testing.assert_allclose(y,np.r_[reactor.T,reactor.phase.Y],rtol=1e-8,atol=1e-12)
