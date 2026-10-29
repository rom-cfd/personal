"""Optional equal-weight physical-output ensemble; reports its 3x ANN cost."""
import json
from pathlib import Path
import numpy as np
from .study import Predictor


class MeanPredictor:
    def __init__(self,paths):
        self.paths=[str(p) for p in paths]
        self.members=[Predictor.load(p) for p in paths]

    def predict(self,x):
        return np.mean([m.predict(x) for m in self.members],axis=0)

    def predict_guarded(self,x):
        predictions=[]
        for net in self.members:
            delta=net.model(net.features(x).astype('float32'),training=False).numpy().astype('float64')
            z=net.norm.inverse(net.norm.forward(net.encode_state(x[:,:10]))+delta)
            upper=net.transform.forward(np.ones((1,9)))
            y=net.transform.inverse(np.clip(z[:,1:],0,upper))
            predictions.append(np.c_[x[:,0],y])
        return np.mean(predictions,axis=0)

    def save(self,path):
        Path(path).write_text(json.dumps(dict(kind='equal_weight_mean',members=self.paths),indent=2))


def load_candidate(info):
    if info.get('ensemble',False):
        return MeanPredictor(info['members'])
    return Predictor.load(info['path'])
