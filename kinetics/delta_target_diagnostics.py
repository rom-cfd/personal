"""Per-output validation loss and signed-delta roundoff amplification audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .delta_model import DeltaPredictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/delta100_raw'))
    root=parser.parse_args().root;out=root/'evaluation'
    cfg=json.loads((root/'config.json').read_text())
    frozen=json.loads((out/'selection_frozen.json').read_text())
    model=DeltaPredictor.load(frozen['selected_path'])
    with np.load(cfg['training_pairs']) as data:
        delta_train=data['y']-data['x'][:,:10]
    encoded=model.encode_delta(delta_train)
    with np.load(cfg['source']) as data:
        ids=np.flatnonzero(data['split']=='val');states=data['states'][ids];pressures=data['pressures'][ids]
        names=['T',*map(str,data['species'])]
    n=states.shape[1]-1
    x=np.c_[states[:,:-1].reshape(-1,10),np.repeat(pressures,n)]
    delta=states[:,1:].reshape(-1,10)-x[:,:10]
    error=np.zeros(10)
    for begin in range(0,len(x),2048):
        xx=x[begin:begin+2048]
        predicted=model.model(model.features(xx).astype('float32'),training=False).numpy()
        target=model.output_norm.forward(model.encode_delta(delta[begin:begin+2048]))
        error+=np.sum((predicted-target)**2,axis=0)
    mse=error/len(x)
    result=pd.DataFrame(dict(channel=names,physical_max_abs_delta_train=np.max(np.abs(delta_train),axis=0),
        physical_std_delta_train=np.std(delta_train,axis=0),encoded_std_delta_train=np.std(encoded,axis=0),
        zero_delta_fraction_train=np.mean(delta_train==0,axis=0),
        validation_standardized_MSE=mse,validation_MSE_share=mse/mse.sum(),
        fraction_abs_delta_le_1e_minus12_val=np.mean(np.abs(delta)<=1e-12,axis=0)))
    result.to_csv(out/'target_diagnostics.csv',index=False)
    ranking=pd.read_csv(out/'validation_member_ranking.csv').set_index('seed')
    np.testing.assert_allclose(mse.mean(),ranking.loc[frozen['selected_seed'],'transformed_mse'],rtol=1e-6,atol=1e-7)
    print(result.to_string(index=False))


if __name__=='__main__':main()
