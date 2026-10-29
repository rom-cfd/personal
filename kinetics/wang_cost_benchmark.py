"""CPU cost estimate only: paper-sized GELU/L1 network, not a reproduction run.

Normalization, initialization and framework choices in this benchmark do not
resolve the missing 2025 training code. No fitted model is saved or activated.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .study import tf


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/wang_implementation_audit/cost_benchmark.json'))
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing benchmark output')
    with np.load('results/long10ms_data/training_pairs.npz') as data:
        x,y=data['x'],data['y']
    # Hypothetical paper-equation variant: direct species G_next, raw T/P.
    features=np.c_[x[:,0],x[:,-1]/101325.,(x[:,1:10]**.1-1)/.1]
    target=(y[:,1:]**.1-1)/.1
    features=(features-features.mean(axis=0))/np.maximum(features.std(axis=0),1e-14)
    target=(target-target.mean(axis=0))/np.maximum(target.std(axis=0),1e-14)
    tf.keras.utils.set_random_seed(11)
    net=tf.keras.Sequential([tf.keras.Input((11,)),
        tf.keras.layers.Dense(1600,activation=tf.nn.gelu),
        tf.keras.layers.Dense(800,activation=tf.nn.gelu),
        tf.keras.layers.Dense(400,activation=tf.nn.gelu),tf.keras.layers.Dense(9)])
    optimizer=tf.keras.optimizers.Adam(1e-4,epsilon=1e-8)
    @tf.function(input_signature=[tf.TensorSpec([None,11],tf.float32),tf.TensorSpec([None,9],tf.float32)])
    def step(a,b):
        with tf.GradientTape() as tape:
            loss=tf.reduce_mean(tf.abs(net(a,training=True)-b))
        optimizer.apply_gradients(zip(tape.gradient(loss,net.trainable_variables),net.trainable_variables))
        return loss
    rows=[];started=time.perf_counter()
    for size,repeats in [(1024,5),(len(x)%1024,3),(len(x),3)]:
        a=tf.constant(features[:size],tf.float32);b=tf.constant(target[:size],tf.float32)
        step(a,b).numpy()  # compile / warmup
        measured=[]
        for _ in range(repeats):
            if time.perf_counter()-started>90:raise TimeoutError('Short benchmark cap reached')
            begin=time.perf_counter();loss=float(step(a,b).numpy());measured.append(time.perf_counter()-begin)
        rows.append(dict(batch=size,median_seconds=float(np.median(measured)),loss_last=loss,repeats=repeats))
    costs={r['batch']:r['median_seconds'] for r in rows}
    full,remainder=divmod(len(x),1024)
    per_model=2500*(full*costs[1024]+costs[remainder])+2500*costs[len(x)]
    result=dict(purpose='resource estimate only; not implementation-fidelity validation or trained-model evaluation',
        framework='TensorFlow CPU',parameters=net.count_params(),layers=[11,1600,800,400,9],
        activation='exact TensorFlow GELU; author legacy settings instead say mygelu',
        loss='L1; hypothetical train-standardized direct G_next',training_pairs=len(x),
        batches=rows,measured_seconds=time.perf_counter()-started,
        paper_stage1_epochs=2500,paper_stage1_batch=1024,paper_stage2_epochs=2500,paper_stage2_batch=131072,
        estimated_single_model_hours=per_model/3600,estimated_100_member_hours=100*per_model/3600,
        estimate_limits='Linear extrapolation, current CPU/backend only; excludes I/O, evaluation, validation and contention; stage2 uses all 8815 available points',
        full_training_launched=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
