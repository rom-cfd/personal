"""Cross-check the independent model against pinned upstream operations and TF."""
import argparse
import ast
import copy
import os
from pathlib import Path
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/kinetics-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/kinetics-cache')

import numpy as np
import torch
from kinetics.flex_model import FlexDeepONet, KerasAdam, author_input_range
from kinetics.flex_data import AUTHOR, AUTHOR_SHA, atomic, sha, stamp


def upstream_method(file, cls, method, namespace):
    tree=ast.parse(Path(file).read_text())
    node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    fn=copy.deepcopy(next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name==method))
    fn.decorator_list=[]
    code=ast.Module(body=[fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(code),str(file),'exec'),namespace)
    return namespace[method]


def verify(output):
    import tensorflow as tf
    start=time.monotonic()
    torch.set_num_threads(2)
    tf.keras.backend.set_floatx('float64')
    tf.keras.backend.set_epsilon(1e-20)
    torch.manual_seed(1711)
    rng=np.random.default_rng(1711)
    x=np.zeros((9,19));x[:,0]=rng.uniform(1000,2000,9);x[:,1]=rng.uniform(.01,.15,9)
    x[:,4]=rng.uniform(.1,.2,9);x[:,18]=1-x[:,1]-x[:,4]
    span=author_input_range(x)
    model=FlexDeepONet(span,np.geomspace(1e-10,1000,19))
    xx=torch.tensor(x,dtype=torch.float64)
    tt=torch.logspace(-32,0,9,dtype=torch.float64)
    tx=tf.constant(x,dtype=tf.float64);time_tf=tf.constant(tt.numpy(),dtype=tf.float64)
    namespace=dict(tf=tf,np=np)
    prenet=upstream_method(AUTHOR/'romnet/architecture/deeponet.py','PreNet','_merge_function',namespace)
    trans=upstream_method(AUTHOR/'romnet/architecture/deeponet.py','InputTransLayer','call',namespace)
    normal=upstream_method(AUTHOR/'romnet/architecture/building_blocks/normalization.py','CustomNormalization','call',namespace)
    class Normal:
        _standardize_inputs=staticmethod(lambda x:x)
        compute_dtype=tf.float64
        centering='none';scaling='range'
        max_vals=tf.constant(np.maximum(1.,x.max(0)),tf.float64)
        min_vals=tf.constant(np.minimum(1.,x.min(0)),tf.float64)
    class Trans:
        NVars=1;indxs=[0];f='log'
    scaled=normal(Normal(),tx)
    scale_error=float(np.max(np.abs(scaled.numpy()-x/span)))
    variables={name:tf.Variable(p.detach().numpy(),dtype=tf.float64) for name,p in model.named_parameters()}
    def mlp(value,network,index):
        for layer in range(4):
            value=value@variables[f'{network}.weights.{layer}'][index]+variables[f'{network}.biases.{layer}'][index]
            if layer<3:value=tf.math.tanh(value)
        return value
    def forward():
        stretch=tf.math.softplus(mlp(scaled,'stretch',0))
        parts=[]
        for i in range(19):
            branch=mlp(scaled,'branch',i)
            coord=prenet(None,[time_tf[:,None],{'Stretch':stretch[:,i:i+1]}])
            coord=trans(Trans(),coord)
            trunk=mlp(coord,'trunk',i)
            parts.append(tf.reduce_sum(branch[:,:8]*trunk,axis=1)+branch[:,8])
        return tf.stack(parts,axis=1)
    def reg():
        return tf.add_n([tf.constant(1e-12,tf.float64)*(tf.reduce_sum(tf.abs(v))+tf.reduce_sum(tf.square(v)))
            for name,v in variables.items() if '.weights.' in name or not name.endswith('.3')])
    target=rng.normal(size=(9,19))
    target_tf=tf.constant(target,tf.float64)
    optimizer=KerasAdam(model.parameters())
    # Keras3 defaults its scalar hyperparameters to float32 even for float64
    # networks. Explicit float64 matches the pinned ROMNet/TF2.8 floatx policy.
    tf_opt=tf.keras.optimizers.Adam(learning_rate=lambda:tf.constant(.001,tf.float64),
        beta_1=tf.constant(.9,tf.float64),beta_2=tf.constant(.999,tf.float64),epsilon=1e-7)
    errors=[]
    for step in range(3):
        optimizer.zero_grad()
        out=model(xx,tt);loss=(out-torch.tensor(target)).square().mean()+model.regularization()
        loss.backward()
        with tf.GradientTape() as tape:
            tf_out=forward();tf_loss=tf.reduce_mean(tf.square(tf_out-target_tf))+reg()
        tf_grads=tape.gradient(tf_loss,list(variables.values()))
        out_error=float(np.max(np.abs(tf_out.numpy()-out.detach().numpy())))
        grad_error=max(float(np.max(np.abs(g.numpy()-p.grad.numpy()))) for g,p in zip(tf_grads,model.parameters()))
        optimizer.step();tf_opt.apply_gradients(zip(tf_grads,variables.values()))
        update_error=max(float(np.max(np.abs(v.numpy()-p.detach().numpy()))) for v,p in zip(variables.values(),model.parameters()))
        errors.append(dict(step=step+1,max_output_abs=out_error,max_gradient_abs=grad_error,max_weight_after_Adam_abs=update_error))
    assert scale_error<1e-13 and max(e['max_output_abs'] for e in errors)<2e-8,errors
    assert max(e['max_gradient_abs'] for e in errors)<2e-8,errors
    assert max(e['max_weight_after_Adam_abs'] for e in errors)<2e-8,errors
    # Full-size timed train updates, after warmup. This is software benchmarking,
    # with synthetic targets, never a selected chemical model or held-out test.
    torch.manual_seed(1712)
    bench=FlexDeepONet();opt=KerasAdam(bench.parameters())
    bx=torch.randn(256,19,dtype=torch.float64);bt=torch.logspace(-32,0,256,dtype=torch.float64)
    for i in range(25):
        begin=time.perf_counter()
        opt.zero_grad();loss=bench(bx,bt).square().mean()+bench.regularization();loss.backward();opt.step()
        if i==4:mark=time.perf_counter()
    seconds=(time.perf_counter()-mark)/20
    record=dict(status='passed',at=stamp(),author_commit=AUTHOR_SHA,
        actual_source_operations_executed=['CustomNormalization.call','PreNet._merge_function','InputTransLayer.call'],
        normalization_max_abs=scale_error,three_updates=errors,
        torch_version=torch.__version__,tensorflow_version=tf.__version__,dtype='float64',threads=2,
        full_model_parameters=sum(p.numel() for p in bench.parameters()),
        synthetic_batch256_seconds_per_update=seconds,wall_seconds=time.monotonic()-start,
        note='Software fidelity only, not evidence of chemical prediction accuracy')
    atomic(Path(output),record)
    print(record)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    verify(parser.parse_args().output)
