"""Independent source-function parity and a synthetic throughput probe."""
import os
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
import sys,time,json,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,'/workspace/personal')
import numpy as np
import jax,jax.numpy as jnp
from jax import random
from jax.nn import relu
from kinetics.paper_react_jax import initialize,network,compiled_training,initial_params,write

started=time.perf_counter();repo=Path('/workspace/personal');follow=Path(__file__).parent
result=subprocess.run(['/workspace/kinetics-venv/bin/python','-m','pytest','-q','tests/test_paper_react_jax.py'],cwd=repo,capture_output=True,text=True)
(follow/'tests.log').write_text(result.stdout+result.stderr)
assert result.returncode==0,result.stdout+result.stderr
nb=json.loads(Path('/workspace/literature_sources/wang_pinned/PI_DeepONet_stiff.ipynb').read_text())
cell=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and 'def modified_MLP' in ''.join(c['source']))
namespace={'np':jnp,'random':random,'relu':relu};exec(cell,namespace)
init,apply=namespace['modified_MLP']([5,80,80,200],activation=jnp.tanh)
p=init(random.PRNGKey(1234));ours=(p[0],(p[1],p[2]),(p[3],p[4]));x=random.normal(random.PRNGKey(2),(13,5))
expected=apply(p,x);actual=network(ours,x)
np.testing.assert_allclose(expected,actual,rtol=2e-6,atol=2e-6)
grad1=jax.grad(lambda a:jnp.mean(apply(p,a)**2))(x)
grad2=jax.grad(lambda a:jnp.mean(network(ours,a)**2))(x)
np.testing.assert_allclose(grad1,grad2,rtol=2e-5,atol=2e-6)
ours_init=initialize([5,80,80,200],random.PRNGKey(1234))
init_diffs=[float(np.max(np.abs(a-b))) for a,b in zip(jax.tree_util.tree_leaves(p),jax.tree_util.tree_leaves((ours_init[0],*ours_init[1],*ours_init[2])))]
assert max(init_diffs)<1e-6
cfg=json.loads((repo/'configs/paper_react_2023_jax.json').read_text())
synthetic=random.uniform(random.PRNGKey(0),(25,5001,5),minval=-1,maxval=1)
opt_init,get,loss,chunk=compiled_training(cfg,synthetic,10)
state=opt_init(initial_params(cfg));key=random.PRNGKey(1234)
t=time.perf_counter();chunk.lower(state,key,jnp.asarray(0)).compile();compile_seconds=time.perf_counter()-t
timings=[]
for i in range(3):
 t=time.perf_counter();(_,_,_),v=chunk(state,key,jnp.asarray(0));np.asarray(v);timings.append(time.perf_counter()-t)
seconds_per_update=float(np.median(timings))/10
checks={
 'status':'passed','tests':'5 passed','author_forward_max_abs_diff':float(np.max(np.abs(expected-actual))),
 'author_input_gradient_max_abs_diff':float(np.max(np.abs(grad1-grad2))),
 'canonical_initialization_max_abs_diff':max(init_diffs),
 'synthetic_updates':30,'synthetic_compile_seconds':compile_seconds,'seconds_per_update':seconds_per_update,
 'estimated_200k_training_seconds':seconds_per_update*200000,
 'benchmark_is_not_chemistry_training':True,'elapsed_seconds':time.perf_counter()-started,
 'jax':jax.__version__,'jaxlib':__import__('jaxlib').__version__,
 'source_sha256':{name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in ['kinetics/paper_react_jax.py','scripts/supervise_paper_react.py','scripts/monitor_paper_react.py','tests/test_paper_react_jax.py','configs/paper_react_2023_jax.json']}}
write(follow/'preflight.json',checks);print(json.dumps(checks),flush=True)
