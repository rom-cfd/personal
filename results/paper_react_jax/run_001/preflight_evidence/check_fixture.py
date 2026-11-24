"""Exercise plotting, serialized scores and checkpoint reload before launch."""
import os
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
import sys,json,pickle,shutil,time
from pathlib import Path
sys.path.insert(0,'/workspace/personal')
import numpy as np
import jax
from jax import random
from jax.example_libraries import optimizers
from kinetics.paper_react_jax import evaluate,initial_params,save_checkpoint,sha,write
started=time.perf_counter();source=Path('/workspace/personal/results/paper_react_jax/run_001');follow=Path(__file__).parent;out=follow/'fixture_001'
out.mkdir();(out/'data').mkdir();(out/'report').mkdir();(out/'evaluations').mkdir()
cfg=json.loads((source/'config.json').read_text());cases=json.loads((source/'cases.json').read_text());new=[];meta=json.loads((source/'dataset.json').read_text())
for split in ['train','val','paper_test','fresh_test']:
 rows=[x for x in cases if x['split']==split];i=next((i for i,x in enumerate(rows) if split!='train' or x['phi']==1.),0);new.append(rows[i])
 with np.load(source/'data'/f'{split}.npz') as f:states=f['states'][i:i+1]
 p=out/'data'/f'{split}.npz';np.savez_compressed(p,states=states);meta['splits'][split]={'cases':1,'sha256':sha(p)}
write(out/'cases.json',new);write(out/'dataset.json',meta);shutil.copyfile(source/'preprocessing.json',out/'preprocessing.json')
write(out/'selection_frozen.json',dict(software_fixture=True,test_used_for_selection=False))
(out/'training.csv').write_text('update,mean_loss,lr\n1,1,.01\n2,.5,.01\n')
params=initial_params(cfg);zero=jax.tree_util.tree_map(lambda x:np.zeros_like(x),params)
evaluate(out,cfg,{'protocol_final':zero,'validation_selected':zero},time.perf_counter()+180)
assert json.loads((out/'verification.json').read_text())['NRMSE_recomputed']==80
init,_,get=optimizers.adam(.01);state=init(params);save_checkpoint(out/'checkpoint.pkl',state,random.PRNGKey(12),17)
with (out/'checkpoint.pkl').open('rb') as f:reload=pickle.load(f)
assert reload['update']==17
for a,b in zip(jax.tree_util.tree_leaves(params),jax.tree_util.tree_leaves(get(reload['optimizer']))):np.testing.assert_array_equal(a,b)
pre=json.loads((follow/'preflight.json').read_text());pre['elapsed_seconds']+=time.perf_counter()-started;pre['fixture']={'status':'passed','NRMSE_checks':80,'plot_cases':4,'zero_weight_model_not_chemistry_success':True,'exact_checkpoint_reload':True}
write(follow/'preflight.json',pre)
print(json.dumps({'status':'passed','fixture_seconds':time.perf_counter()-started,'NRMSE_checks':80}),flush=True)
