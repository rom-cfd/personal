"""Exact pure-MSE state transfer and cumulative150-second member allowances."""
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
from .gamma_sweep_training import GammaTrainer
from .p_only_training import PressureTrainer, write_json
from .tp_phi_training import PhiTrainer
from .deeponet_long_training import new_monitor
from .study import tf


class MemberAllowance:
    def __init__(self, path, seconds=150):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists():
            self.state=json.loads(self.path.read_text())
            if self.state['seconds']!=seconds:raise ValueError('Cannot change member allowance')
        else:
            self.state=dict(seconds=seconds,consumed_seconds=0.,visits=[],open_visit=None)
            self._save()
        opened=self.state['open_visit']
        if opened:
            elapsed=min(opened['allowed_seconds'],max(0.,time.time()-opened['started_unix']))
            self.state['consumed_seconds']+=elapsed
            self.state['visits'].append(dict(**opened,charged_seconds=elapsed,recovered_interruption=True))
            self.state['open_visit']=None;self._save()
        self.started_perf=None

    def _save(self):
        temporary=self.path.with_name(self.path.name+f'.tmp_{time.time_ns()}')
        write_json(temporary,self.state);os.replace(temporary,self.path)

    @property
    def remaining(self):return max(0.,self.state['seconds']-self.state['consumed_seconds'])

    def begin(self,stage,outer_deadline=float('inf')):
        if self.state['open_visit'] is not None:raise ValueError('A visit is already active')
        allowed=min(self.remaining,max(0.,outer_deadline-time.perf_counter()))
        if allowed<=0:return None
        self.started_perf=time.perf_counter()
        self.state['open_visit']=dict(stage=stage,started_unix=time.time(),allowed_seconds=allowed)
        self._save();return self.started_perf+allowed

    def close(self):
        if self.started_perf is None:return
        elapsed=time.perf_counter()-self.started_perf;opened=self.state['open_visit']
        self.state['visits'].append(dict(**opened,charged_seconds=elapsed,recovered_interruption=False))
        self.state['consumed_seconds']+=elapsed;self.state['open_visit']=None
        self.started_perf=None;self._save()


def bridge_legacy(record,x,y,family):
    """Original restorer checks source/data/runtime; every optimizer/RNG value is copied."""
    cp=Path(record['checkpoint']);state=json.loads((cp/'state.json').read_text());config=state['signature']['config']
    original=(PhiTrainer if family=='tp_phi' else PressureTrainer).restore(cp,x,y,config,record['seed'])
    trainer=GammaTrainer(x,y,config,record['seed'],family,.1,predictor=original.predictor)
    before=[dict(name=v.name,shape=list(v.shape)) for v in original.optimizer.variables]
    after=[dict(name=v.name,shape=list(v.shape)) for v in trainer.optimizer.variables]
    if before!=after:raise ValueError('Legacy Adam layout differs')
    for a,b in zip(original.optimizer.variables,trainer.optimizer.variables):
        b.assign(a);np.testing.assert_array_equal(a.numpy(),b.numpy())
    np.testing.assert_array_equal(original.features.numpy(),trainer.features.numpy())
    np.testing.assert_array_equal(original.targets.numpy(),trainer.targets.numpy())
    trainer.rng.bit_generator.state=copy.deepcopy(original.rng.bit_generator.state)
    trainer.tf_rng=tf.random.Generator.from_state(original.tf_rng.state.numpy(),int(original.tf_rng.algorithm))
    trainer.update=original.update;trainer.carried_seconds=original.elapsed_seconds
    trainer.history=copy.deepcopy(original.history);trainer.full_train_history=copy.deepcopy(original.full_train_history)
    if trainer.update!=record['updates'] or int(trainer.optimizer.iterations.numpy())!=record['updates']:
        raise ValueError('Legacy origin counter differs from its record')
    return trainer


def restore_origin(record,x,y):
    if record['origin_kind']=='legacy_exact_checkpoint':return bridge_legacy(record,x,y,record['family'])
    state=json.loads((Path(record['checkpoint'])/'state.json').read_text())
    return GammaTrainer.restore(record['checkpoint'],x,y,state['signature']['config'],record['seed'],record['family'],record['gamma'])


def start_long_monitor(trainer,policy,run_sha256,origin):
    monitor=new_monitor(policy,origin_update=10000,origin_seconds=trainer.elapsed_seconds)
    monitor.update(long_run_sha256=run_sha256,origin_checkpoint=origin['checkpoint'],origin_kind=origin['origin_kind'])
    trainer.plateau_monitor=monitor


def restore_long(checkpoint,x,y,record,run_sha256):
    state=json.loads((Path(checkpoint)/'state.json').read_text())
    trainer=GammaTrainer.restore(checkpoint,x,y,state['signature']['config'],record['seed'],record['family'],record['gamma'])
    if trainer.plateau_monitor.get('long_run_sha256')!=run_sha256:
        raise ValueError('Long monitor belongs to another run')
    return trainer
