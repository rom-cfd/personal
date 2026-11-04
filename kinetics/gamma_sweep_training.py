"""Gamma-only preprocessing changes with the original architectures and pure MSE."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from .p_only_training import PressureTrainer, signature, sha256, write_json
from .deeponet_long_training import SerialSteps
from .tp_phi_training import PhiSerialSteps
from .soft_constraint_training import LOADERS, FAMILIES
from .transforms import Transform, Standardizer
from .study import tf


def gamma_tag(gamma):
    return f'gamma_{float(gamma):g}'


def make_predictor(family, gamma, x, y, config, seed):
    if family not in FAMILIES or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('Known family and positive finite gamma required')
    # Instantiate the unchanged architecture and its exact paired initialization.
    # Legacy P-only/phi constructors require .1; no update is performed here.
    p = LOADERS[family](dict(kind='boxcox', gamma=.1, epsilon=1e-8), x, y, config, seed)
    if gamma != .1:
        p.spec = dict(kind='boxcox', gamma=float(gamma), epsilon=1e-8)
        p.transform = Transform('boxcox', gamma, 1e-8)
        a, b = p.encode_state(x[:, :10]), p.encode_state(y)
        p.norm = Standardizer().fit(np.r_[a, b])
        p.input_norm = Standardizer().fit(np.c_[p.norm.forward(a), np.log(x[:, -1])])
        p.train_features = p.features(x).astype('float32')
        p.train_target = (p.norm.forward(b) - p.norm.forward(a)).astype('float32')
    if p.transform.gamma != gamma:
        raise ValueError('Requested gamma was not applied')
    return p


def gamma_signature(x, y, config, family, gamma):
    result = signature(x, y, config)
    result.update(family=family, gamma=float(gamma), loss='transformed_residual_MSE_only',
        gamma_source_sha256={str(p): sha256(p) for p in sorted(Path('kinetics').glob('gamma_sweep_*.py'))},
        serial_engine_sha256=sha256(Path('kinetics/deeponet_long_training.py')),
        phi_source_sha256={str(p): sha256(p) for p in sorted(Path('kinetics').glob('tp_phi_*.py'))})
    return result


def weights_hash(p):
    h = hashlib.sha256()
    for value in p.model.get_weights():
        h.update(str((value.shape, value.dtype.str)).encode()); h.update(value.tobytes())
    return h.hexdigest()


class GammaTrainer(PressureTrainer):
    def __init__(self, x, y, config, seed, family, gamma, predictor=None):
        p = predictor if predictor is not None else make_predictor(family, gamma, x, y, config, seed)
        super().__init__(x, y, config, seed, predictor=p)
        self.family, self.gamma = family, float(gamma)
        self.signature = gamma_signature(x, y, config, family, gamma)
        self.plateau_monitor = dict(enabled=False, reason='10000-update ceiling with member wall cap; no plateau selection',
            timing=dict(rng_seconds=0., optimizer_seconds=0., full_train_seconds=0., trace_seconds=0., checkpoint_seconds=0.))
        self.serial = (PhiSerialSteps if family == 'tp_phi' else SerialSteps)(self)

    def advance(self, target_update, deadline=float('inf')):
        if type(target_update) is not int or target_update < self.update:
            raise ValueError('Target update cannot move backwards')
        return self.serial.advance(target_update, deadline)

    def full_train_loss(self, deadline=float('inf')):
        return self.serial.full_train_loss(deadline)

    @classmethod
    def restore(cls, directory, x, y, config, seed, family, gamma):
        directory = Path(directory)
        marker = json.loads((directory / 'complete.json').read_text())
        required = {'model.keras', 'model.json', 'adam.npz', 'state.json', 'history.csv', 'full_train_history.csv'}
        if set(marker['output_sha256']) != required:
            raise ValueError('Incomplete gamma checkpoint')
        for name, expected in marker['output_sha256'].items():
            if Path(name).name != name or sha256(directory / name) != expected:
                raise ValueError('Gamma checkpoint changed')
        state = json.loads((directory / 'state.json').read_text())
        if state['format_version'] != 1 or state['seed'] != seed or state['signature'] != gamma_signature(x, y, config, family, gamma):
            raise ValueError('Gamma checkpoint provenance mismatch')
        p = LOADERS[family].load(directory / 'model')
        if p.transform.gamma != gamma:
            raise ValueError('Saved preprocessing has a different gamma')
        self = cls(x, y, config, seed, family, gamma, predictor=p)
        actual = [dict(name=v.name, shape=list(v.shape)) for v in self.optimizer.variables]
        if actual != state['optimizer_variables']:
            raise ValueError('Adam layout changed')
        with np.load(directory / 'adam.npz') as f:
            if len(f.files) != len(actual):
                raise ValueError('Incomplete Adam state')
            for i, v in enumerate(self.optimizer.variables):
                v.assign(f[f'v{i}'])
        if int(self.optimizer.iterations.numpy()) != state['update'] or state['optimizer_iteration'] != state['update']:
            raise ValueError('Adam/update mismatch')
        self.rng.bit_generator.state = state['rng_state']
        self.tf_rng = tf.random.Generator.from_state(state['tf_rng_state'], state['tf_rng_algorithm'])
        self.update, self.carried_seconds = state['update'], state['elapsed_seconds']
        self.history, self.full_train_history = state['history'], state['full_train_history']
        self.plateau_monitor = state['plateau_monitor']
        return self
