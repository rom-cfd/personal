"""Temperature/pressure-query branch/trunk residual predictor, at fixed dt=2 microseconds.

This is a finite-dimensional DeepONet-style surrogate, not evidence of learning
an operator on arbitrary input functions. It shares the legacy ANN's transforms.
"""
import json
import time
from pathlib import Path

import numpy as np
from .study import Predictor, tf
from .transforms import Transform, Standardizer


@tf.keras.utils.register_keras_serializable(package='kinetics')
class BranchTrunkProduct(tf.keras.layers.Layer):
    """Ten composition coefficient vectors contracted with a T/P basis."""

    def build(self, input_shape):
        self.bias = self.add_weight(name='output_bias', shape=(10,), initializer='zeros')

    def call(self, inputs):
        branch, trunk = inputs
        return tf.einsum('bkl,bl->bk', branch, trunk) + self.bias


class ThermoDeepONet(Predictor):
    family = 'temperature_pressure_deeponet'

    def __init__(self, spec, x, y, config, seed):
        if spec['kind'] != 'boxcox' or spec.get('species', False):
            raise ValueError('This comparison uses the existing fixed-epsilon Box-Cox baseline')
        if config['input_partition'] != 'current_Y9__current_T_logP':
            raise ValueError('Branch requires current Y9; trunk requires current T and log(P)')
        self.spec = spec
        self.config = config
        self.transform = Transform('boxcox', spec.get('gamma', .1), spec.get('epsilon', 1e-8))
        a = self.encode_state(x[:, :10])
        b = self.encode_state(y)
        self.norm = Standardizer().fit(np.r_[a, b])
        self.input_norm = Standardizer().fit(np.c_[self.norm.forward(a), np.log(x[:, -1])])
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        inputs = tf.keras.Input((11,), name='standardized_state_and_log_pressure')
        branch = inputs[:, 1:10]
        for i, width in enumerate(config['branch_layers']):
            branch = tf.keras.layers.Dense(width, activation='tanh', name=f'branch_{i}')(branch)
        latent = config['latent_width']
        branch = tf.keras.layers.Dense(10 * latent, kernel_initializer='zeros',
                                       bias_initializer='zeros', name='branch_coefficients')(branch)
        branch = tf.keras.layers.Reshape((10, latent))(branch)
        trunk = tf.keras.layers.Concatenate(name='current_temperature_and_log_pressure')(
            [inputs[:, :1], inputs[:, 10:]])
        for i, width in enumerate(config['trunk_layers']):
            trunk = tf.keras.layers.Dense(width, activation='tanh', name=f'trunk_{i}')(trunk)
        # Nonzero trunk initialization lets the initially zero branch learn.
        trunk = tf.keras.layers.Dense(latent, name='thermo_basis')(trunk)
        residual = BranchTrunkProduct(name='component_residual')([branch, trunk])
        self.model = tf.keras.Model(inputs, residual, name=self.family)
        self.optimizer = tf.keras.optimizers.Adam(config['learning_rate'])
        self.train_features = self.features(x).astype('float32')
        self.train_target = (self.norm.forward(b) - self.norm.forward(a)).astype('float32')

    def save(self, path):
        path = Path(path)
        if any(Path(str(path) + suffix).exists() for suffix in ('.keras', '.json')):
            raise FileExistsError('Never overwrite a saved model')
        super().save(path)
        info_path = Path(str(path) + '.json')
        info = json.loads(info_path.read_text())
        info.update(family=self.family, architecture={k: self.config[k] for k in
                    ('branch_layers', 'trunk_layers', 'latent_width', 'input_partition')}, dt=2e-6)
        info_path.write_text(json.dumps(info, indent=2))

    @classmethod
    def load(cls, path):
        info = json.loads(Path(str(path) + '.json').read_text())
        if info.get('family') != cls.family or info.get('dt') != 2e-6:
            raise ValueError('Not a fixed-2us temperature/pressure-query model')
        self = super().load(path)
        self.config = info['architecture']
        return self


def train_member(spec, seed, updates, x, y, config, deadline=float('inf')):
    """Independent Adam/RNG, identical batch/update/lr/clip rules to the ANN.

    Only completed members are resumable; this does not restore optimizer state
    midway through a member. Callers must supply an explicitly approved budget.
    """
    if updates < 1 or config['batch_size'] < 1:
        raise ValueError('Positive updates and batch size required')
    started = time.perf_counter()
    if started >= deadline:
        raise TimeoutError('Training budget exhausted before initialization')
    p = ThermoDeepONet(spec, x, y, config, seed)
    features = tf.constant(p.train_features)
    targets = tf.constant(p.train_target)

    @tf.function(input_signature=[tf.TensorSpec([None], tf.int32), tf.TensorSpec([], tf.float32)])
    def step(ids, learning_rate):
        p.optimizer.learning_rate.assign(learning_rate)
        with tf.GradientTape() as tape:
            error = p.model(tf.gather(features, ids), training=True) - tf.gather(targets, ids)
            loss = tf.debugging.check_numerics(tf.reduce_mean(error ** 2), 'nonfinite training loss')
        variables = p.model.trainable_variables
        grads, _ = tf.clip_by_global_norm(tape.gradient(loss, variables), 5.)
        p.optimizer.apply_gradients(zip(grads, variables))
        return loss

    rng = np.random.default_rng(seed)
    history = []
    for k in range(updates):
        if time.perf_counter() >= deadline:
            raise TimeoutError('Training budget exhausted; incomplete member not registered')
        lr = config['learning_rate'] * (1 if k < updates * .6 else .3 if k < updates * .9 else .1)
        ids = rng.integers(0, len(x), config['batch_size'], dtype=np.int32)
        loss = step(ids, np.float32(lr))
        if (k == 0 or (k + 1) % 50 == 0 or k == updates - 1 or
                k in (int(updates * .6), int(updates * .9))):
            history.append(dict(seed=seed, update=k + 1, loss=float(loss.numpy()), learning_rate=lr))
        if (k + 1) % 1000 == 0:
            print('MEMBER_PROGRESS', seed, 'update', k + 1, 'loss', float(loss.numpy()),
                  'learning_rate', lr, 'seconds', round(time.perf_counter() - started, 2), flush=True)
    if time.perf_counter() >= deadline:
        raise TimeoutError('Training budget exhausted after update')
    return p, time.perf_counter() - started, history
