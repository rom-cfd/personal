"""Independent sequential Adam training with immutable, verified checkpoints.

No dropout or stochastic TensorFlow operations are used. Minibatch randomness
comes exclusively from the member's saved NumPy Generator. An owned TensorFlow
Generator is also saved for explicit RNG provenance; ambient RNGs are not used.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
from .p_only_deeponet import PressureDeepONet
from .study import tf


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def learning_rate(update):
    """Absolute one-based update; extension/resume never moves the LR boundaries."""
    if type(update) is not int or update < 1:
        raise ValueError('Positive integer update required')
    return .001 if update <= 6000 else .0003 if update <= 9000 else .0001


def signature(x, y, config):
    def array_hash(a):
        a = np.ascontiguousarray(a)
        h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
        return h.hexdigest()
    package = Path(__file__).parent
    return dict(config=json.loads(json.dumps(config)), x_sha256=array_hash(x), y_sha256=array_hash(y),
        source_sha256={name: sha256(package / name) for name in
            ('p_only_deeponet.py', 'p_only_training.py', 'deeponet.py', 'study.py', 'transforms.py')},
        versions={name: version(name) for name in ('tensorflow_cpu', 'keras', 'numpy')},
        onednn=os.environ.get('TF_ENABLE_ONEDNN_OPTS'),
        tensorflow_intra_threads=tf.config.threading.get_intra_op_parallelism_threads(),
        tensorflow_inter_threads=tf.config.threading.get_inter_op_parallelism_threads())


class PressureTrainer:
    def __init__(self, x, y, config, seed, predictor=None):
        self.started = time.perf_counter()
        if type(config['batch_size']) is not int or config['batch_size'] < 1:
            raise ValueError('Positive integer batch size required')
        if config.get('learning_rate', .001) != .001:
            raise ValueError('The agreed LR schedule starts at .001')
        self.config, self.seed = dict(config), int(seed)
        self.signature = signature(x, y, config)
        self.predictor = predictor or PressureDeepONet(dict(kind='boxcox', gamma=.1), x, y, config, seed)
        p = self.predictor
        a, b = p.encode_state(x[:, :10]), p.encode_state(y)
        self.features = tf.constant(p.features(x).astype('float32'))
        self.full_targets = p.norm.forward(b)-p.norm.forward(a)
        self.targets = tf.constant(self.full_targets.astype('float32'))
        self.optimizer = tf.keras.optimizers.Adam(.001)
        self.optimizer.build(p.model.trainable_variables)
        self.rng = np.random.default_rng(seed)
        self.tf_rng = tf.random.Generator.from_seed(seed)
        self.update, self.carried_seconds = 0, 0.
        self.history, self.full_train_history = [], []
        # Long-training plateau policy must be agreed before it is enabled.
        self.plateau_monitor = dict(enabled=False, reason='Fixed 10k baseline, no plateau selection')

        @tf.function(input_signature=[tf.TensorSpec([None], tf.int32), tf.TensorSpec([], tf.float32)])
        def step(ids, lr):
            self.optimizer.learning_rate.assign(lr)
            with tf.GradientTape() as tape:
                error = p.model(tf.gather(self.features, ids), training=True)-tf.gather(self.targets, ids)
                loss = tf.debugging.check_numerics(tf.reduce_mean(error**2), 'nonfinite loss')
            variables = p.model.trainable_variables
            grads = tape.gradient(loss, variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.)
            self.optimizer.apply_gradients(zip(grads, variables))
            return loss
        self.step = step

    @property
    def elapsed_seconds(self):
        return self.carried_seconds + time.perf_counter() - self.started

    def advance(self, target_update, deadline=float('inf')):
        if type(target_update) is not int or target_update < self.update:
            raise ValueError('Target update cannot move backwards')
        while self.update < target_update:
            if time.perf_counter() >= deadline:
                return 'time_limit'
            ids = self.rng.integers(0, len(self.features), self.config['batch_size'], dtype=np.int32)
            u = self.update + 1
            loss = float(self.step(ids, np.float32(learning_rate(u))).numpy())
            self.update = u
            if u == 1 or u % 50 == 0 or u in (6001, 9001):
                self.history.append(dict(seed=self.seed, update=u, loss=loss,
                    learning_rate=learning_rate(u)))
        return 'target_update'

    def full_train_loss(self, deadline=float('inf')):
        squared, count = 0., 0
        for start in range(0, len(self.features), 512):
            if time.perf_counter() >= deadline:
                return None
            output = self.predictor.model(self.features[start:start+512], training=False)
            error = output.numpy().astype('float64')-self.full_targets[start:start+512]
            squared += float(np.sum(error**2))
            count += error.size
        loss = squared/count
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite full train loss')
        self.full_train_history.append(dict(seed=self.seed, update=self.update, full_train_loss=loss))
        return loss

    def save_checkpoint(self, directory, reason):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.predictor.save(directory / 'model')
        variables = self.optimizer.variables
        np.savez_compressed(directory / 'adam.npz', **{f'v{i}': v.numpy() for i, v in enumerate(variables)})
        state = dict(format_version=1, seed=self.seed, update=self.update, signature=self.signature,
            optimizer_variables=[dict(name=v.name, shape=list(v.shape)) for v in variables],
            optimizer_iteration=int(self.optimizer.iterations.numpy()),
            learning_rate=float(self.optimizer.learning_rate.numpy()),
            rng_state=self.rng.bit_generator.state,
            tf_rng_state=self.tf_rng.state.numpy().tolist(), tf_rng_algorithm=int(self.tf_rng.algorithm),
            history=self.history, full_train_history=self.full_train_history,
            plateau_monitor=self.plateau_monitor, elapsed_seconds=self.elapsed_seconds, reason=reason)
        write_json(directory / 'state.json', state)
        pd.DataFrame(self.history, columns=['seed','update','loss','learning_rate']).to_csv(
            directory / 'history.csv', index=False)
        pd.DataFrame(self.full_train_history, columns=['seed','update','full_train_loss']).to_csv(
            directory / 'full_train_history.csv', index=False)
        write_json(directory / 'complete.json', dict(output_sha256={
            p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}))
        return directory

    @classmethod
    def restore(cls, directory, x, y, config, seed):
        directory = Path(directory)
        marker = json.loads((directory / 'complete.json').read_text())
        required = {'model.keras','model.json','adam.npz','state.json','history.csv','full_train_history.csv'}
        if set(marker['output_sha256']) != required:
            raise ValueError('Incomplete checkpoint manifest')
        for name, expected in marker['output_sha256'].items():
            if Path(name).name != name or sha256(directory / name) != expected:
                raise ValueError('Checkpoint file hash mismatch')
        state = json.loads((directory / 'state.json').read_text())
        if (state['format_version'] != 1 or state['seed'] != seed
                or state['signature'] != signature(x, y, config)):
            raise ValueError('Checkpoint data/config/source/runtime mismatch')
        p = PressureDeepONet.load(directory / 'model')
        self = cls(x, y, config, seed, predictor=p)
        variables = self.optimizer.variables
        actual = [dict(name=v.name, shape=list(v.shape)) for v in variables]
        if actual != state['optimizer_variables']:
            raise ValueError('Adam variable layout changed')
        with np.load(directory / 'adam.npz', allow_pickle=False) as saved:
            if len(saved.files) != len(variables):
                raise ValueError('Incomplete Adam state')
            for i, variable in enumerate(variables):
                variable.assign(saved[f'v{i}'])
        if int(self.optimizer.iterations.numpy()) != state['update'] or state['optimizer_iteration'] != state['update']:
            raise ValueError('Optimizer iteration/update mismatch')
        self.rng.bit_generator.state = state['rng_state']
        self.tf_rng = tf.random.Generator.from_state(state['tf_rng_state'], state['tf_rng_algorithm'])
        self.update, self.carried_seconds = state['update'], state['elapsed_seconds']
        self.history, self.full_train_history = state['history'], state['full_train_history']
        self.plateau_monitor = state['plateau_monitor']
        return self
