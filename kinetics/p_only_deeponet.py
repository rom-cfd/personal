"""Fixed-2us DeepONet-style transition model: current T,Y branch; log(P) trunk."""
import json
from pathlib import Path

import numpy as np
from .deeponet import BranchTrunkProduct
from .study import Predictor, tf
from .transforms import Standardizer, Transform


class PressureDeepONet(Predictor):
    family = 'pressure_only_deeponet'

    def __init__(self, spec, x, y, config, seed):
        if (spec['kind'] != 'boxcox' or spec.get('gamma', .1) != .1
                or spec.get('epsilon', 1e-8) != 1e-8 or spec.get('species', False)):
            raise ValueError('Use the common Box-Cox gamma=.1, epsilon=1e-8 baseline')
        if config['input_partition'] != 'current_T_Y9__logP':
            raise ValueError('Branch requires current T,Y9; trunk requires log(P) only')
        widths = config['branch_layers'] + config['trunk_layers'] + [config['latent_width']]
        if not widths or any(type(w) is not int or w < 1 for w in widths):
            raise ValueError('Positive integer network widths required')
        self.spec, self.config = dict(spec), dict(config)
        self.transform = Transform('boxcox', .1, 1e-8)
        a, b = self.encode_state(x[:, :10]), self.encode_state(y)
        self.norm = Standardizer().fit(np.r_[a, b])
        self.input_norm = Standardizer().fit(np.c_[self.norm.forward(a), np.log(x[:, -1])])
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        inputs = tf.keras.Input((11,), name='standardized_state_and_log_pressure')
        branch = inputs[:, :10]
        for i, width in enumerate(config['branch_layers']):
            branch = tf.keras.layers.Dense(width, activation='tanh', name=f'branch_{i}')(branch)
        latent = config['latent_width']
        branch = tf.keras.layers.Dense(10 * latent, kernel_initializer='zeros',
            bias_initializer='zeros', name='branch_coefficients')(branch)
        branch = tf.keras.layers.Reshape((10, latent))(branch)
        trunk = inputs[:, 10:11]
        for i, width in enumerate(config['trunk_layers']):
            trunk = tf.keras.layers.Dense(width, activation='tanh', name=f'trunk_{i}')(trunk)
        trunk = tf.keras.layers.Dense(latent, name='pressure_basis')(trunk)
        residual = BranchTrunkProduct(name='component_residual')([branch, trunk])
        self.model = tf.keras.Model(inputs, residual, name=self.family)
        self.train_features = self.features(x).astype('float32')
        self.train_target = (self.norm.forward(b) - self.norm.forward(a)).astype('float32')

    def save(self, path):
        path = Path(path)
        if any(Path(str(path) + suffix).exists() for suffix in ('.keras', '.json')):
            raise FileExistsError('Never overwrite a saved P-only model')
        super().save(path)
        info_path = Path(str(path) + '.json')
        info = json.loads(info_path.read_text())
        info.update(family=self.family, dt=2e-6, architecture={k: self.config[k] for k in
                    ('branch_layers', 'trunk_layers', 'latent_width', 'input_partition')})
        info_path.write_text(json.dumps(info, indent=2) + '\n')

    @classmethod
    def load(cls, path):
        info = json.loads(Path(str(path) + '.json').read_text())
        if (info.get('family') != cls.family or info.get('dt') != 2e-6
                or info['architecture']['input_partition'] != 'current_T_Y9__logP'):
            raise ValueError('Not a fixed-2us P-only model')
        self = super().load(path)
        self.config = info['architecture']
        return self
