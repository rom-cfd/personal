"""Fixed-2us Y9 branch / current T, log(P), initial equivalence-ratio trunk.

Physical input rows are [T, Y9, phi_initial, P_Pa]. Phi is supplied explicitly,
kept constant during a rollout, and never recomputed from a drifting prediction.
The first eleven standardized features match the original ANN exactly; the last
is phi standardized using the selected training inputs only.
"""
import json
from pathlib import Path

import numpy as np
from .deeponet import BranchTrunkProduct
from .study import Predictor, tf
from .transforms import Standardizer, Transform


class PhiDeepONet(Predictor):
    family = 'temperature_pressure_initial_phi_deeponet'

    def __init__(self, spec, x, y, config, seed):
        self.validate_inputs(x)
        if (spec['kind'] != 'boxcox' or spec.get('gamma', .1) != .1
                or spec.get('epsilon', 1e-8) != 1e-8 or spec.get('species', False)):
            raise ValueError('Use the common fixed-epsilon Box-Cox gamma=.1 baseline')
        if config['input_partition'] != 'current_Y9__current_T_logP_initial_phi':
            raise ValueError('Expected Y9 branch and current T/logP/initial phi trunk')
        self.spec, self.config = dict(spec), dict(config)
        self.transform = Transform('boxcox', .1, 1e-8)
        a, b = self.encode_state(x[:, :10]), self.encode_state(y)
        self.norm = Standardizer().fit(np.r_[a, b])
        self.input_norm = Standardizer().fit(np.c_[self.norm.forward(a), np.log(x[:, -1])])
        self.phi_norm = Standardizer().fit(x[:, 10:11])
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        inputs = tf.keras.Input((12,), name='common_state_logP_and_initial_phi')
        branch = inputs[:, 1:10]
        for i, width in enumerate(config['branch_layers']):
            branch = tf.keras.layers.Dense(width, activation='tanh', name=f'branch_{i}')(branch)
        latent = config['latent_width']
        branch = tf.keras.layers.Dense(10 * latent, kernel_initializer='zeros',
            bias_initializer='zeros', name='branch_coefficients')(branch)
        branch = tf.keras.layers.Reshape((10, latent))(branch)
        trunk = tf.keras.layers.Concatenate(name='current_T_logP_initial_phi')(
            [inputs[:, :1], inputs[:, 10:12]])
        for i, width in enumerate(config['trunk_layers']):
            trunk = tf.keras.layers.Dense(width, activation='tanh', name=f'trunk_{i}')(trunk)
        trunk = tf.keras.layers.Dense(latent, name='thermo_phi_basis')(trunk)
        self.model = tf.keras.Model(inputs,
            BranchTrunkProduct(name='component_residual')([branch, trunk]), name=self.family)
        self.train_features = self.features(x).astype('float32')
        self.train_target = (self.norm.forward(b) - self.norm.forward(a)).astype('float32')

    @staticmethod
    def validate_inputs(x):
        if (np.ndim(x) != 2 or x.shape[1] != 12
                or not np.isfinite(x[:, 10:12]).all() or np.any(x[:, 10:12] <= 0)):
            raise ValueError('Explicit positive initial phi and pressure required: [T,Y9,phi,P]')

    def features(self, x):
        self.validate_inputs(x)
        return np.c_[super().features(x), self.phi_norm.forward(x[:, 10:11])]

    def save(self, path):
        path = Path(path)
        if any(Path(str(path) + s).exists() for s in ('.keras', '.json')):
            raise FileExistsError('Never overwrite a saved T/P/phi model')
        super().save(path)
        meta_path = Path(str(path) + '.json')
        meta = json.loads(meta_path.read_text())
        meta.update(family=self.family, dt=2e-6, physical_input='T_Y9_phi_initial_P',
            phi_mean=self.phi_norm.mean.tolist(), phi_scale=self.phi_norm.scale.tolist(),
            architecture={k:self.config[k] for k in
                ('branch_layers','trunk_layers','latent_width','input_partition')})
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')

    @classmethod
    def load(cls, path):
        meta = json.loads(Path(str(path) + '.json').read_text())
        if (meta.get('family') != cls.family or meta.get('dt') != 2e-6
                or meta.get('physical_input') != 'T_Y9_phi_initial_P'
                or meta['architecture']['input_partition'] != 'current_Y9__current_T_logP_initial_phi'):
            raise ValueError('Not a fixed-2us T/P/initial-phi model')
        self = super().load(path)
        self.config = meta['architecture']
        self.phi_norm = Standardizer(np.array(meta['phi_mean']), np.array(meta['phi_scale']))
        return self
