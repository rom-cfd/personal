"""Unconstrained next-state increments with signed Box-Cox preprocessing.

Fit f(Y_next-Y_now), NOT f(Y_next)-f(Y_now). Temperature increments are
linearly standardized. No clipping, normalization, element/enthalpy projection,
or kinetics/thermodynamic closure is called during prediction.
"""
import json
from pathlib import Path
import numpy as np
from .study import tf
from .transforms import Transform, Standardizer


class DeltaPredictor:
    def __init__(self, spec, x, y, config, seed):
        self.spec = spec
        self.transform = Transform('boxcox', gamma=spec['gamma'], signed=True)
        self.input_norm = Standardizer().fit(self.encode_input(x))
        self.output_norm = Standardizer().fit(self.encode_delta(y-x[:, :10]))
        self.train_features = self.features(x).astype('float32')
        self.train_target = self.output_norm.forward(self.encode_delta(y-x[:, :10])).astype('float32')
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        inputs = tf.keras.Input((11,))
        h = inputs
        for width in config['hidden_layers']:
            h = tf.keras.layers.Dense(width, activation=config['activation'])(h)
        output = tf.keras.layers.Dense(10, kernel_initializer='zeros', bias_initializer='zeros')(h)
        self.model = tf.keras.Model(inputs, output)

    def encode_input(self, x):
        # Signed extension accepts negative self-fed values without clipping.
        # All fitted training concentrations themselves are nonnegative.
        return np.c_[x[:, 0]/1000, self.transform.forward(x[:, 1:10]), np.log(x[:, -1])]

    def features(self, x):
        return self.input_norm.forward(self.encode_input(x))

    def encode_delta(self, delta):
        return np.c_[delta[:, 0]/1000, self.transform.forward(delta[:, 1:])]

    def decode(self, standardized):
        z = self.output_norm.inverse(standardized)
        return np.concatenate([z[..., :1]*1000, self.transform.inverse(z[..., 1:])], axis=-1)

    def predict_delta(self, x):
        z = self.model(self.features(x).astype('float32'), training=False).numpy().astype('float64')
        return self.decode(z)

    def predict(self, x):
        return x[:, :10] + self.predict_delta(x)

    def save(self, path):
        self.model.save(Path(str(path)+'.keras'))
        Path(str(path)+'.json').write_text(json.dumps(dict(
            model_type='physical_increment_signed_boxcox', spec=self.spec,
            input_mean=self.input_norm.mean.tolist(), input_scale=self.input_norm.scale.tolist(),
            output_mean=self.output_norm.mean.tolist(), output_scale=self.output_norm.scale.tolist()), indent=2))

    @classmethod
    def load(cls, path):
        info = json.loads(Path(str(path)+'.json').read_text())
        if info['model_type'] != 'physical_increment_signed_boxcox':
            raise ValueError('Incorrect model family')
        self = cls.__new__(cls)
        self.spec = info['spec']; self.transform = Transform('boxcox', self.spec['gamma'], signed=True)
        self.input_norm = Standardizer(np.array(info['input_mean']), np.array(info['input_scale']))
        self.output_norm = Standardizer(np.array(info['output_mean']), np.array(info['output_scale']))
        self.model = tf.keras.models.load_model(Path(str(path)+'.keras'), compile=False)
        return self


class DeltaCommittee:
    def __init__(self, paths, seeds, metric, rule='mean'):
        from .committee import choose_medoid
        if len(paths) != len(seeds) or not len(paths):
            raise ValueError('Empty or mismatched committee')
        if rule not in ('mean', 'medoid', 'single') or (rule == 'single' and len(paths) != 1):
            raise ValueError('Invalid raw aggregation rule')
        self.paths = list(map(str, paths)); self.seeds = np.array(seeds)
        self.metric, self.rule = metric, rule
        self.members = [DeltaPredictor.load(path) for path in paths]
        self.base = self.members[0]
        for p in self.members[1:]:
            for a, b in [(p.input_norm.mean, self.base.input_norm.mean),
                         (p.input_norm.scale, self.base.input_norm.scale),
                         (p.output_norm.mean, self.base.output_norm.mean),
                         (p.output_norm.scale, self.base.output_norm.scale)]:
                np.testing.assert_array_equal(a, b)
        @tf.function(input_signature=[tf.TensorSpec([None, 11], tf.float32)])
        def forward(features):
            return tf.stack([p.model(features, training=False) for p in self.members], axis=0)
        self.forward = forward
        self.last_diagnostics = {}

    def candidates(self, x):
        z = self.forward(self.base.features(x).astype('float32')).numpy().astype('float64')
        return x[None, :, :10] + self.base.decode(z)

    def predict(self, x):
        from .committee import choose_medoid
        c = self.candidates(x)
        valid = np.all(np.isfinite(c), axis=-1)
        if self.rule == 'single':
            out = c[0]; choice = np.where(valid[0], 0, -1)
        elif self.rule == 'mean':
            # Negative/out-of-range finite candidates remain in the mean.
            count = valid.sum(axis=0)
            out = np.where(valid[..., None], c, 0).sum(axis=0) / np.maximum(count, 1)[:, None]
            out[count == 0] = np.nan
            choice = np.full(len(x), -1)
        else:
            # Actual unconstrained candidate; no physics-based candidate filter.
            out, choice, _ = choose_medoid(c, self.metric, 'scaled')
        selected = np.full(len(x), -1)
        if self.rule != 'mean':
            live = choice >= 0; selected[live] = self.seeds[choice[live]]
        self.last_diagnostics = dict(selected_seed=selected, valid_count=valid.sum(axis=0))
        return out

    def save(self, path):
        Path(path).write_text(json.dumps(dict(paths=self.paths, seeds=self.seeds.tolist(),
                                             metric=self.metric, rule=self.rule), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
