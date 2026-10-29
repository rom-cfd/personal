"""Float64 transforms with explicit inverse domains and training-only scales.

Box--Cox uses Y**gamma/gamma: the conventional formula plus 1/gamma.
This affine offset disappears under standardization and avoids cancellation at
zero and trace concentrations. It is not an epsilon-clipped baseline.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Transform:
    kind: str
    gamma: float = 0.1
    epsilon: object = 1e-8
    signed: bool = False

    def __post_init__(self):
        if self.kind not in {'boxcox', 'softlog', 'asinh', 'shifted_boxcox'}:
            raise ValueError('unknown transformation')
        if not np.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError('gamma must be finite and positive')
        self.epsilon = np.asarray(self.epsilon, dtype=np.float64)
        if np.any(~np.isfinite(self.epsilon)) or np.any(self.epsilon <= 0):
            raise ValueError('epsilon must be finite and positive')

    def forward(self, values):
        x = np.asarray(values, dtype=np.float64)
        if np.any(~np.isfinite(x)):
            raise ValueError('nonfinite inputs')
        if not self.signed and np.any(x < 0):
            raise ValueError('positive-only transform received negative values')
        a = np.abs(x) if self.signed else x
        if self.kind == 'boxcox':
            y = np.power(a, self.gamma) / self.gamma
        elif self.kind == 'softlog':
            y = np.log1p(a / self.epsilon)
        elif self.kind == 'asinh':
            y = np.arcsinh(a / self.epsilon)
        else:
            y = np.expm1(self.gamma * np.log1p(a / self.epsilon)) / self.gamma
        return np.sign(x) * y if self.signed else y

    def inverse(self, values):
        """Return NaN outside mathematical domain; never silently clip outputs.

        Softlog, asinh and shifted Box--Cox may produce negative physical values;
        evaluation must count those before any optional rollout repair.
        """
        z = np.asarray(values, dtype=np.float64)
        a = np.abs(z) if self.signed else z
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            if self.kind == 'boxcox':
                y = np.where(a >= 0, np.power(self.gamma * a, 1 / self.gamma), np.nan)
            elif self.kind == 'softlog':
                y = self.epsilon * np.expm1(a)
            elif self.kind == 'asinh':
                y = self.epsilon * np.sinh(a)
            else:
                y = self.epsilon * np.expm1(np.log1p(self.gamma * a) / self.gamma)
                y = np.where(1 + self.gamma * a > 0, y, np.nan)
        return np.sign(z) * y if self.signed else y


def species_epsilon(training_y, multiplier=1e-3, quantile=0.5, floor=1e-16):
    """Positive training-only quantiles, with a declared numerical lower floor."""
    y = np.asarray(training_y, dtype=np.float64)
    if y.ndim != 2 or np.any(~np.isfinite(y)) or multiplier <= 0:
        raise ValueError('expected finite (samples, species) and positive multiplier')
    scales = [np.quantile(np.abs(v[v != 0]), quantile) if np.any(v != 0) else floor
              for v in y.T]
    return np.maximum(np.asarray(scales) * multiplier, floor)


@dataclass
class Standardizer:
    mean: object = None
    scale: object = None

    def fit(self, training_values):
        x = np.asarray(training_values, dtype=np.float64)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale = np.where(self.scale > 1e-14, self.scale, 1.0)
        return self

    def forward(self, values):
        return (np.asarray(values) - self.mean) / self.scale

    def inverse(self, values):
        return np.asarray(values) * self.scale + self.mean
