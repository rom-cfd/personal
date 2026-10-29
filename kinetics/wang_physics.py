"""Wang et al., Combustion and Flame 275 (2025) 114105, Eqs. (6)-(9).

Use the signed residual in Eq. (7), E=A(Y_current-Y_predicted).
The appendix pseudocode prints the opposite residual sign without the matching
minus sign. Equations (5)-(9), rather than that inconsistent line, define this
implementation. Corrections are ADDED IN Y SPACE, not inverse-transformed G.
No positivity constraint, floor, clipping or old-projector fallback is added.
"""
import numpy as np
from .vector_physics import VectorThermo


def wang_project(proposal, current, matrix, gamma=.1):
    """Batched minimum ||delta_G||_2 subject to B delta_G=A(current-proposal).

    Row scaling improves conditioning without changing the feasible set or
    objective. Moore-Penrose solve handles zero-support/rank-deficient cases;
    callers must reject residuals if the constraints cannot be satisfied.
    Returns corrected Y, including possible negatives, and element residual.
    """
    if not 0 < gamma <= 1:
        raise ValueError('require 0 < gamma <= 1')
    v = np.asarray(proposal, dtype=float)
    c = np.broadcast_to(np.asarray(current, dtype=float), v.shape)
    if v.ndim != 2 or not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ValueError('finite nonnegative two-dimensional proposals required')
    derivative_inverse = v ** (1 - gamma)
    b = matrix[None, :, :] * derivative_inverse[:, None, :]
    residual = (c - v) @ matrix.T
    row_norm = np.linalg.norm(b, axis=2)
    row_norm = np.where(row_norm > 0, row_norm, 1.)
    b = b / row_norm[:, :, None]
    residual = residual / row_norm
    gram = b @ b.swapaxes(1, 2)
    multiplier = (np.linalg.pinv(gram, rcond=1e-14) @ residual[..., None])[..., 0]
    delta_g = np.einsum('nei,ne->ni', b, multiplier)
    corrected = v + derivative_inverse * delta_g
    error = np.max(np.abs((corrected - c) @ matrix.T), axis=1)
    return corrected, error


class WangThermo(VectorThermo):
    """Paper projection plus existing common adiabatic fixed-P thermal closure.

    Negative/inconsistent candidates are rejected, never clipped/renormalized.
    Zero species have zero correction weight for gamma<1 (the analytic limit).
    """
    def __init__(self, gamma=.1):
        super().__init__()
        self.gamma = gamma
        self.counts = dict(proposals=0, nonfinite=0, negative=0,
                           inconsistent=0, thermal_invalid=0, accepted=0)

    def repair(self, proposals, x):
        shape = proposals.shape
        members = shape[0]
        raw = proposals.reshape(-1, 10)
        current = np.tile(x[:, :10], (members, 1))
        pressure = np.tile(x[:, -1], members)
        out = np.full_like(raw, np.nan)
        finite = np.all(np.isfinite(raw), axis=1) & np.all(raw[:, 1:] >= 0, axis=1)
        self.counts['proposals'] += len(raw)
        self.counts['nonfinite'] += int((~finite).sum())
        live = np.flatnonzero(finite)
        if not len(live):
            return out.reshape(shape)
        y, residual = wang_project(raw[live, 1:], current[live, 1:], self.matrix, self.gamma)
        negative = np.any(y < 0, axis=1)
        inconsistent = (~np.isfinite(residual)) | (residual > 1e-10)
        self.counts['negative'] += int(negative.sum())
        self.counts['inconsistent'] += int(inconsistent.sum())
        good = ~negative & ~inconsistent
        y, live = y[good], live[good]
        if not len(live):
            return out.reshape(shape)
        hs, _ = self.species_h_cp(x[:, 0])
        normalized = x[:, 1:10] / x[:, 1:10].sum(axis=1)[:, None]
        h0 = np.sum(normalized * hs, axis=1)
        t = self.temperatures(y, current[live, 0], np.tile(h0, members)[live], pressure[live])
        self.counts['thermal_invalid'] += int((~np.isfinite(t)).sum())
        self.counts['accepted'] += int(np.isfinite(t).sum())
        out[live] = np.c_[t, y]
        return out.reshape(shape)
