"""Differentiable TensorFlow version of the rollout physics (WangHardCorrection, guard on).

Used inside unrolled training ("physics in the loop", approach 5). One step maps the
current encoded state z = [T, P_atm, BCT(Y)] (float64, carried between steps like the
rollout's float64 state) and the network's standardized increment to the next encoded
state exactly as the evaluation rollout does:

  1. z_next = z + delta (encoded [T, P_atm, BCT(Y)]); pressure increment is pinned to 0
  2. decoder guard: BCT(Y) clipped to [-1/gamma, 0]; Y_prop = (1 + gamma b)^(1/gamma)
  3. Wang 2025 Eqs. (5)-(9) element projection with weights Y_prop^(1-gamma)
     (row-scaled Gram solve; same feasible set/objective as kinetics.wang_physics.wang_project)
  4. reject (mask) a sample when any corrected Y < 0, the element residual > 1e-10, or the
     enthalpy root is infeasible -- the rollout rejects the same candidates
  5. temperature from the conserved mixture enthalpy (NASA-7, Cantera-normalized Y):
     Newton iterations without gradient, then one differentiable Newton step at the root
     (implicit-function gradient dT/dY = -(dh/dY)/cp)
  6. re-encode; features(z) = float32((z - mean)/scale) is the next network input.

Gradients flow through the projection and the temperature closure. Powers of Y near 0 use
a safe branch (no gradient below 1e-20) so 0 * inf never appears.
"""
import numpy as np
from .vector_physics import VectorThermo

FLOOR = 1e-20


class TFWangPhysics:
    def __init__(self, info, gamma=.1, newton_iterations=12):
        import tensorflow as tf
        self.tf = tf
        th = VectorThermo()
        f64 = lambda a: tf.constant(np.asarray(a, dtype=np.float64))
        self.gamma = float(gamma)
        self.M = f64(th.matrix)                                # (3, 9) element mass fractions, H/O/N
        self.mid, self.low, self.high = f64(th.mid), f64(th.low), f64(th.high)
        self.r_over_w = f64(th.r_over_w)
        self.hmin, self.hmax = f64(th.hmin), f64(th.hmax)
        self.im, self.isc = f64(info['input_mean']), f64(info['input_scale'])
        self.om, self.osc = f64(info['output_mean']), f64(info['output_scale'])
        self.iterations = newton_iterations

    # --- small helpers -------------------------------------------------------------
    def _pow_safe(self, y, p):
        tf = self.tf
        ok = y > FLOOR
        ys = tf.where(ok, y, tf.ones_like(y))
        return tf.where(ok, ys ** p, tf.stop_gradient(tf.maximum(y, 0.) ** p))

    def species_h_cp(self, t):
        tf = self.tf
        tt = t[:, None]
        a = tf.where((tt <= self.mid)[..., None], self.low[None], self.high[None])   # (n, 9, 7)
        cp = ((((a[..., 4] * tt + a[..., 3]) * tt + a[..., 2]) * tt + a[..., 1]) * tt + a[..., 0]) * self.r_over_w
        hrt = a[..., 0] + tt * (a[..., 1] / 2 + tt * (a[..., 2] / 3 + tt * (a[..., 3] / 4 + tt * a[..., 4] / 5))) + a[..., 5] / tt
        return hrt * tt * self.r_over_w, cp

    def z_from_features(self, x_std):
        return self.tf.cast(x_std, self.tf.float64) * self.isc + self.im

    def features(self, z):
        return self.tf.cast((z - self.im) / self.isc, self.tf.float32)

    def encode(self, x):
        """raw [T, Y9, P_Pa] (float64) -> z."""
        tf = self.tf
        x = tf.cast(x, tf.float64)
        return tf.concat([x[:, :1], x[:, -1:] / 101325., (self._pow_safe(x[:, 1:10], self.gamma) - 1.) / self.gamma], axis=1)

    def decode(self, z):
        """z -> (T, P_atm, Y) in float64."""
        b = self.tf.clip_by_value(z[:, 2:], -1. / self.gamma, 0.)
        return z[:, 0], z[:, 1], (1. + self.gamma * b) ** (1. / self.gamma)

    def enthalpy(self, t, y):
        tf = self.tf
        yn = y / tf.reduce_sum(y, axis=1, keepdims=True)
        hs, _ = self.species_h_cp(t)
        return tf.reduce_sum(yn * hs, axis=1)

    # --- one corrected step --------------------------------------------------------
    def step(self, z, pred_std, h0):
        """z (n,11) float64, standardized prediction -> (next z float64, ok mask, diagnostics)."""
        tf = self.tf
        g = self.gamma
        delta = tf.cast(pred_std, tf.float64) * self.osc + self.om
        t_cur, p_atm = z[:, 0], z[:, 1]                             # pressure pinned: delta P ignored
        b_cur = tf.clip_by_value(z[:, 2:], -1. / g, 0.)
        y_cur = (1. + g * b_cur) ** (1. / g)
        b_new = tf.clip_by_value(z[:, 2:] + delta[:, 2:], -1. / g, 0.)   # decoder guard
        y_prop = (1. + g * b_new) ** (1. / g)
        # Wang projection: min ||dG|| s.t. M (y_prop + d*dG) = M y_cur, d = y_prop^(1-gamma)
        d = self._pow_safe(y_prop, 1. - g)
        B = self.M[None] * d[:, None, :]                            # (n, 3, 9)
        r = tf.einsum('ni,ei->ne', y_cur - y_prop, self.M)          # (n, 3)
        rn = tf.norm(B, axis=2)
        rn = tf.where(rn > 0, rn, tf.ones_like(rn))
        B = B / rn[..., None]
        r = r / rn
        gram = tf.matmul(B, B, transpose_b=True) + 1e-14 * tf.eye(3, dtype=tf.float64)[None]
        lam = tf.linalg.solve(gram, r[..., None])[..., 0]
        y_new = y_prop + d * tf.einsum('nei,ne->ni', B, lam)
        resid = tf.reduce_max(tf.abs(tf.einsum('ni,ei->ne', y_new - y_cur, self.M)), axis=1)
        ok = tf.reduce_all(y_new >= 0., axis=1) & (resid <= 1e-10) & tf.reduce_all(tf.math.is_finite(y_new), axis=1)
        y_pos = tf.maximum(y_new, 0.)
        # enthalpy closure on normalized composition (same convention as VectorThermo.temperatures)
        yn = y_pos / tf.reduce_sum(y_pos, axis=1, keepdims=True)
        feasible = (tf.linalg.matvec(yn, self.hmin) <= h0 + 1e-7) & (tf.linalg.matvec(yn, self.hmax) >= h0 - 1e-7)
        t = tf.stop_gradient(tf.clip_by_value(t_cur, 200., 6000.))
        yn_s = tf.stop_gradient(yn); h0_s = tf.stop_gradient(h0)
        for _ in range(self.iterations):
            hs, cp = self.species_h_cp(t)
            e = tf.reduce_sum(yn_s * hs, axis=1) - h0_s
            t = tf.clip_by_value(t - e / tf.reduce_sum(yn_s * cp, axis=1), 200., 6000.)
        t = tf.stop_gradient(t)
        hs, cp = self.species_h_cp(t)
        t_new = t - (tf.reduce_sum(yn * hs, axis=1) - h0) / tf.stop_gradient(tf.reduce_sum(yn_s * cp, axis=1))
        converged = tf.abs(tf.reduce_sum(yn_s * hs, axis=1) - h0_s) <= 1e-3
        ok = ok & feasible & converged
        b_out = (self._pow_safe(y_pos, g) - 1.) / g
        z_next = tf.concat([t_new[:, None], p_atm[:, None], b_out], axis=1)
        return z_next, ok, dict(residual=resid, y=y_pos, t=t_new)
