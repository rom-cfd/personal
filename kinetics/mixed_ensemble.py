"""Same-state means for ANN, T/P-query DeepONet and their heterogeneous union.

No medoid, reference-state feedback, ODE fallback, or family-level reweighting.
"""
import time
import numpy as np
from .study import tf
from .vector_physics import VectorThermo


def require_shared_preprocessing(members):
    if not members:
        raise ValueError('Empty ensemble')
    base = members[0]
    for member in members[1:]:
        if member.spec['kind'] != base.spec['kind'] or member.transform.gamma != base.transform.gamma:
            raise ValueError('All families must use the same transformation')
        for a, b in ((member.norm.mean, base.norm.mean), (member.norm.scale, base.norm.scale),
                     (member.input_norm.mean, base.input_norm.mean),
                     (member.input_norm.scale, base.input_norm.scale),
                     (member.transform.epsilon, base.transform.epsilon)):
            np.testing.assert_array_equal(a, b)


class MixedMean:
    def __init__(self, members, families, mode='repaired'):
        require_shared_preprocessing(members)
        if len(families) != len(members) or not set(families) <= {'ann', 'deeponet'}:
            raise ValueError('One known family label is required per member')
        if mode not in ('raw', 'repaired'):
            raise ValueError('Mode must be raw or repaired')
        self.members = list(members)
        self.families = np.asarray(families)
        self.base = members[0]
        self.mode = mode
        self.physics = VectorThermo()
        self.reset_timing()

        @tf.function(input_signature=[tf.TensorSpec([None, 11], tf.float32)])
        def forward(features):
            return tf.stack([p.model(features, training=False) for p in self.members])
        self.forward = forward
        self.last_diagnostics = {}

    def reset_timing(self):
        self.timing = dict(neural_decode=0., candidate_repair=0., mean_repair=0.)

    def proposals(self, x):
        base = self.base
        delta = self.forward(base.features(x).astype('float32')).numpy().astype('float64')
        z = base.norm.inverse(base.norm.forward(base.encode_state(x[:, :10]))[None, ...] + delta)
        ys = z[..., 1:]
        if self.mode == 'repaired':
            ys = np.clip(ys, 0, base.transform.forward(np.ones((1, 9))))
        return np.concatenate([z[..., :1] * 1000, base.transform.inverse(ys)], axis=-1)

    def predict(self, x):
        start = time.perf_counter()
        candidates = self.proposals(x)
        self.timing['neural_decode'] += time.perf_counter() - start
        if self.mode == 'repaired':
            start = time.perf_counter()
            candidates = self.physics.repair(candidates, x)
            self.timing['candidate_repair'] += time.perf_counter() - start
        valid = np.all(np.isfinite(candidates), axis=-1)
        count = valid.sum(axis=0)
        # Mask complete candidate rows, never individual coordinates.
        total = np.where(valid[..., None], candidates, 0.).sum(axis=0)
        out = total / np.maximum(count, 1)[:, None]
        if self.mode == 'raw':
            # Raw baseline is an actual all-member mean, not invalid filtering.
            out[count != len(self.members)] = np.nan
        else:
            out[count == 0] = np.nan
            start = time.perf_counter()
            out = self.physics.repair(out[None, ...], x)[0]
            self.timing['mean_repair'] += time.perf_counter() - start
        self.last_diagnostics = dict(
            selected_seed=np.full(len(x), -1), valid_count=count,
            valid_ann=valid[self.families == 'ann'].sum(axis=0),
            valid_deeponet=valid[self.families == 'deeponet'].sum(axis=0),
            spread=np.full(len(x), np.nan), central_cost=np.full(len(x), np.nan))
        return out


def rollout(predictor, initial_states, pressures, steps, dt, ids=None, deadline=float('inf')):
    """Self-fed states only: future reference states are not accepted as arguments."""
    state = np.asarray(initial_states).copy()
    ids = np.arange(len(state)) if ids is None else np.asarray(ids)
    histories = [[s.copy()] for s in state]
    active = np.ones(len(state), bool)
    events = []
    for step in range(1, steps + 1):
        if time.perf_counter() >= deadline:
            raise TimeoutError('Rollout budget exhausted')
        live = np.flatnonzero(active)
        if not len(live):
            break
        x = np.c_[state[live], pressures[live]]
        pred = predictor.predict(x)
        for j, k in enumerate(live):
            out = pred[j]
            accepted = bool(np.all(np.isfinite(out)) and 200 <= out[0] <= 6000
                            and np.all(out[1:] >= 0) and out[1:].sum() > 0)
            diag = predictor.last_diagnostics
            events.append(dict(trajectory=int(ids[k]), step=step, time=step * dt,
                               valid_count=int(diag['valid_count'][j]),
                               valid_ann=int(diag['valid_ann'][j]),
                               valid_deeponet=int(diag['valid_deeponet'][j]), accepted=accepted))
            if accepted:
                state[k] = out
                histories[k].append(out.copy())
            else:
                active[k] = False
    return [np.asarray(h) for h in histories], events
