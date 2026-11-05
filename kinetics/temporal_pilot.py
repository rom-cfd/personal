"""Matched-time, sequential pilot of interval and self-fed multi-step DeepONets.

Existing data/results are immutable. Training uses only the 54 training ICs.
Prediction always advances 2 us, including the interval-conditioned model.
"""
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/kinetics-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/kinetics-cache')

import argparse
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .study import tf
from .deeponet import BranchTrunkProduct
from .transforms import Standardizer
from .vector_physics import VectorThermo

METHODS = ('single_step', 'interval', 'multistep')
GAMMA = .2
DT = 2e-6
SOURCE_SHA = '8d41f19f2145e12f9e89ee3eaa88fb0b1b93a84dcaaee65dcaa2e8b264077563'
PAIR_SHA = '44db17b16e167ea389631dacb8015528f76cbd5c2072e9fb919701e54d50fa3e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def encode(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all() or np.any(x[..., 1:] < 0):
        raise ValueError('Expected finite nonnegative physical states')
    return np.concatenate([x[..., :1] / 1000., x[..., 1:] ** GAMMA / GAMMA], axis=-1)


def decode(z, repaired=False):
    """Raw inverse-domain failures remain NaN; clipping is repaired-only."""
    e = np.array(z, dtype=np.float64, copy=True)
    if repaired:
        e[..., 1:] = np.clip(e[..., 1:], 0., 1. / GAMMA)
    with np.errstate(invalid='ignore', over='ignore'):
        y = np.where(e[..., 1:] >= 0, (GAMMA * e[..., 1:]) ** (1. / GAMMA), np.nan)
    return np.concatenate([e[..., :1] * 1000., y], axis=-1)


def valid_starts(parent, k, steps):
    return np.flatnonzero(parent[:, 1] + k <= steps).astype(np.int32)


def prepare_training(source, pairs):
    if sha(source) != SOURCE_SHA or sha(pairs) != PAIR_SHA:
        raise ValueError('Original data hash mismatch')
    with np.load(pairs) as p:
        x, y, parent = p['x'], p['y'], p['parent']
    with np.load(source) as d:
        ids = np.flatnonzero(d['split'] == 'train')
        if set(parent[:, 0]) != set(ids) or len(ids) != 54:
            raise ValueError('Training trajectory membership mismatch')
        train_states = d['states'][ids]
        np.testing.assert_array_equal(x[:, :10], d['states'][parent[:, 0], parent[:, 1]])
        np.testing.assert_array_equal(y, d['states'][parent[:, 0], parent[:, 1] + 1])
        np.testing.assert_array_equal(x[:, -1], d['pressures'][parent[:, 0]])
        if float(d['dt']) != DT:
            raise ValueError('Unexpected timestep')
    a, b = encode(x[:, :10]), encode(y)
    norm = Standardizer().fit(np.concatenate([a, b]))
    z = norm.forward(a)
    feature_norm = Standardizer().fit(np.c_[z, np.log(x[:, -1])])
    remap = {int(v): i for i, v in enumerate(ids)}
    local_parent = np.array([[remap[int(i)], j] for i, j in parent], dtype=np.int32)
    return dict(x=x, y=y, parent=parent, local_parent=local_parent,
                dense=norm.forward(encode(train_states)).astype(np.float32),
                z=z.astype(np.float32), q=feature_norm.forward(np.c_[z, np.log(x[:, -1])])[:, -1:].astype(np.float32),
                norm=norm, feature_norm=feature_norm, train_ids=ids)


def build_model(method, seed):
    if method not in METHODS:
        raise ValueError(method)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    size = 12 if method == 'interval' else 11
    inputs = tf.keras.Input((size,), name='standardized_features')
    branch = inputs[:, :11] if method == 'interval' else inputs[:, :10]
    trunk = inputs[:, 11:] if method == 'interval' else inputs[:, 10:11]
    for i in range(2):
        branch = tf.keras.layers.Dense(64, activation='tanh', name=f'branch_{i}')(branch)
    branch = tf.keras.layers.Dense(70, kernel_initializer='zeros', bias_initializer='zeros',
                                   name='coefficients')(branch)
    branch = tf.keras.layers.Reshape((10, 7))(branch)
    trunk = tf.keras.layers.Dense(32, activation='tanh', name='trunk_hidden')(trunk)
    trunk = tf.keras.layers.Dense(7, name='basis')(trunk)
    out = BranchTrunkProduct(name='residual')([branch, trunk])
    return tf.keras.Model(inputs, out, name=method)


def normalized_features(z, q, mean, scale, k=None):
    features = tf.concat([(z - mean[:10]) / scale[:10], q], axis=-1)
    if k is not None:
        # Fixed map of log2(k) = {0,1,2,3} to [-1,1]; no validation/test fit.
        tau = (tf.math.log(tf.cast(k, tf.float32)) / tf.math.log(2.) - 1.5) / 1.5
        features = tf.concat([features, tf.reshape(tau, [-1, 1])], axis=-1)
    return features


def recursive_loss(model, z, q, targets, mean, scale):
    """All subsequent inputs are predictions, with BPTT through every step."""
    k = tf.shape(targets)[1]

    def body(j, state, total):
        state = state + model(normalized_features(state, q, mean, scale), training=True)
        total = total + tf.reduce_mean((state - targets[:, j]) ** 2)
        return j + 1, state, total

    _, _, total = tf.while_loop(lambda j, *_: j < k, body,
                               (tf.constant(0), z, tf.constant(0., tf.float32)),
                               parallel_iterations=1)
    return total / tf.cast(k, tf.float32)


def train_one(method, seed, prep, root, seconds):
    out = root / 'models' / f'{method}_seed{seed}'
    out.mkdir(parents=True)
    wall_start = time.perf_counter()
    model = build_model(method, seed)
    optimizer = tf.keras.optimizers.Adam(.001)
    optimizer.build(model.trainable_variables)
    z = tf.constant(prep['z'])
    q = tf.constant(prep['q'])
    dense = tf.constant(prep['dense'])
    parents = tf.constant(prep['local_parent'])
    mean = tf.constant(prep['feature_norm'].mean, tf.float32)
    scale = tf.constant(prep['feature_norm'].scale, tf.float32)

    @tf.function(input_signature=[tf.TensorSpec([512], tf.int32),
                                 tf.TensorSpec([512], tf.int32),
                                 tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.float32)])
    def step(ids, horizons, length, lr):
        optimizer.learning_rate.assign(lr)
        initial = tf.gather(z, ids)
        pressure = tf.gather(q, ids)
        par = tf.gather(parents, ids)
        with tf.GradientTape() as tape:
            if method == 'multistep':
                times = par[:, 1:2] + tf.range(1, length + 1)[None, :]
                ic = tf.broadcast_to(par[:, :1], tf.shape(times))
                truth = tf.gather_nd(dense, tf.stack([ic, times], axis=-1))
                loss = recursive_loss(model, initial, pressure, truth, mean, scale)
            else:
                endpoint = tf.gather_nd(dense, tf.stack([par[:, 0], par[:, 1] + horizons], axis=-1))
                features = normalized_features(initial, pressure, mean, scale,
                                               horizons if method == 'interval' else None)
                predicted = initial + model(features, training=True)
                loss = tf.reduce_mean((predicted - endpoint) ** 2)
            loss = tf.debugging.check_numerics(loss, 'nonfinite training loss')
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, norm = tf.clip_by_global_norm(gradients, 5.)
        for grad in gradients:
            tf.debugging.assert_all_finite(grad, 'nonfinite gradient')
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, norm

    # Trace graph without a gradient update. Compilation time is reported separately.
    compile_start = time.perf_counter()
    step.get_concrete_function()
    compile_seconds = time.perf_counter() - compile_start
    rng = np.random.default_rng(seed)
    pools = {k: valid_starts(prep['parent'], k, prep['dense'].shape[1] - 1)
             for k in (1, 2, 4, 8, 16, 64)}
    active = 0.
    updates = 0
    evaluations = 0
    history = []
    by_length = {}
    previous_length = None
    last_log = -10.
    while active < seconds:
        started = time.perf_counter()
        fraction = active / seconds
        length = (1, 4, 16, 64)[min(int(fraction * 4), 3)] if method == 'multistep' else 1
        lr = .001 * (1. if fraction < .6 else .3 if fraction < .9 else .1)
        if method == 'interval':
            horizons = np.r_[np.ones(256, np.int32), rng.choice([2, 4, 8], 256).astype(np.int32)]
            ids = np.empty(512, np.int32)
            for k in (1, 2, 4, 8):
                mask = horizons == k
                ids[mask] = rng.choice(pools[k], int(mask.sum()))
        else:
            horizons = np.ones(512, np.int32)
            ids = rng.choice(pools[length], 512)
        loss, gradnorm = step(ids, horizons, np.int32(length), np.float32(lr))
        loss, gradnorm = float(loss.numpy()), float(gradnorm.numpy())
        elapsed = time.perf_counter() - started
        active += elapsed
        updates += 1
        evaluations += length
        rec = by_length.setdefault(str(length), dict(updates=0, active_seconds=0.))
        rec['updates'] += 1
        rec['active_seconds'] += elapsed
        if updates == 1 or updates % 100 == 0 or length != previous_length or active >= seconds:
            history.append(dict(update=updates, length=length, loss=loss, gradient_norm=gradnorm,
                                learning_rate=lr, active_seconds=active))
        if active - last_log >= 10. or length != previous_length or active >= seconds:
            status = dict(phase='training', method=method, seed=seed, update=updates,
                          length=length, loss=loss, active_seconds=active, budget_seconds=seconds)
            write_json(root / 'status.json', status)
            print('TRAIN', json.dumps(status), flush=True)
            last_log = active
        previous_length = length
    model.compile(optimizer=optimizer, loss='mse')
    model.save(out / 'model.keras')
    tf.train.Checkpoint(model=model, optimizer=optimizer).write(str(out / 'optimizer_checkpoint'))
    probe = normalized_features(z[:8], q[:8], mean, scale,
                                tf.ones(8,tf.int32) if method=='interval' else None)
    reloaded = tf.keras.models.load_model(out / 'model.keras',compile=True)
    np.testing.assert_array_equal(model(probe).numpy(),reloaded(probe).numpy())
    if int(reloaded.optimizer.iterations.numpy()) != updates:
        raise ValueError('Adam state reload mismatch')
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    info = dict(method=method, seed=seed, gamma=GAMMA, dt=DT,
                active_seconds=active, budget_seconds=seconds, compile_seconds=compile_seconds,
                wall_seconds=time.perf_counter() - wall_start, updates=updates,
                network_forward_steps=evaluations, parameters=model.count_params(),
                by_length=by_length, termination='time_cap', plateau_reached=False,
                budget_note='from scratch; no 10k prerequisite; active minibatch/forward/backward/Adam time',
                model_sha256=sha(out / 'model.keras'))
    write_json(out / 'training.json', info)
    return info


class PilotPredictor:
    def __init__(self, model_paths, preprocessing, mode='raw'):
        if mode not in ('raw', 'repaired'):
            raise ValueError(mode)
        self.models = [tf.keras.models.load_model(p, compile=False) for p in model_paths]
        self.mode = mode
        self.mean = np.asarray(preprocessing['output_mean'])
        self.scale = np.asarray(preprocessing['output_scale'])
        self.input_mean = np.asarray(preprocessing['input_mean'])
        self.input_scale = np.asarray(preprocessing['input_scale'])
        self.physics = VectorThermo() if mode == 'repaired' else None
        self.valid_min = len(self.models)

        @tf.function(input_signature=[tf.TensorSpec([None, 11], tf.float32)])
        def forward(features):
            values = []
            for model in self.models:
                if model.input_shape[-1] == 12:
                    f = tf.concat([features, -tf.ones_like(features[:, :1])], axis=-1)
                else:
                    f = features
                values.append(model(f, training=False))
            return tf.stack(values)

        self.forward = forward

    def candidates(self, x):
        standardized = (encode(x[:, :10]) - self.mean) / self.scale
        features = (np.c_[standardized, np.log(x[:, -1])] - self.input_mean) / self.input_scale
        residual = self.forward(features.astype(np.float32)).numpy().astype(np.float64)
        encoded = (standardized[None, ...] + residual) * self.scale + self.mean
        return decode(encoded, repaired=self.mode == 'repaired')

    def predict(self, x):
        candidates = self.candidates(x)
        if self.mode == 'repaired':
            candidates = self.physics.repair(candidates, x)
        valid = np.isfinite(candidates).all(axis=-1)
        counts = valid.sum(axis=0)
        self.valid_min = min(self.valid_min, int(counts.min()))
        if self.mode == 'raw':
            result = candidates.mean(axis=0)
            result[counts != len(self.models)] = np.nan
            return result
        result = np.where(valid[..., None], candidates, 0.).sum(axis=0) / np.maximum(counts, 1)[:, None]
        result[counts == 0] = np.nan
        return self.physics.repair(result[None, ...], x)[0]


def nrmse(reference, predicted):
    """User util.py default: RMSE / RMS(reference), dimensionless, no percent."""
    if reference.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError('NRMSE requires a complete aligned finite profile')
    rmse = np.sqrt(np.mean((reference - predicted) ** 2, axis=0))
    rms = np.sqrt(np.mean(reference ** 2, axis=0))
    return np.divide(rmse, rms, out=np.where(rmse == 0, 0., np.inf), where=rms > 0)


def advance(predictor, initial, pressure, steps, callback=None):
    """No future reference array accepted: every accepted output is fed back."""
    state = initial.copy()
    histories = [[x.copy()] for x in state]
    active = np.ones(len(state), bool)
    failures = ['completed'] * len(state)
    for j in range(1, steps + 1):
        live = np.flatnonzero(active)
        if not len(live):
            break
        predicted = predictor.predict(np.c_[state[live], pressure[live]])
        for k, value in zip(live, predicted):
            if not np.isfinite(value).all():
                reason = 'nonfinite_or_inverse_domain'
            elif not 200 <= value[0] <= 6000:
                reason = 'temperature_outside_200_6000K'
            elif np.any(value[1:] < 0) or value[1:].sum() <= 0:
                reason = 'invalid_mass_fraction'
            else:
                state[k] = value
                histories[k].append(value.copy())
                continue
            failures[k] = reason
            active[k] = False
        if callback is not None and j % 500 == 0:
            callback(j, int(active.sum()))
    return [np.asarray(x) for x in histories], failures


def evaluate_split(root, source, seeds, split, preprocessing):
    with np.load(source) as d:
        ids = np.flatnonzero(d['split'] == split)
        initial = d['states'][ids, 0]
        reference = d['states'][ids]
        pressures = d['pressures'][ids]
        variables = ['T'] + d['species'].tolist()
        matrix = d['element_matrix']
    steps = reference.shape[1] - 1
    rows = []
    scores = []
    one_step_rows = []
    eval_root = root / 'evaluations' / split
    eval_root.mkdir(parents=True)
    for method in METHODS:
        for members in [[seed] for seed in seeds] + [list(seeds)]:
            tag = f'{method}_' + ('mean3' if len(members) > 1 else f'seed{members[0]}')
            paths = [root / 'models' / f'{method}_seed{s}' / 'model.keras' for s in members]
            # True-input one-step diagnostic, once per member/ensemble, always raw.
            raw = PilotPredictor(paths, preprocessing, 'raw')
            x = np.c_[reference[:, :-1].reshape(-1, 10), np.repeat(pressures, steps)]
            true_next = reference[:, 1:].reshape(-1, 10)
            chunks = [raw.predict(x[j:j+2048]) for j in range(0, len(x), 2048)]
            one = np.concatenate(chunks)
            valid = np.isfinite(one).all(axis=1)
            for col, variable in enumerate(variables):
                one_step_rows.append(dict(method=method, predictor=tag, split=split, variable=variable,
                    finite_fraction=float(valid.mean()),
                    nrmse_all=float(nrmse(true_next, one)[col]) if valid.all() else np.nan,
                    nrmse_finite_only=float(nrmse(true_next[valid], one[valid])[col]) if valid.any() else np.nan))
            pd.DataFrame(one_step_rows).to_csv(eval_root / 'one_step_NRMSE.csv', index=False)
            for mode in ('raw', 'repaired'):
                predictor = raw if mode == 'raw' else PilotPredictor(paths, preprocessing, mode)
                destination = eval_root / f'{tag}_{mode}'
                destination.mkdir()
                started = time.perf_counter()

                def progress(step, active):
                    write_json(root / 'status.json', dict(phase='evaluation', split=split,
                               predictor=tag, mode=mode, step=step, active=active))

                print('EVALUATE', split, tag, mode, flush=True)
                traces, failures = advance(predictor, initial, pressures, steps, progress)
                np.savez_compressed(destination / 'traces.npz', **{f'IC_{i}': h for i, h in zip(ids, traces)})
                for i, truth, trace, failure in zip(ids, reference, traces, failures):
                    complete = len(trace) == len(truth)
                    full = nrmse(truth, trace) if complete else np.full(10, np.nan)
                    early_n = min(len(truth), 51)
                    early = nrmse(truth[:early_n], trace[:early_n]) if len(trace) >= early_n else np.full(10, np.nan)
                    element = (trace[:, 1:] - truth[0, 1:]) @ matrix.T
                    row = dict(method=method, predictor=tag, split=split, mode=mode, trajectory=int(i),
                        complete=complete, completed_steps=len(trace)-1, termination=failure,
                        full_worst9=float(np.max(full[:9])) if complete else np.nan,
                        early_worst9=float(np.max(early[:9])) if np.isfinite(early).all() else np.nan,
                        joint9_le_0p1=bool(complete and np.max(full[:9]) <= .1),
                        temperature_rmse=float(np.sqrt(np.mean((trace[:, 0] - truth[:len(trace), 0]) ** 2))),
                        mass_sum_max=float(np.max(np.abs(trace[:, 1:].sum(axis=1)-1))),
                        elemental_max=float(np.max(np.abs(element))),
                        valid_members_min=predictor.valid_min)
                    rows.append(row)
                    for j, variable in enumerate(variables):
                        scores.append(dict(method=method, predictor=tag, split=split, mode=mode,
                            trajectory=int(i), variable=variable, complete=complete,
                            nrmse=full[j], early_0p1ms_nrmse=early[j]))
                write_json(destination / 'complete.json', dict(status='passed', seconds=time.perf_counter()-started,
                    traces_sha256=sha(destination / 'traces.npz'), complete=sum(len(t)==steps+1 for t in traces), total=len(ids)))
                pd.DataFrame(rows).to_csv(eval_root / 'rollouts.csv', index=False)
                pd.DataFrame(scores).to_csv(eval_root / 'NRMSE.csv', index=False)
                print('EVALUATED', split, tag, mode, sum(len(t)==steps+1 for t in traces), '/', len(ids), flush=True)
    frame = pd.DataFrame(rows)
    summary = []
    for (method, predictor, mode), group in frame.groupby(['method', 'predictor', 'mode']):
        finite = group.loc[group.complete, 'full_worst9']
        summary.append(dict(method=method, predictor=predictor, split=split, mode=mode,
            complete=int(group.complete.sum()), total=len(group), joint9_le_0p1=int(group.joint9_le_0p1.sum()),
            median_worst9=float(finite.median()) if len(finite) else np.nan,
            p90_worst9=float(finite.quantile(.9)) if len(finite) else np.nan,
            mass_sum_max=float(group.mass_sum_max.max()), elemental_max=float(group.elemental_max.max())))
    pd.DataFrame(summary).to_csv(eval_root / 'summary.csv', index=False)
    return summary


def freeze_selection(root, val_summary):
    # Raw coverage first, then whole-chemistry coverage; no temperature-only rank.
    candidates = [v for v in val_summary if v['mode']=='raw' and v['predictor'].endswith('mean3')]
    ordered = sorted(candidates, key=lambda v: (-v['complete'], -v['joint9_le_0p1'],
                     v['median_worst9'] if np.isfinite(v['median_worst9']) else float('inf')))
    write_json(root / 'selection_frozen.json', dict(test_accessed=False,
        method=ordered[0]['method'], criterion='validation raw completion, then joint9<=0.1 count, then median worst9',
        note='gamma=.2 and methods fixed in advance; no gamma search; not a general chemistry optimum'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/temporal_pilot/run_001')
    parser.add_argument('--seconds', type=float, default=150.)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise FileExistsError('Use a new experiment path')
    if args.seconds <= 0 or len(args.seeds) != 3:
        raise ValueError('Positive budget and exactly three seeds required')
    root.mkdir(parents=True)
    source = Path('results/long10ms_data/trajectories.npz')
    pairs = Path('results/long10ms_data/training_pairs.npz')
    started = time.perf_counter()
    try:
        prep = prepare_training(source, pairs)
        preprocessing = dict(output_mean=prep['norm'].mean.tolist(), output_scale=prep['norm'].scale.tolist(),
            input_mean=prep['feature_norm'].mean.tolist(), input_scale=prep['feature_norm'].scale.tolist(),
            fit_scope='selected training current and 2us next states, identical for all methods', gamma=GAMMA)
        write_json(root / 'preprocessing.json', preprocessing)
        config = dict(methods=list(METHODS), seeds=args.seeds, gamma=GAMMA, soft_constraint=False,
            active_seconds_per_member=args.seconds, sequential=True, training_from_scratch=True,
            batch_size=512, branch_layers=[64,64], trunk_layers=[32], latent_width=7,
            interval_k=[1,2,4,8], interval_probabilities=[.5,1/6,1/6,1/6],
            multistep_curriculum=[1,4,16,64], curriculum_active_time_fractions=[.25]*4,
            optimizer='Adam', learning_rates=[.001,.0003,.0001], learning_rate_boundaries=[.6,.9], clipnorm=5.,
            plateau_stopping=False, dt=DT, evaluation_steps=5000, train_pairs=len(prep['x']),
            training_ids=prep['train_ids'].tolist(), source=str(source), source_sha256=SOURCE_SHA,
            pairs=str(pairs), pairs_sha256=PAIR_SHA, raw_is_primary=True,
            source_git_head=os.popen('git rev-parse HEAD').read().strip(),
            python=platform.python_version(), tensorflow=tf.__version__,
            valid_pairs_by_k={str(k):len(valid_starts(prep['parent'],k,5000)) for k in (1,2,4,8,16,64)})
        write_json(root / 'config.json', config)
        write_json(root / 'source_snapshot.json', {str(p):sha(p) for p in
            [Path(__file__),Path(__file__).with_name('temporal_report.py'),Path(__file__).with_name('transforms.py'),
             Path(__file__).with_name('vector_physics.py'),Path(__file__).with_name('deeponet.py')]})
        training = []
        for seed in args.seeds:
            for method in METHODS:
                training.append(train_one(method, seed, prep, root, args.seconds))
                write_json(root / 'final_members.json', training)
        pd.DataFrame([{k:v for k,v in m.items() if k!='by_length'} for m in training]).to_csv(root / 'training.csv', index=False)
        write_json(root / 'training_complete.json', dict(models=len(training), plateau=0, time_cap=len(training),
                                                       active_seconds=sum(m['active_seconds'] for m in training)))
        val = evaluate_split(root, source, args.seeds, 'val', preprocessing)
        freeze_selection(root, val)
        evaluate_split(root, source, args.seeds, 'test', preprocessing)
        from .temporal_report import produce_report, verify
        produce_report(root, source)
        verification = verify(root, source, pairs)
        write_json(root / 'verification.json', verification)
        if verification['status'] != 'passed':
            raise RuntimeError('Independent verification failed')
        write_json(root / 'complete.json', dict(status='passed', wall_seconds=time.perf_counter()-started,
                                               verification_sha256=sha(root / 'verification.json')))
        write_json(root / 'status.json', dict(phase='complete', status='passed'))
        print('COMPLETE', root, flush=True)
    except Exception as exc:
        write_json(root / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
