"""Immutable, trajectory-disjoint Cantera benchmark generation."""
import hashlib
import json
from pathlib import Path
import cantera as ct
import numpy as np

SPECIES = ['H2','H','O','O2','OH','H2O','HO2','H2O2','N2']


def mechanism():
    full = ct.Solution('h2o2.yaml')
    species = [full.species(s) for s in SPECIES]
    reactions = [r for r in full.reactions()
                 if set(r.reactants) | set(r.products) <= set(SPECIES)]
    return ct.Solution(thermo='ideal-gas', kinetics='gas', species=species, reactions=reactions)


def generate(path, trajectories=90, steps=200, dt=2e-6, seed=20261029):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Never overwrite a dataset; use a new path')
    rng = np.random.default_rng(seed)
    order = rng.permutation(trajectories)
    split = np.empty(trajectories, dtype='U5')
    split[order[:int(.6*trajectories)]] = 'train'
    split[order[int(.6*trajectories):int(.8*trajectories)]] = 'val'
    split[order[int(.8*trajectories):]] = 'test'
    states, pressures, initials = [], [], []
    minimum = 0.
    for i in range(trajectories):
        gas = mechanism()
        t, p, phi = rng.uniform(900,1600), rng.uniform(1,5)*ct.one_atm, rng.uniform(.5,2.)
        gas.TP = t,p
        gas.set_equivalence_ratio(phi, 'H2', 'O2:1,N2:3.76')
        reactor = ct.IdealGasConstPressureReactor(gas, clone=True)
        sim = ct.ReactorNet([reactor]); sim.rtol=1e-10; sim.atol=1e-20
        trace = []
        for j in range(steps+1):
            if j: sim.advance(j*dt)
            y = reactor.phase.Y
            minimum = min(minimum,float(y.min()))
            if y.min() < -1e-14: raise RuntimeError('Cantera negative beyond roundoff')
            trace.append(np.r_[reactor.T, np.maximum(y,0)])
        states.append(trace); pressures.append(p); initials.append([t,p,phi])
    gas = mechanism()
    elements = ['H','O','N']
    matrix = np.array([[gas.n_atoms(s,e)*gas.atomic_weight(e)/gas.molecular_weights[k]
                        for k,s in enumerate(SPECIES)] for e in elements])
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path, states=states, pressures=pressures, initials=initials,
                        split=split, species=SPECIES, elements=elements, element_matrix=matrix, dt=dt)
    metadata = dict(seed=seed, trajectories=trajectories, steps=steps, dt=dt,
                    cantera=ct.__version__, species=SPECIES, n_reactions=gas.n_reactions,
                    min_solver_mass_fraction=minimum,
                    note='Only negative Cantera roundoff in [-1e-14,0) replaced by zero; no renormalization.',
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    return metadata


def samples(data, split):
    ids = np.flatnonzero(data['split']==split)
    states = data['states'][ids]
    x = np.concatenate([states[:,:-1], np.broadcast_to(data['pressures'][ids,None,None],
                         (*states[:,:-1].shape[:2],1))],axis=-1).reshape(-1,11)
    y = states[:,1:].reshape(-1,10)
    return x,y,ids
