"""Per-state cost of residual selection pieces vs network inference vs Cantera CVODE, H2 (9 sp) and GRI-Mech 3.0 (53 sp)."""
import time, json, numpy as np, cantera as ct, os, sys
sys.path.insert(0, os.getcwd())
from kinetics.data import mechanism
np.random.seed(0)
def tm(f, n):
    f(); t = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - t) / n
out = {}
for name, gas, fuel, T0 in [('H2_9sp', mechanism(), 'H2', 1050.), ('GRI30_53sp', ct.Solution('gri30.yaml'), 'CH4', 1400.)]:
    N = gas.n_species; R = gas.n_reactions
    gas.TP = T0, ct.one_atm; gas.set_equivalence_ratio(1.0, fuel, 'O2:1,N2:3.76')
    r = ct.IdealGasConstPressureReactor(gas, clone=True); net = ct.ReactorNet([r]); net.rtol, net.atol = 1e-10, 1e-20
    # advance to mid-ignition (T0+400 K) to get a representative stiff state
    t = 0.
    while r.T < T0 + 400: t += 2e-6; net.advance(t)
    Y = r.phase.Y.copy(); T = r.T; P = r.phase.P
    g = gas; k = [0]
    def rates():  # perturb T so Cantera cannot reuse cached rates
        k[0] += 1; g.TPY = T * (1 + 1e-9 * (k[0] % 7)), P, Y; return g.net_production_rates
    t_rate = tm(rates, 3000)
    g.TPY = T, P, Y
    def jac():
        k[0] += 1; g.TPY = T * (1 + 1e-9 * (k[0] % 7)), P, Y; return g.net_production_rates_ddCi
    try: t_jac = tm(jac, 300)
    except Exception as e: t_jac = float('nan')
    A = np.random.rand(20000, N + 1, N + 1) + (N + 1) * np.eye(N + 1); b = np.random.rand(20000, N + 1, 1)
    t_solve = tm(lambda: np.linalg.solve(A, b), 3) / 20000      # batched LU + solve, per state
    # network 1600/800/400 GELU-sized matmuls, float32, batch 20000, per state
    W = [np.random.rand(a, c).astype('float32') for a, c in [(N + 2, 1600), (1600, 800), (800, 400), (400, N + 2)]]
    X = np.random.rand(20000, N + 2).astype('float32')
    def nn():
        h = X
        for w in W: h = np.tanh(h @ w)
        return h
    t_nn = tm(nn, 3) / 20000
    macs = sum(a * c for a, c in [(N + 2, 1600), (1600, 800), (800, 400), (400, N + 2)])
    # CVODE: cost per 2 us output step over induction+ignition (to T0+1000 K or 2000 steps), two tolerances
    cv = {}
    for rtol, atol in [(1e-10, 1e-20), (1e-6, 1e-12)]:
        gas.TP = T0, ct.one_atm; gas.set_equivalence_ratio(1.0, fuel, 'O2:1,N2:3.76')
        r2 = ct.IdealGasConstPressureReactor(gas, clone=True); n2 = ct.ReactorNet([r2]); n2.rtol, n2.atol = rtol, atol
        st = 0; t0 = time.perf_counter(); tt = 0.
        while st < 2000 and r2.T < T0 + 1000: st += 1; tt += 2e-6; n2.advance(tt)
        el = time.perf_counter() - t0; s = n2.solver_stats if hasattr(n2, 'solver_stats') else {}
        cv[f'rtol{rtol:g}'] = dict(per_output_step_s=el / st, output_steps=st, rhs_evals_per_step=s.get('rhs_evals', np.nan) / st if s else None,
                                   jac_evals_per_step=s.get('jac_evals', np.nan) / st if s else None, internal_steps_per_step=s.get('steps', np.nan) / st if s else None)
    M = 10
    sel = (M + 1) * t_rate + t_jac + M * t_solve          # residual selection (analytic Jacobian, python-level calls)
    imp = 3 * t_rate + t_jac + 3 * t_solve                  # direct trapezoidal step, ~3 Newton iterations
    out[name] = dict(species=N, reactions=R, T_state=T, rate_eval_s=t_rate, analytic_jac_s=t_jac, lu_solve_per_state_s=t_solve,
                     nn_member_per_state_s=t_nn, nn_member_macs=macs, ensemble10_per_state_s=M * t_nn,
                     selection_check_s=sel, direct_implicit_step_s=imp, cvode=cv)
    print(json.dumps({name: out[name]}, indent=1), flush=True)
json.dump(out, open('results/cost_benchmark/bench.json', 'w'), indent=1)
