"""Experiments added in response to review (Astra):

  (a) Star, weak signal: single-pass NONUNIFORM (geometric) grid baselines vs
      uniform single-pass vs adaptive, at matched forwarded sample count M.
  (b) Multihop, weak signal: single-pass on the GLOBAL grid [0, alpha] (as in
      PX23) vs single-pass on the LOCAL interval [0, alpha_x] vs adaptive, so
      that the effects of local rescaling and of two-pass refinement are
      separated.

Run:  PYTHONPATH=. python3 sim/review_experiments.py geo   -> sim/results/exp_geo_weak.npz
      PYTHONPATH=. python3 sim/review_experiments.py mhglob -> sim/results/exp_multihop_globalgrid.npz
"""
import sys, os
import numpy as np
from sim.algorithms import (_empirical_cdf_samples, sample_and_forward_star,
                            adaptive_sample_and_forward_star,
                            qute_sample_and_forward, adaptive_qute_sample_and_forward)
from sim.data import generate_star_network_weak, sample_pvals_independent, random_graph_neighbors
from sim.verify import fdr_power
from sim.experiments import save_npz, pooled_bh_tau


def geometric_locations(alpha, M, floor):
    """M locations: 0, then M-1 geometrically spaced points from `floor` up to alpha."""
    if M < 3:
        return np.linspace(0, alpha, M)
    return np.concatenate([[0.0], np.geomspace(floor, alpha, M - 1)])


def sf_star_locations(node_pvals, alpha, locations):
    """Single-pass sample-and-forward with arbitrary (shared) locations."""
    m_i = np.array([len(p) for p in node_pvals], dtype=float)
    m_total = m_i.sum()
    F = np.zeros(len(locations))
    for i, p in enumerate(node_pvals):
        F += (m_i[i] / m_total) * _empirical_cdf_samples(p, locations)
    crossing = F >= locations / alpha
    if not crossing.any():
        return 0.0
    I = int(np.max(np.where(crossing)[0]))
    return float(alpha * F[I])


def adaptive_sf_star_locations(node_pvals, alpha, coarse, M2):
    """Two-pass adaptive S&F with an arbitrary shared coarse grid `coarse`
    (sorted, coarse[0]=0, coarse[-1]=alpha) and M2 uniform refined samples on
    the coarse bin containing the round-1 crossing.  Theorem 1 (FDR) applies
    verbatim: its proof uses only that the refined grid lives on the crossing
    bin, not that the coarse grid is equispaced."""
    m_i = np.array([len(p) for p in node_pvals], dtype=float)
    m_total = m_i.sum()
    F1 = np.zeros(len(coarse))
    for i, p in enumerate(node_pvals):
        F1 += (m_i[i] / m_total) * _empirical_cdf_samples(p, coarse)
    c1 = F1 >= coarse / alpha
    j = int(np.max(np.where(c1)[0]))
    if j == len(coarse) - 1:
        return float(alpha * F1[j])
    Z2 = np.linspace(coarse[j], coarse[j + 1], M2)
    F2 = np.zeros(M2)
    for i, p in enumerate(node_pvals):
        F2 += (m_i[i] / m_total) * _empirical_cdf_samples(p, Z2)
    c2 = F2 >= Z2 / alpha
    if not c2.any():
        return float(alpha * F1[j])
    return float(alpha * F2[int(np.max(np.where(c2)[0]))])


def exp_geo_weak(n_trials=2000, alpha=0.2, N=100, lam=3,
                 M_grid=(6, 8, 10, 12, 16, 20), seed=51, tag=''):
    rng = np.random.default_rng(seed)
    mbar = N * lam  # expected pooled size (design constant for the geometric grids)
    cols = ['BH', 'uniform', 'geo_m', 'geo_sqrtm', 'adaptive', 'geo_adaptive']
    K = len(M_grid)
    fdr = np.zeros((K, len(cols))); powr = np.zeros((K, len(cols)))
    for k, M in enumerate(M_grid):
        M1 = M // 2; M2 = M - M1
        loc_geo_m = geometric_locations(alpha, M, alpha / mbar)
        loc_geo_s = geometric_locations(alpha, M, alpha / np.sqrt(mbar))
        loc_geo_c = geometric_locations(alpha, M1, alpha / np.sqrt(mbar))  # coarse grid for hybrid
        for _ in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, nl = sample_pvals_independent(nodes, rng)
            taus = [pooled_bh_tau(p, alpha),
                    sample_and_forward_star(p, alpha, M),
                    sf_star_locations(p, alpha, loc_geo_m),
                    sf_star_locations(p, alpha, loc_geo_s),
                    adaptive_sample_and_forward_star(p, alpha, M1, M2),
                    adaptive_sf_star_locations(p, alpha, loc_geo_c, M2)]
            for c, tau in enumerate(taus):
                f, t = fdr_power(p, nl, tau); fdr[k, c] += f; powr[k, c] += t
        fdr[k] /= n_trials; powr[k] /= n_trials
        print(f'M={M:2d}: ' + '  '.join(f'{c}={powr[k,i]:.3f}' for i, c in enumerate(cols)), flush=True)
    save_npz('exp_geo_weak' + tag, M_grid=np.array(M_grid), fdr=fdr, power=powr, cols=np.array(cols))


def qute_sf_global_grid(node_pvals, neighbors, alpha, M):
    """PX23 multihop: every node samples the GLOBAL grid on [0, alpha]; node x
    pools its neighbourhood and thresholds with local test size alpha_x."""
    N = len(node_pvals)
    m_i = np.array([len(p) for p in node_pvals], dtype=float)
    m_total = m_i.sum()
    locations = np.linspace(0, alpha, M)
    F_i = [_empirical_cdf_samples(p, locations) for p in node_pvals]
    tau0 = np.zeros(N)
    for x in range(N):
        nbrs = neighbors[x]
        n_x = sum(m_i[i] for i in nbrs)
        alpha_x = (n_x / m_total) * alpha
        F_pool = sum((m_i[i] / n_x) * F_i[i] for i in nbrs)
        crossing = F_pool >= locations / alpha_x
        if crossing.any():
            I = int(np.max(np.where(crossing)[0]))
            tau0[x] = float(alpha_x * F_pool[I])
    return np.array([max(tau0[i] for i in neighbors[x]) for x in range(N)])


def _mh_eval(p_list, null_list, taus):
    N = len(p_list)
    V = R = nonn = 0
    for i in range(N):
        rej = p_list[i] <= taus[i]
        V += int((rej & null_list[i]).sum()); R += int(rej.sum())
        nonn += int((~null_list[i]).sum())
    return V / max(R, 1), (R - V) / max(nonn, 1)


def exp_multihop_globalgrid(n_trials=1000, alpha=0.2, N=50, lam=3, M_total=10,
                            degree_grid=(2, 4, 6, 8, 12, 16, 20), seed=42, tag=''):
    rng = np.random.default_rng(seed)
    M1 = M_total // 2; M2 = M_total - M1
    K = len(degree_grid)
    out = {k: np.zeros(K) for k in ['fdr_glob', 'pow_glob', 'fdr_loc', 'pow_loc', 'fdr_adap', 'pow_adap']}
    for k, deg in enumerate(degree_grid):
        nbrs = random_graph_neighbors(N, avg_degree=float(deg), seed=seed + k)
        for _ in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, nl = sample_pvals_independent(nodes, rng)
            for name, taus in [('glob', qute_sf_global_grid(p, nbrs, alpha, M_total)),
                               ('loc', qute_sample_and_forward(p, nbrs, alpha, M_total)),
                               ('adap', adaptive_qute_sample_and_forward(p, nbrs, alpha, M1, M2))]:
                f, t = _mh_eval(p, nl, taus)
                out['fdr_' + name][k] += f; out['pow_' + name][k] += t
        for key in out: out[key][k] /= n_trials
        print(f'deg={deg:2d}: global={out["pow_glob"][k]:.3f} local={out["pow_loc"][k]:.3f} '
              f'adaptive={out["pow_adap"][k]:.3f}', flush=True)
    save_npz('exp_multihop_globalgrid' + tag, degree_grid=np.array(degree_grid), **out)


if __name__ == '__main__':
    {'geo': exp_geo_weak, 'mhglob': exp_multihop_globalgrid}[sys.argv[1]]()
