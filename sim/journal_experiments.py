"""Experiments for the journal extension.  Each function accepts `tag` so it can
be run in chunks; merge with merge_chunks().

  J1  K-round zoom: power / threshold gap / certificate vs total samples   (star)
  J2  eps-certified stopping: rounds, power, FDR, certified vs actual misses
  J3  Multihop K-round with per-node certificates vs uncompressed QuTE
  J4  Sparse/weak regime: pi1 -> 0, geometric vs uniform vs adaptive vs K-round
  J5  Heterogeneous budgets: equal vs sqrt-size vs oracle allocation
"""
import sys
import numpy as np
from scipy.stats import norm
from sim.algorithms import bh_reject, sample_and_forward_star, adaptive_sample_and_forward_star, _empirical_cdf_samples
from sim.journal_algorithms import kround_star, kround_qute, qute_uncompressed, hetero_star, allocate
from sim.review_experiments import geometric_locations, sf_star_locations, adaptive_sf_star_locations
from sim.data import generate_star_network_weak, sample_pvals_independent, random_graph_neighbors
from sim.verify import fdr_power
from sim.experiments import save_npz, pooled_bh_tau


def generate_star_network_sparse(N, lam, pi1, mu_lo, mu_hi, rng):
    nodes = []
    for i in range(1, N + 1):
        m_i = max(1, rng.poisson(lam))
        mu_base = mu_lo + (mu_hi - mu_lo) * (i / N)
        nodes.append({'m': m_i, 'pi1': pi1, 'mu_base': mu_base, 'is_null': rng.random(m_i) >= pi1})
    return nodes


def _bh_stats(p, nl, alpha):
    allp = np.concatenate(p); m = len(allp)
    mask = bh_reject(allp, alpha)
    return int(mask.sum()), alpha * mask.sum() / m, m


# ---------------------------------------------------------------- J1
def exp_J1(n_trials=1500, alpha=0.2, N=100, lam=3, seed=101, tag='',
           per_round=(3, 4, 5), K_max=5, refute=1):
    rng = np.random.default_rng(seed)
    rows = []   # (M, K, samples, power, fdr, gap, cert_missed, true_missed, bh_power)
    for M in per_round:
        acc = np.zeros((K_max, 7))
        for _ in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, nl = sample_pvals_independent(nodes, rng)
            R_bh, tau_bh, m = _bh_stats(p, nl, alpha)
            fb, pb = fdr_power(p, nl, tau_bh)
            allp = np.concatenate(p)
            res = kround_star(p, alpha, [M] * K_max, refute=refute, redirect=True)
            for k in range(K_max):
                th, tb, C, smp = res['trace'][min(k, len(res['trace']) - 1)]
                f, t = fdr_power(p, nl, th)
                R_hat = int((allp <= th).sum())
                acc[k] += [t, f, tau_bh - th, m * C, R_bh - R_hat, pb, smp]
        for k in range(K_max):
            a = acc[k] / n_trials
            rows.append([M, k + 1, a[6], a[0], a[1], a[2], a[3], a[4], a[5]])
            print(f'M={M} K={k+1} S={a[6]:5.2f}: pow={a[0]:.3f} fdr={a[1]:.3f} gap={a[2]:.4f} '
                  f'cert_missed={a[3]:.2f} true_missed={a[4]:.2f} BH={a[5]:.3f}', flush=True)
    # uniform single-pass reference at matched S
    uni = []
    for S in sorted(set(int(round(r[2])) for r in rows) | {6, 8, 10, 12, 16, 20, 25}):
        pw = fd = 0.0
        rng2 = np.random.default_rng(seed + S)
        for _ in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng2)
            p, nl = sample_pvals_independent(nodes, rng2)
            f, t = fdr_power(p, nl, sample_and_forward_star(p, alpha, S)); pw += t; fd += f
        uni.append([S, pw / n_trials, fd / n_trials])
        print(f'uniform S={S}: pow={pw/n_trials:.3f}', flush=True)
    save_npz('expJ1' + tag, rows=np.array(rows), uniform=np.array(uni))


# ---------------------------------------------------------------- J2
def exp_J2(n_trials=1500, alpha=0.2, N=100, lam=3, seed=102, tag='',
           eps_grid=(0.02, 0.05, 0.1, 0.2, 0.5, 1.0), M=4, K_max=6, refute=1):
    rng = np.random.default_rng(seed)
    out = np.zeros((len(eps_grid), 8))  # rounds, samples, power, fdr, cert_missed, true_missed, bh_power, certified_frac
    for _ in range(n_trials):
        nodes = generate_star_network_weak(N, lam, rng)
        p, nl = sample_pvals_independent(nodes, rng)
        R_bh, tau_bh, m = _bh_stats(p, nl, alpha)
        _, pb = fdr_power(p, nl, tau_bh); allp = np.concatenate(p)
        for e, eps in enumerate(eps_grid):
            res = kround_star(p, alpha, [M] * K_max, eps=eps, refute=refute, redirect=True, uncertified='largest')
            f, t = fdr_power(p, nl, res['tau'])
            out[e] += [res['rounds'], res['samples'], t, f, m * res['C'], R_bh - int((allp <= res['tau']).sum()), pb, float(res['certified'])]
    out /= n_trials
    for e, eps in enumerate(eps_grid):
        print(f'eps={eps}: rounds={out[e,0]:.2f} samples={out[e,1]:.1f} pow={out[e,2]:.3f} fdr={out[e,3]:.3f} '
              f'cert={out[e,4]:.2f} true_missed={out[e,5]:.2f} BH={out[e,6]:.3f} certified={out[e,7]:.3f}', flush=True)
    save_npz('expJ2' + tag, eps_grid=np.array(eps_grid), out=out)


# ---------------------------------------------------------------- J3
def _mh_eval(p, nl, taus):
    V = R = nonn = 0
    for i in range(len(p)):
        rej = p[i] <= taus[i]
        V += int((rej & nl[i]).sum()); R += int(rej.sum()); nonn += int((~nl[i]).sum())
    return V / max(R, 1), (R - V) / max(nonn, 1), R


def exp_J3(n_trials=600, alpha=0.2, N=50, lam=3, seed=103, tag='',
           degree_grid=(2, 4, 8, 16), M=4, K_grid=(1, 2, 3, 4), eps=0.2, refute=1):
    rng = np.random.default_rng(seed)
    # rows: deg, K, power, fdr, qute_power, cert_missed(total), true_missed(total), rounds(eps-stop mean), power(eps-stop), fdr(eps-stop)
    rows = []
    for deg in degree_grid:
        nbrs = random_graph_neighbors(N, float(deg), seed + deg)
        acc = {K: np.zeros(4) for K in K_grid}; acc_stop = np.zeros(4); qp = 0.0; cert_frac = 0.0; mean_samples = 0.0
        for _ in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, nl = sample_pvals_independent(nodes, rng)
            ref = qute_uncompressed(p, nbrs, alpha)
            fq, pq, Rq = _mh_eval(p, nl, ref); qp += pq
            for K in K_grid:
                res = kround_qute(p, nbrs, alpha, [M] * K, refute=refute, redirect=True)
                f, t, R = _mh_eval(p, nl, res['tau'])
                true_missed = sum(int(((p[x] > res['tau'][x]) & (p[x] <= ref[x])).sum()) for x in range(N))
                acc[K] += [t, f, res['missed'].sum(), true_missed]
            res = kround_qute(p, nbrs, alpha, [M] * max(K_grid), eps=eps, refute=refute, redirect=True, uncertified='largest')
            f, t, R = _mh_eval(p, nl, res['tau'])
            acc_stop += [t, f, res['rounds'].mean(), res['missed'].sum()]
            cert_frac += res['certified'].mean(); mean_samples += res['samples'].mean()
        for K in K_grid:
            a = acc[K] / n_trials
            rows.append([deg, K, a[0], a[1], qp / n_trials, a[2], a[3], 0, 0, 0])
            print(f'deg={deg} K={K}: pow={a[0]:.3f} fdr={a[1]:.3f} QuTE={qp/n_trials:.3f} cert={a[2]:.2f} true={a[3]:.2f}', flush=True)
        s = acc_stop / n_trials
        rows.append([deg, -1, s[0], s[1], qp / n_trials, s[3], cert_frac / n_trials, s[2], mean_samples / n_trials, s[1]])
        print(f'deg={deg} eps-stop: pow={s[0]:.3f} fdr={s[1]:.3f} rounds={s[2]:.2f} samples={mean_samples/n_trials:.1f} cert={s[3]:.2f} certified={cert_frac/n_trials:.3f}', flush=True)
    save_npz('expJ3' + tag, rows=np.array(rows))


# ---------------------------------------------------------------- J4
def exp_J4(n_trials=400, alpha=0.2, N=300, lam=10, seed=104, tag='',
           pi1_grid=(0.3, 0.1, 0.03, 0.01), M=12, mu=(2.5, 3.5)):
    rng = np.random.default_rng(seed)
    mbar = N * lam
    cols = ['BH', 'uniform', 'geo_m', 'adaptive', 'kround', 'geo_adaptive']
    powr = np.zeros((len(pi1_grid), len(cols))); fdr = np.zeros_like(powr); tau_star = np.zeros(len(pi1_grid))
    M1 = M // 2; M2 = M - M1
    for k, pi1 in enumerate(pi1_grid):
        loc_geo = geometric_locations(alpha, M, alpha / mbar)
        loc_geo_c = geometric_locations(alpha, M1, alpha / mbar)
        for _ in range(n_trials):
            nodes = generate_star_network_sparse(N, lam, pi1, mu[0], mu[1], rng)
            p, nl = sample_pvals_independent(nodes, rng)
            tau_bh = pooled_bh_tau(p, alpha); tau_star[k] += tau_bh
            taus = [tau_bh, sample_and_forward_star(p, alpha, M), sf_star_locations(p, alpha, loc_geo),
                    adaptive_sample_and_forward_star(p, alpha, M1, M2),
                    kround_star(p, alpha, [4] * (M // 4), refute=0)['tau'],
                    adaptive_sf_star_locations(p, alpha, loc_geo_c, M2)]
            for c, tau in enumerate(taus):
                f, t = fdr_power(p, nl, tau); fdr[k, c] += f; powr[k, c] += t
        powr[k] /= n_trials; fdr[k] /= n_trials; tau_star[k] /= n_trials
        print(f'pi1={pi1}: tauBH~{tau_star[k]:.4f} ' + ' '.join(f'{c}={powr[k,i]:.3f}' for i, c in enumerate(cols)), flush=True)
    save_npz('expJ4' + tag, pi1_grid=np.array(pi1_grid), power=powr, fdr=fdr, cols=np.array(cols), tau_bh=tau_star)


# ---------------------------------------------------------------- J5
def _gprime(mu, t):
    z = norm.ppf(1 - t / 2)
    return np.exp(-mu ** 2 / 2) * np.cosh(mu * z)


def exp_J5(n_trials=500, alpha=0.2, N=100, seed=105, tag='', per_node=10,
           lam_range=(1, 30), rules=('equal', 'sqrt', 'oracle')):
    rng = np.random.default_rng(seed)
    total = per_node * N
    cols = ['BH'] + list(rules)
    powr = np.zeros(len(cols)); fdr = np.zeros(len(cols))
    for _ in range(n_trials):
        # heterogeneous sizes: lambda_i log-uniform on lam_range
        lams = np.exp(rng.uniform(np.log(lam_range[0]), np.log(lam_range[1]), size=N))
        nodes = []
        for i in range(N):
            m_i = max(1, rng.poisson(lams[i])); mu_base = 1.0 + 1.5 * ((i + 1) / N)
            nodes.append({'m': m_i, 'pi1': 0.3, 'mu_base': mu_base, 'is_null': rng.random(m_i) >= 0.3})
        p, nl = sample_pvals_independent(nodes, rng)
        m_i = np.array([len(x) for x in p])
        tau_bh = pooled_bh_tau(p, alpha)
        f, t = fdr_power(p, nl, tau_bh); fdr[0] += f; powr[0] += t
        dens = 0.7 + 0.3 * np.array([_gprime(nd['mu_base'], max(tau_bh, 1e-4)) for nd in nodes])
        for c, rule in enumerate(rules, start=1):
            M1 = allocate(m_i, total // 2, rule, density=dens); M2 = allocate(m_i, total - total // 2, rule, density=dens)
            tau = hetero_star(p, alpha, M1, M2)
            f, t = fdr_power(p, nl, tau); fdr[c] += f; powr[c] += t
    powr /= n_trials; fdr /= n_trials
    print(' '.join(f'{c}: pow={powr[i]:.3f} fdr={fdr[i]:.3f}' for i, c in enumerate(cols)), flush=True)
    save_npz('expJ5' + tag, power=powr, fdr=fdr, cols=np.array(cols), per_node=per_node)


def merge_chunks(name, tags, key='rows'):
    import os
    parts = [np.load(f'sim/results/{name}{t}.npz', allow_pickle=True) for t in tags]
    out = {}
    for k in parts[0].files:
        arrs = [pt[k] for pt in parts]
        out[k] = np.concatenate(arrs) if arrs[0].ndim >= 1 and k in (key, 'uniform') else arrs[0]
    np.savez(f'sim/results/{name}.npz', **out)


if __name__ == '__main__':
    {'J1': exp_J1, 'J2': exp_J2, 'J3': exp_J3, 'J4': exp_J4, 'J5': exp_J5}[sys.argv[1]]()
