"""Experiments for revision 5.  Every run stores PER-TRIAL outcomes so that
tables can report Monte Carlo standard errors and paired differences.
All counts use endpoint reuse; every method is run through kround_star, so
the certificate of Lemma 1 is available for fixed baselines too (computed
from the counts they already transmit, with no extra queries).

  S1  sparse regime (lead experiment)                       -> expS1_pi{pi}.npz
  S2  weak regime, nonuniform grids at reuse-matched counts -> expS2_S{S}.npz
  S3  weak regime, fixed-K zoom and certificate tightness   -> expS3_M{M}.npz, expS3_uniform.npz
  S4  weak regime, certified stopping vs eps (3 variants)   -> expS4_{variant}.npz
  S5  multihop                                              -> expS5_deg{d}.npz
  S6  heterogeneous budgets                                 -> expS6_B{B}.npz
"""
import sys
import numpy as np
from scipy.stats import norm
from sim.algorithms import bh_reject
from sim.journal_algorithms import kround_star, kround_qute, qute_uncompressed, hetero_star_cert, allocate
from sim.review_experiments import geometric_locations
from sim.journal_experiments import generate_star_network_sparse, _mh_eval
from sim.data import generate_star_network_weak, sample_pvals_independent, random_graph_neighbors
from sim.verify import fdr_power
from sim.experiments import save_npz

EPS = 0.2


def _bh(p, alpha):
    allp = np.concatenate(p); m = len(allp)
    R = int(bh_reject(allp, alpha).sum())
    return allp, m, R, alpha * R / m


def _record(p, nl, allp, m, R_bh, res, eps=EPS):
    """power, fdp, rejections, exact recovery, counts, exchanges, relatively certified,
    exactly certified, certified bound mC, actual missed."""
    tau = res['tau']
    f, t = fdr_power(p, nl, tau)
    R = int((allp <= tau).sum())
    mC = m * res['C']
    exact_cert = mC < 0.5
    rel_cert = (tau > 0) and (res['tau_bar'] <= (1 + eps) * tau) and not exact_cert
    return [t, f, R, float(R == R_bh), res['samples'], res['exchanges'], float(rel_cert),
            float(exact_cert), mC, R_bh - R]


FIELDS = ['power', 'fdp', 'R', 'exact', 'counts', 'exch', 'relcert', 'exactcert', 'mC', 'missed']


# ---------------------------------------------------------------- S1
S1_METHODS = ['BH', 'uniform12', 'geo12', 'twopass77', 'zoom4x5', 'bisect3x10', 'hybrid77',
              'cert_fixed', 'cert_fixed_geo', 'cert_redirect_abstain', 'cert_fixed_levelcorr']


def exp_S1(pi1, n_trials=300, alpha=0.2, N=300, lam=10, eps=EPS, seed=205):
    rng = np.random.default_rng(seed + int(1000 * pi1))
    mbar = N * lam
    geo12 = geometric_locations(alpha, 12, alpha / mbar)
    geo7 = geometric_locations(alpha, 7, alpha / mbar)
    geo6 = geometric_locations(alpha, 6, alpha / mbar)
    a2 = alpha / (1 + eps)
    geo6_2 = geometric_locations(a2, 6, a2 / mbar)
    out = np.zeros((n_trials, len(S1_METHODS), len(FIELDS)))
    tau_bh_all = np.zeros(n_trials)
    for k in range(n_trials):
        nodes = generate_star_network_sparse(N, lam, pi1, 2.5, 3.5, rng)
        p, nl = sample_pvals_independent(nodes, rng)
        allp, m, R_bh, tau_bh = _bh(p, alpha); tau_bh_all[k] = tau_bh
        runs = {
            'uniform12': kround_star(p, alpha, [12]),
            'geo12': kround_star(p, alpha, [12], first_grid=geo12),
            'twopass77': kround_star(p, alpha, [7, 7]),
            'zoom4x5': kround_star(p, alpha, [4] * 5),
            'bisect3x10': kround_star(p, alpha, [3] * 10),
            'hybrid77': kround_star(p, alpha, [7, 7], first_grid=geo7),
            'cert_fixed': kround_star(p, alpha, [4] * 8, eps=eps, refute=1, redirect=False, uncertified='zoom'),
            'cert_fixed_geo': kround_star(p, alpha, [6] + [4] * 7, eps=eps, refute=1, redirect=False,
                                          uncertified='zoom', first_grid=geo6),
            'cert_redirect_abstain': kround_star(p, alpha, [4] * 8, eps=eps, refute=1, redirect=True,
                                                 uncertified='abstain'),
            'cert_fixed_levelcorr': kround_star(p, a2, [4] * 8, eps=eps, refute=1, redirect=False, uncertified='zoom'),
        }
        f, t = fdr_power(p, nl, tau_bh)
        out[k, 0] = [t, f, R_bh, 1, 0, 0, 0, 0, 0, 0]
        for c, name in enumerate(S1_METHODS[1:], start=1):
            out[k, c] = _record(p, nl, allp, m, R_bh, runs[name], eps)
    print(f'pi1={pi1}: mean tauBH={tau_bh_all.mean():.4f}', flush=True)
    for c, name in enumerate(S1_METHODS):
        mu = out[:, c].mean(0); se = out[:, c].std(0, ddof=1) / np.sqrt(n_trials)
        d = out[:, c, 0] - out[:, 0, 0]
        print(f'   {name:22s} pow={mu[0]:.3f}({se[0]:.3f}) dBH={d.mean():+.3f}({d.std(ddof=1)/np.sqrt(n_trials):.3f}) '
              f'fdr={mu[1]:.3f} S={mu[4]:.1f} X={mu[5]:.1f} rel={mu[6]:.2f} exc={mu[7]:.2f} exact={mu[3]:.2f}', flush=True)
    save_npz(f'expS1_pi{pi1}', out=out, methods=np.array(S1_METHODS), fields=np.array(FIELDS), tau_bh=tau_bh_all)


# ---------------------------------------------------------------- S2
S2_METHODS = ['BH', 'uniform', 'geo_m', 'geo_sqrtm', 'twopass', 'hybrid']


def exp_S2(S, n_trials=2000, alpha=0.2, N=100, lam=3, seed=251):
    rng = np.random.default_rng(seed + S)
    mbar = N * lam
    M = S // 2 + 1                     # two-pass (M, M) uses 2M - 2 = S counts with reuse
    g_m = geometric_locations(alpha, S, alpha / mbar)
    g_s = geometric_locations(alpha, S, alpha / np.sqrt(mbar))
    g_h = geometric_locations(alpha, M, alpha / np.sqrt(mbar))
    out = np.zeros((n_trials, len(S2_METHODS), len(FIELDS)))
    for k in range(n_trials):
        nodes = generate_star_network_weak(N, lam, rng)
        p, nl = sample_pvals_independent(nodes, rng)
        allp, m, R_bh, tau_bh = _bh(p, alpha)
        f, t = fdr_power(p, nl, tau_bh); out[k, 0] = [t, f, R_bh, 1, 0, 0, 0, 0, 0, 0]
        runs = [kround_star(p, alpha, [S]), kround_star(p, alpha, [S], first_grid=g_m),
                kround_star(p, alpha, [S], first_grid=g_s), kround_star(p, alpha, [M, M]),
                kround_star(p, alpha, [M, M], first_grid=g_h)]
        for c, r in enumerate(runs, start=1):
            out[k, c] = _record(p, nl, allp, m, R_bh, r)
    mu = out.mean(0); se = out.std(0, ddof=1) / np.sqrt(n_trials)
    print(f'S={S}: ' + '  '.join(f'{n}={mu[c,0]:.3f}({se[c,0]:.3f})/S{mu[c,4]:.1f}' for c, n in enumerate(S2_METHODS)), flush=True)
    save_npz(f'expS2_S{S}', out=out, methods=np.array(S2_METHODS), S=S)


# ---------------------------------------------------------------- S3
def exp_S3(M, n_trials=1000, alpha=0.2, N=100, lam=3, K_max=5, seed=201):
    rng = np.random.default_rng(seed)
    out = np.zeros((n_trials, K_max + 1, len(FIELDS)))   # index 0 = BH
    for k in range(n_trials):
        nodes = generate_star_network_weak(N, lam, rng)
        p, nl = sample_pvals_independent(nodes, rng)
        allp, m, R_bh, tau_bh = _bh(p, alpha)
        f, t = fdr_power(p, nl, tau_bh); out[k, 0] = [t, f, R_bh, 1, 0, 0, 0, 0, 0, 0]
        for K in range(1, K_max + 1):
            out[k, K] = _record(p, nl, allp, m, R_bh, kround_star(p, alpha, [M] * K, refute=1, redirect=False))
    mu = out.mean(0)
    for K in range(1, K_max + 1):
        d = out[:, K, 0] - out[:, 0, 0]
        print(f'M={M} K={K}: S={mu[K,4]:.2f} X={mu[K,5]:.2f} pow={mu[K,0]:.3f} dBH={d.mean():+.4f}({d.std(ddof=1)/np.sqrt(n_trials):.4f}) '
              f'mC={mu[K,8]:.2f} missed={mu[K,9]:.2f}', flush=True)
    save_npz(f'expS3_M{M}', out=out, M=M)


def exp_S3_norefute(M=4, n_trials=1000, alpha=0.2, N=100, lam=3, K=5, seed=201):
    rng = np.random.default_rng(seed); mC = np.zeros(n_trials); missed = np.zeros(n_trials); S = np.zeros(n_trials)
    for k in range(n_trials):
        nodes = generate_star_network_weak(N, lam, rng)
        p, nl = sample_pvals_independent(nodes, rng)
        allp, m, R_bh, _ = _bh(p, alpha)
        for KK in range(1, K + 1):          # keep the rng stream identical to exp_S3 (it does not consume rng)
            r = kround_star(p, alpha, [M] * KK, refute=0)
        mC[k] = m * r['C']; missed[k] = R_bh - int((allp <= r['tau']).sum()); S[k] = r['samples']
    print(f'no refutation M={M} K={K}: S={S.mean():.1f} mC={mC.mean():.2f}({mC.std(ddof=1)/np.sqrt(n_trials):.2f}) missed={missed.mean():.2f}')
    save_npz('expS3_norefute', mC=mC, missed=missed, S=S)


def exp_S3_uniform(S_grid=(3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25), n_trials=1000, alpha=0.2, N=100, lam=3, seed=202):
    out = []
    for S in S_grid:
        rng = np.random.default_rng(seed + S); pw = []
        for _ in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, nl = sample_pvals_independent(nodes, rng)
            pw.append(fdr_power(p, nl, kround_star(p, alpha, [S])['tau'])[1])
        pw = np.array(pw); out.append([S, pw.mean(), pw.std(ddof=1) / np.sqrt(n_trials)])
        print(f'uniform S={S}: pow={pw.mean():.3f}({out[-1][2]:.3f})', flush=True)
    save_npz('expS3_uniform', out=np.array(out))


# ---------------------------------------------------------------- S4
def exp_S4(variant, n_trials=900, alpha=0.2, N=100, lam=3, M=4, K_max=6, seed=203,
           eps_grid=(0.02, 0.05, 0.1, 0.2, 0.5, 1.0)):
    kw = dict(fixed=dict(redirect=False, uncertified='zoom'),
              redirect_abstain=dict(redirect=True, uncertified='abstain'),
              fixed_levelcorr=dict(redirect=False, uncertified='zoom'))[variant]
    rng = np.random.default_rng(seed)
    out = np.zeros((n_trials, len(eps_grid) + 1, len(FIELDS)))   # index 0 = BH at alpha
    for k in range(n_trials):
        nodes = generate_star_network_weak(N, lam, rng)
        p, nl = sample_pvals_independent(nodes, rng)
        allp, m, R_bh, tau_bh = _bh(p, alpha)
        f, t = fdr_power(p, nl, tau_bh); out[k, 0] = [t, f, R_bh, 1, 0, 0, 0, 0, 0, 0]
        for e, eps in enumerate(eps_grid, start=1):
            a = alpha / (1 + eps) if variant == 'fixed_levelcorr' else alpha
            res = kround_star(p, a, [M] * K_max, eps=eps, refute=1, **kw)
            out[k, e] = _record(p, nl, allp, m, R_bh, res, eps)
            if variant == 'fixed_levelcorr':      # certification is w.r.t. BH at the corrected level
                _, _, R2, _ = _bh(p, a)
                out[k, e, 9] = R2 - int((allp <= res['tau']).sum())
    mu = out.mean(0); se = out.std(0, ddof=1) / np.sqrt(n_trials)
    for e, eps in enumerate(eps_grid, start=1):
        print(f'{variant} eps={eps}: S={mu[e,4]:.1f} X={mu[e,5]:.2f} pow={mu[e,0]:.3f}({se[e,0]:.3f}) fdr={mu[e,1]:.3f}({se[e,1]:.3f}) '
              f'cert={mu[e,6]+mu[e,7]:.3f} exactcert={mu[e,7]:.3f} exact={mu[e,3]:.3f} mC={mu[e,8]:.2f} missed={mu[e,9]:.2f}', flush=True)
    save_npz(f'expS4_{variant}', out=out, eps_grid=np.array(eps_grid))


# ---------------------------------------------------------------- S5
def exp_S5(deg, n_trials=400, alpha=0.2, N=50, lam=3, M=4, K_grid=(1, 2, 3, 4), eps=EPS, seed=204):
    rng = np.random.default_rng(seed)
    nbrs = random_graph_neighbors(N, float(deg), seed + deg)
    # columns: 0 QuTE, 1..4 fixed K, 5 certified fixed policy, 6 redirect + network abstention
    out = np.zeros((n_trials, 7, 8))   # power fdp certsum missed samples exch nodes_cert all_cert
    for k in range(n_trials):
        nodes = generate_star_network_weak(N, lam, rng)
        p, nl = sample_pvals_independent(nodes, rng)
        ref = qute_uncompressed(p, nbrs, alpha)
        fq, pq, _ = _mh_eval(p, nl, ref); out[k, 0, :2] = [pq, fq]
        runs = [kround_qute(p, nbrs, alpha, [M] * K, refute=1, redirect=False) for K in K_grid]
        runs.append(kround_qute(p, nbrs, alpha, [M] * max(K_grid), eps=eps, refute=1, redirect=False, uncertified='zoom'))
        runs.append(kround_qute(p, nbrs, alpha, [M] * max(K_grid), eps=eps, refute=1, redirect=True,
                                uncertified='largest', network_abstain=True))
        for c, r in enumerate(runs, start=1):
            f, t, _ = _mh_eval(p, nl, r['tau'])
            missed = sum(int(((p[x] > r['tau'][x]) & (p[x] <= ref[x])).sum()) for x in range(N))
            out[k, c] = [t, f, r['missed'].sum(), missed, r['samples'].mean(), r['exchanges'].mean(),
                         r['certified'].mean(), float(r['all_certified'])]
    mu = out.mean(0)
    for c, lab in enumerate(['QuTE', 'K=1', 'K=2', 'K=3', 'K=4', 'cert_fixed', 'redirect_netabstain']):
        d = out[:, c, 0] - out[:, 0, 0]
        print(f'deg={deg} {lab:20s} pow={mu[c,0]:.3f} dQ={d.mean():+.4f}({d.std(ddof=1)/np.sqrt(n_trials):.4f}) fdr={mu[c,1]:.3f} '
              f'cert={mu[c,2]:.2f} missed={mu[c,3]:.2f} S={mu[c,4]:.2f} X={mu[c,5]:.2f} nodes={mu[c,6]:.3f} all={mu[c,7]:.3f}', flush=True)
    save_npz(f'expS5_deg{deg}', out=out)


# ---------------------------------------------------------------- S6
def _gprime(mu, t):
    z = norm.ppf(1 - t / 2)
    return np.exp(-mu ** 2 / 2) * np.cosh(mu * z)


def exp_S6(per_node, n_trials=1000, alpha=0.2, N=100, eps=EPS, seed=105, lam_range=(1, 30)):
    rng = np.random.default_rng(seed + per_node)
    total = per_node * N
    rules = ('equal', 'sqrt', 'oracle')
    out = np.zeros((n_trials, 4, 5))   # power fdp relcert exactcert exact
    for k in range(n_trials):
        lams = np.exp(rng.uniform(np.log(lam_range[0]), np.log(lam_range[1]), size=N))
        nodes = []
        for i in range(N):
            m_i = max(1, rng.poisson(lams[i])); mu_base = 1.0 + 1.5 * ((i + 1) / N)
            nodes.append({'m': m_i, 'pi1': 0.3, 'mu_base': mu_base, 'is_null': rng.random(m_i) >= 0.3})
        p, nl = sample_pvals_independent(nodes, rng)
        allp, m, R_bh, tau_bh = _bh(p, alpha)
        m_i = np.array([len(x) for x in p])
        f, t = fdr_power(p, nl, tau_bh); out[k, 0] = [t, f, 0, 0, 1]
        # ORACLE uses the realized tau_BH: a data-dependent benchmark, outside Thm 8
        dens = 0.7 + 0.3 * np.array([_gprime(nd['mu_base'], max(tau_bh, 1e-4)) for nd in nodes])
        for c, rule in enumerate(rules, start=1):
            M1 = allocate(m_i, total // 2, rule, density=dens); M2 = allocate(m_i, total - total // 2, rule, density=dens)
            th, tb, C = hetero_star_cert(p, alpha, M1, M2)
            f, t = fdr_power(p, nl, th)
            ex_c = m * C < 0.5
            out[k, c] = [t, f, float((th > 0) and tb <= (1 + eps) * th and not ex_c), float(ex_c),
                         float(int((allp <= th).sum()) == R_bh)]
    mu = out.mean(0); se = out.std(0, ddof=1) / np.sqrt(n_trials)
    d1 = out[:, 2, 0] - out[:, 1, 0]; d2 = out[:, 2, 0] - out[:, 3, 0]
    print(f'B={per_node}: ' + ' '.join(f'{n}={mu[c,0]:.3f}({se[c,0]:.3f}) fdr={mu[c,1]:.3f} cert={mu[c,2]+mu[c,3]:.2f}'
                                       for c, n in enumerate(['BH', 'equal', 'sqrt', 'oracle'])) +
          f' | sqrt-equal={d1.mean():+.3f}({d1.std(ddof=1)/np.sqrt(n_trials):.3f}) sqrt-oracle={d2.mean():+.4f}({d2.std(ddof=1)/np.sqrt(n_trials):.4f})',
          flush=True)
    save_npz(f'expS6_B{per_node}', out=out)


# ---------------------------------------------------------------- S7 (Table: rescaling vs refinement, per trial)
def exp_S7(degrees, n_trials=1000, alpha=0.2, N=50, lam=3, M_total=10, seed=42):
    from sim.review_experiments import qute_sf_global_grid
    from sim.algorithms import qute_sample_and_forward, adaptive_qute_sample_and_forward
    grid_all = (2, 4, 6, 8, 12, 16, 20)
    for deg in degrees:
        k = grid_all.index(deg)
        rng = np.random.default_rng(seed + 1000 + deg)
        nbrs = random_graph_neighbors(N, float(deg), seed + k)
        out = np.zeros((n_trials, 3, 2))
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, nl = sample_pvals_independent(nodes, rng)
            for c, taus in enumerate((qute_sf_global_grid(p, nbrs, alpha, M_total),
                                      qute_sample_and_forward(p, nbrs, alpha, M_total),
                                      adaptive_qute_sample_and_forward(p, nbrs, alpha, M_total // 2, M_total - M_total // 2))):
                f, pw, _ = _mh_eval(p, nl, taus); out[t, c] = [pw, f]
        mu = out.mean(0); d = out[:, 2, 0] - out[:, 1, 0]
        print(f'deg={deg}: global={mu[0,0]:.3f} local={mu[1,0]:.3f} twopass={mu[2,0]:.3f} '
              f'twopass-local={d.mean():+.4f}({d.std(ddof=1)/np.sqrt(n_trials):.4f}) maxfdr={mu[:,1].max():.3f}', flush=True)
        save_npz(f'expS7_deg{deg}', out=out)


if __name__ == '__main__':
    cmd, arg = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None
    {'S1': lambda: exp_S1(float(arg)), 'S2': lambda: exp_S2(int(arg)), 'S3': lambda: exp_S3(int(arg)),
     'S3u': exp_S3_uniform, 'S3n': exp_S3_norefute, 'S4': lambda: exp_S4(arg), 'S5': lambda: exp_S5(int(arg)),
     'S6': lambda: exp_S6(int(arg)),
     'S7': lambda: exp_S7((2, 4, 6, 8, 12, 16, 20) if arg is None else (int(arg),))}[cmd]()
