"""Algorithms for the journal extension.

  kround_star        K-round adaptive sample-and-forward (star) with the
                     missed-discovery certificate and optional eps-stopping.
  kround_qute        Multihop (QuTE-coupled) version with per-node certificates.
  hetero_star        Heterogeneous per-node budgets (union-grid crossing, capped
                     threshold) + allocation rules.
  certificate        Ceiling-envelope certificate from any set of known points.

All thresholds are of the form alpha * F(Z) for a sampled grid point Z, so
every transmitted quantity is a count (O(log m) bits).
"""
import numpy as np
from sim.algorithms import _empirical_cdf_samples, bh_reject


# ----------------------------------------------------------------------------
# Certificate from known points
# ----------------------------------------------------------------------------
def certificate(known_z, known_F, alpha, tau_hat, F_at_tau_hat_point):
    """Given sorted known points z_0=0<z_1<...<z_B with exact pooled CDF values
    F(z_b), return (tau_bar, C) where
        tau_bar = sup{t : Fbar(t) >= t/alpha},  Fbar(t) = F(min{z >= t})
    is a computable upper bound on the pooled-BH threshold, and
        C = Fbar(tau_bar) - F(Z_I)
    bounds (R_BH - R_hat)/m.  `F_at_tau_hat_point` is F(Z_I) = tau_hat/alpha."""
    z = np.asarray(known_z); Fv = np.asarray(known_F)
    tau_bar = 0.0
    for b in range(1, len(z)):
        cand = min(z[b], alpha * Fv[b])
        if alpha * Fv[b] > z[b - 1]:    # STRICT: the left-open bin (z_{b-1}, z_b] admits a crossing
            tau_bar = max(tau_bar, cand)
    tau_bar = max(tau_bar, tau_hat)
    # Fbar(tau_bar) = F at the smallest known point >= tau_bar
    idx = int(np.searchsorted(z, tau_bar, side='left'))
    idx = min(idx, len(z) - 1)
    Fbar = Fv[idx]
    C = max(0.0, float(Fbar - F_at_tau_hat_point))
    return float(tau_bar), C


def _pooled_F(node_pvals, w, locations):
    F = np.zeros(len(locations))
    for i, p in enumerate(node_pvals):
        if w[i] > 0:
            F += w[i] * _empirical_cdf_samples(p, locations)
    return F


# ----------------------------------------------------------------------------
# K-round star
# ----------------------------------------------------------------------------
def admitting_bins(known_z, known_F, alpha, lo, hi):
    """Bins (z_{b-1}, z_b] of the known-point set, lying outside [lo, hi], that
    still admit a BH crossing under the ceiling envelope: alpha F(z_b) > z_{b-1}."""
    out = []
    for b in range(1, len(known_z)):
        a, c = known_z[b - 1], known_z[b]
        if c <= lo or a >= hi:
            if alpha * known_F[b] > a and c > a:
                out.append((a, c))
    return out


def kround_star(node_pvals, alpha, M_list, eps=None, weights=None, refute=0,
                first_grid=None, redirect=False, uncertified='zoom'):
    """Certified zoom on a star network.

    M_list   : grid sizes per round (len K_max).  Round 1 costs M_1 counts; a
               later round costs M_r - 2 NEW counts, because both window
               endpoints were sampled in an earlier round and are reused.
    eps      : None -> fixed-K mode (run all rounds); else eps-certified stopping:
               stop at the first round with tau_bar <= (1+eps) * tau_out.
    refute   : per round, one extra count at the midpoint of each of the
               `refute` widest admitting bins outside the window.
    redirect : if True and a refutation count reveals a crossing above the
               window, restart the zoom from the known bin containing it; the
               output threshold is then alpha F(z*) at the largest known
               crossing z*.  If False, the output is the monotone zoom-chain
               threshold and refutation counts only inform the certificate.
    uncertified : what to output if eps-mode ends at K_max uncertified:
               'zoom' (monotone chain), 'largest' (largest known crossing) or
               'abstain' (0).
    Returns dict with tau (output), tau_zoom, tau_largest, tau_bar, C,
    certified (bool), redirected (bool), rounds, samples (new counts incl.
    refutation), exchanges (sequential request-response exchanges: one per
    zoom request, plus one per refutation request, since refutation
    locations are chosen after the zoom counts arrive),
    trace [(tau_out, tau_bar, C, samples_so_far)].
    """
    m_i = np.array([len(p) for p in node_pvals], dtype=float)
    m_total = m_i.sum()
    empty = dict(tau=0.0, tau_zoom=0.0, tau_largest=0.0, tau_bar=0.0, C=0.0, certified=True,   # nothing to miss
                 redirected=False, rounds=0, samples=0, exchanges=0, trace=[])
    if m_total == 0:
        return empty
    w = m_i / m_total if weights is None else np.asarray(weights)
    known = {0.0: float(_pooled_F(node_pvals, w, np.array([0.0]))[0])}
    L, U = 0.0, alpha
    tau_zoom = 0.0; tau_out = 0.0; z_out = 0.0
    trace = []; samples = 1; certified = False; redirected = False   # the count at t=0 is transmitted
    exchanges = 0
    tau_bar = 0.0; C = 0.0

    def largest_crossing():
        zs = [z for z, f in known.items() if f >= z / alpha]
        z = max(zs); return z, known[z]

    for r, M in enumerate(M_list):
        if r == 0 and first_grid is not None:
            Z = np.asarray(first_grid, dtype=float); M = len(Z)
        else:
            Z = np.linspace(L, U, M)
        new_pts = [float(z) for z in Z if float(z) not in known]
        if new_pts:
            exchanges += 1
            Fn = _pooled_F(node_pvals, w, np.array(new_pts)); samples += len(new_pts)
            for zz, ff in zip(new_pts, Fn):
                known[zz] = float(ff)
        F = np.array([known[float(z)] for z in Z])
        crossing = F >= Z / alpha
        I = int(np.max(np.where(crossing)[0]))
        tau_zoom = max(tau_zoom, float(alpha * F[I]))
        top = (I == M - 1)
        if not top:
            L, U = float(Z[I]), float(Z[I + 1])
        # refutation
        if refute > 0 and not top:
            kz = np.array(sorted(known)); kF = np.array([known[z] for z in kz])
            bins = admitting_bins(kz, kF, alpha, L, U)
            bins.sort(key=lambda ab: -(ab[1] - ab[0]))
            mids = [0.5 * (a + c) for a, c in bins[:refute]]
            mids = [z for z in mids if z not in known]
            if mids:
                exchanges += 1
                Fm = _pooled_F(node_pvals, w, np.array(mids)); samples += len(mids)
                for zz, ff in zip(mids, Fm):
                    known[zz] = float(ff)
        # output threshold and (optional) redirection
        z_star, F_star = largest_crossing()
        if redirect and z_star >= alpha - 1e-15:
            # largest known crossing at the top: tau_hat = tau_bar = alpha, C = 0 (exactly certified)
            tau_out = float(alpha * F_star); tau_bar, C = float(alpha), 0.0
            trace.append((tau_out, tau_bar, C, samples)); certified = eps is not None
            break
        if redirect:
            if z_star > U + 1e-15 and not top:
                kz = np.array(sorted(known)); above = kz[kz > z_star]
                L, U = float(z_star), float(above[0]) if len(above) else float(alpha)
                redirected = True
            tau_out, z_out = float(alpha * F_star), z_star
        else:
            tau_out, z_out = tau_zoom, None
        F_out = tau_out / alpha
        kz = np.array(sorted(known)); kF = np.array([known[z] for z in kz])
        tau_bar, C = certificate(kz, kF, alpha, tau_out, F_out)
        trace.append((tau_out, tau_bar, C, samples))
        # certification: relative (tau_bar <= (1+eps) tau_out) or exact (m*C < 1 => same rejections as BH)
        if eps is not None:
            certified = (tau_out > 0 and tau_bar <= (1 + eps) * tau_out) or (m_total * C < 0.5)   # mC is integer-valued exactly; < 0.5 means 0
        if top or (eps is not None and certified):
            break
    if eps is not None and not certified:
        if uncertified == 'abstain':
            tau_out = 0.0
        elif uncertified == 'zoom':
            tau_out = tau_zoom
        # 'largest': keep tau_out
        kz = np.array(sorted(known)); kF = np.array([known[z] for z in kz])
        tau_bar, C = certificate(kz, kF, alpha, tau_out, tau_out / alpha)
    return dict(tau=tau_out, tau_zoom=tau_zoom, tau_largest=float(alpha * largest_crossing()[1]),
                tau_bar=tau_bar, C=C, certified=certified, redirected=redirected,
                rounds=len(trace), samples=samples, exchanges=exchanges, trace=trace)


# ----------------------------------------------------------------------------
# K-round multihop (QuTE coupling)
# ----------------------------------------------------------------------------
def kround_qute(node_pvals, neighbors, alpha, M_list, eps=None, refute=0, redirect=False,
                uncertified='zoom', network_abstain=False):
    """Each node x runs kround_star on its closed neighbourhood with local test
    size alpha_x; exchange takes the max of thresholds AND of certificate upper
    bounds.  If network_abstain, every node rejects nothing whenever some local
    run is uncertified (requires flooding one bit over the graph).
    Returns per-node arrays tau (final), tau_bar (upper bound on the
    uncompressed-QuTE threshold at x), rounds, samples, exchanges, certified,
    and the exact per-node count of possibly-missed discoveries
    #{j at x: tau_x < P_j <= tau_bar_x}; all_certified (bool)."""
    N = len(node_pvals)
    m_i = np.array([len(p) for p in node_pvals], dtype=float)
    m_total = m_i.sum()
    if m_total == 0:   # empty network: no rejections
        z = np.zeros(N)
        return dict(tau=z, tau_bar=z, rounds=np.zeros(N, int), missed=np.zeros(N, int), tau0=z,
                    samples=np.zeros(N, int), exchanges=np.zeros(N, int), certified=np.ones(N, bool),
                    all_certified=True)
    tau0 = np.zeros(N); bar0 = np.zeros(N); rounds = np.zeros(N, dtype=int)
    samples = np.zeros(N, dtype=int); exch = np.zeros(N, dtype=int); cert = np.zeros(N, dtype=bool)
    for x in range(N):
        nb = neighbors[x]
        n_x = sum(m_i[i] for i in nb)
        alpha_x = (n_x / m_total) * alpha
        res = kround_star([node_pvals[i] for i in nb], alpha_x, M_list, eps=eps, refute=refute,
                          redirect=redirect, uncertified=uncertified)
        tau0[x], bar0[x], rounds[x], samples[x], exch[x], cert[x] = (
            res['tau'], res['tau_bar'], res['rounds'], res['samples'], res['exchanges'], res['certified'])
    all_cert = bool(cert.all())
    if network_abstain and not all_cert:
        tau0 = np.zeros(N)
    tau = np.array([max(tau0[i] for i in neighbors[x]) for x in range(N)])
    tau_bar = np.array([max(bar0[i] for i in neighbors[x]) for x in range(N)])
    missed = np.array([int(((node_pvals[x] > tau[x]) & (node_pvals[x] <= tau_bar[x])).sum())
                       for x in range(N)])
    return dict(tau=tau, tau_bar=tau_bar, rounds=rounds, missed=missed, tau0=tau0, samples=samples,
                exchanges=exch, certified=cert, all_certified=all_cert)


def qute_uncompressed(node_pvals, neighbors, alpha):
    """Reference: QuTE with full p-values (local BH at level alpha_x, then max)."""
    N = len(node_pvals)
    m_i = np.array([len(p) for p in node_pvals], dtype=float)
    m_total = m_i.sum()
    tau0 = np.zeros(N)
    for x in range(N):
        nb = neighbors[x]
        pool = np.concatenate([node_pvals[i] for i in nb])
        alpha_x = (len(pool) / m_total) * alpha
        mask = bh_reject(pool, alpha_x)
        tau0[x] = float(alpha_x * mask.sum() / len(pool))   # threshold functional; same rejections
    return np.array([max(tau0[i] for i in neighbors[x]) for x in range(N)])


# ----------------------------------------------------------------------------
# Heterogeneous budgets (star)
# ----------------------------------------------------------------------------
def _floor_pooled(node_pvals, w, grids, ts):
    """Pooled floor-interpolated CDF at points ts, node i sampled on grids[i]."""
    out = np.zeros(len(ts))
    for i, p in enumerate(node_pvals):
        Z = grids[i]
        Fz = _empirical_cdf_samples(p, Z)
        idx = np.searchsorted(Z, ts, side='right') - 1
        vals = np.where(idx >= 0, Fz[np.clip(idx, 0, len(Z) - 1)], 0.0)
        out += w[i] * vals
    return out


def hetero_star(node_pvals, alpha, M1_list, M2_list):
    """Two-pass adaptive S&F with heterogeneous per-node budgets.
    Round 1: node i samples M1_i equispaced points on [0, alpha].  The centre
    forms the pooled floor-interpolated CDF and finds the largest crossing on
    the UNION of the round-1 grids; the window is the union-grid bin [L, U].
    Round 2: node i samples M2_i equispaced points on [L, U]; the centre finds
    the largest crossing on the union of round-2 grids and sets
        tau_hat = alpha * Fhat2(z_I).
    Monotone because L is sampled exactly by every node in round 2, so
    tau_hat >= alpha F(L) (see paper, Appendix A)."""
    m_i = np.array([len(p) for p in node_pvals], dtype=float)
    m_total = m_i.sum()
    w = m_i / m_total
    g1 = [np.linspace(0, alpha, int(M)) for M in M1_list]
    U1 = np.unique(np.concatenate(g1))
    F1 = _floor_pooled(node_pvals, w, g1, U1)
    c1 = F1 >= U1 / alpha
    j = int(np.max(np.where(c1)[0]))
    if j == len(U1) - 1:
        return float(alpha * F1[j])
    L, U = float(U1[j]), float(U1[j + 1])
    g2 = [np.linspace(L, U, int(M)) for M in M2_list]
    U2 = np.unique(np.concatenate(g2))
    F2 = _floor_pooled(node_pvals, w, g2, U2)
    c2 = F2 >= U2 / alpha
    if not c2.any():
        return float(alpha * F1[j])
    I = int(np.max(np.where(c2)[0]))
    return float(alpha * F2[I])


def hetero_certificate(node_pvals, alpha, grids, tau_hat=None):
    """Certificate for heterogeneous per-node grids.  grids[i] = sorted array of
    ALL points node i has sampled (both rounds; each contains 0 and alpha).
    Pooled floor  Fhat(t) = sum_i w_i F_i(z_i(t)),   z_i(t)  = max{z in grid_i : z <= t}
    pooled ceiling Fbar(t) = sum_i w_i F_i(z_i^+(t)), z_i^+(t) = min{z in grid_i : z >= t}
    (empty nodes contribute nothing).  Both are step functions constant on the
    bins of the union grid U.  Returns (tau_hat, tau_bar, C) with
    tau_hat the procedure's threshold (any self-consistent threshold, i.e.
    R(tau_hat) >= m tau_hat / alpha; default alpha * Fhat(z_I) at the largest
    union-grid crossing of Fhat), tau_bar = max(tau_hat, sup{t <= alpha :
    Fbar(t) >= t/alpha}) and C = Fbar(tau_bar) - tau_hat/alpha."""
    m_i = np.array([len(p) for p in node_pvals], dtype=float)
    m_total = m_i.sum()
    if m_total == 0:
        return 0.0, 0.0, 0.0
    w = m_i / m_total
    U = np.unique(np.concatenate([np.asarray(g, float) for g in grids]))
    lo = np.zeros(len(U)); hi = np.zeros(len(U))
    for i, p in enumerate(node_pvals):
        if m_i[i] == 0:
            continue
        g = np.asarray(grids[i], float)
        Fg = _empirical_cdf_samples(p, g)
        il = np.searchsorted(g, U, side='right') - 1            # largest grid pt <= u
        ih = np.minimum(np.searchsorted(g, U, side='left'), len(g) - 1)  # smallest grid pt >= u
        lo += w[i] * np.where(il >= 0, Fg[np.clip(il, 0, None)], 0.0)
        hi += w[i] * Fg[ih]
    if tau_hat is None:
        cross = lo >= U / alpha
        I = int(np.max(np.where(cross)[0]))
        tau_hat = float(alpha * lo[I])
    tau_bar = tau_hat
    for b in range(1, len(U)):
        if alpha * hi[b] > U[b - 1]:
            tau_bar = max(tau_bar, min(U[b], alpha * hi[b]))
    idx = min(int(np.searchsorted(U, tau_bar, side='left')), len(U) - 1)
    C = max(0.0, float(hi[idx] - tau_hat / alpha))
    return float(tau_hat), float(tau_bar), C


def hetero_star_cert(node_pvals, alpha, M1_list, M2_list):
    """hetero_star's output tau_hat together with its certificate (tau_bar, C),
    computed from every point each node sampled in either round."""
    m_i = np.array([len(p) for p in node_pvals], dtype=float)
    if m_i.sum() == 0:
        return 0.0, 0.0, 0.0
    w = m_i / m_i.sum()
    g1 = [np.linspace(0, alpha, int(M)) for M in M1_list]
    U1 = np.unique(np.concatenate(g1))
    F1 = _floor_pooled(node_pvals, w, g1, U1)
    j = int(np.max(np.where(F1 >= U1 / alpha)[0]))
    if j == len(U1) - 1:
        grids = g1
    else:
        L, U = float(U1[j]), float(U1[j + 1])
        grids = [np.unique(np.concatenate([g1[i], np.linspace(L, U, int(M2_list[i]))])) for i in range(len(g1))]
    return hetero_certificate(node_pvals, alpha, grids, tau_hat=hetero_star(node_pvals, alpha, M1_list, M2_list))


def allocate(m_i, total, rule='equal', density=None, min_per_node=3):
    """Allocate a total number of samples across nodes for one pass.
       'equal' : same for all;  'sqrt' : M_i - 1 proportional to sqrt(m_i);
       'oracle': M_i - 1 proportional to sqrt(m_i * density_i)."""
    N = len(m_i)
    if rule == 'equal':
        base = np.full(N, total // N)
    else:
        a = np.sqrt(np.asarray(m_i, dtype=float))
        if rule == 'oracle':
            a = np.sqrt(np.asarray(m_i, dtype=float) * np.asarray(density))
        share = a / a.sum() * (total - N * min_per_node)
        base = min_per_node + np.floor(share).astype(int)
    # distribute remainder
    rem = int(total - base.sum())
    order = np.argsort(-np.asarray(m_i))
    for k in range(rem):
        base[order[k % N]] += 1
    return np.maximum(base, 2)
