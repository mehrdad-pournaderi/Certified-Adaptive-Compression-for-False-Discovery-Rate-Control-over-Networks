"""Randomised checks of the journal-extension claims.
  python3 sim/journal_tests.py cert   -> certificate validity (tau_hat<=tau_BH<=tau_bar, R_BH-R_hat<=mC)
  python3 sim/journal_tests.py mono   -> exact-arithmetic monotonicity of K-round and hetero thresholds
  python3 sim/journal_tests.py fdr    -> FDR of eps-stopped procedure vs (1+eps) alpha m0/m
"""
import sys
from fractions import Fraction as Fr
import numpy as np
from sim.algorithms import bh_reject
from sim.journal_algorithms import kround_star, hetero_star, kround_qute, qute_uncompressed
from sim.data import generate_star_network_weak, sample_pvals_independent, random_graph_neighbors


def test_certificate(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0; viol = 0
    for t in range(n):
        N = int(rng.integers(1, 6)); alpha = float(rng.choice([0.05, 0.1, 0.2, 0.3]))
        pv = [rng.random(int(rng.integers(1, 40))) ** float(rng.choice([1, 2, 4])) for _ in range(N)]
        K = int(rng.integers(1, 5)); M = [int(rng.integers(2, 7)) for _ in range(K)]
        eps = None if rng.random() < 0.5 else float(rng.choice([0.05, 0.2, 0.5]))
        res = kround_star(pv, alpha, M, eps=eps, refute=int(rng.integers(0, 3)), redirect=bool(rng.random() < 0.5),
                          uncertified=str(rng.choice(['zoom', 'largest', 'abstain'])))
        allp = np.concatenate(pv); m = len(allp)
        mask = bh_reject(allp, alpha)
        R_bh = int(mask.sum()); R_hat = int((allp <= res['tau']).sum())
        tau_bh = alpha * R_bh / m          # BH threshold functional sup{t: F(t) >= t/alpha}
        ok = (res['tau'] <= tau_bh + 1e-12) and (tau_bh <= res['tau_bar'] + 1e-12) and (R_bh - R_hat <= m * res['C'] + 1e-9)
        if not ok:
            viol += 1
            if viol <= 3: print('VIOLATION', res, tau_bh, R_bh, R_hat, m)
        worst = max(worst, (R_bh - R_hat) - m * res['C'])
    print(f'certificate: {viol} violations in {n} trials (max slack overshoot {worst:.3g})')
    # multihop
    viol = 0
    for t in range(300):
        N = int(rng.integers(3, 12)); nbrs = random_graph_neighbors(N, float(rng.integers(2, 5)), int(rng.integers(0, 10**6)))
        pv = [rng.random(int(rng.integers(1, 15))) ** 2 for _ in range(N)]
        res = kround_qute(pv, nbrs, 0.2, [int(rng.integers(2, 6)) for _ in range(int(rng.integers(1, 4)))],
                          eps=None if rng.random() < 0.5 else 0.2, refute=1, redirect=bool(rng.random() < 0.5))
        ref = qute_uncompressed(pv, nbrs, 0.2)
        for x in range(N):
            if not (res['tau'][x] <= ref[x] + 1e-12 <= res['tau_bar'][x] + 2e-12):
                viol += 1
            missed_true = int(((pv[x] > res['tau'][x]) & (pv[x] <= ref[x])).sum())
            if missed_true > res['missed'][x]:
                viol += 1
    print(f'multihop certificate: {viol} violations in 300 graphs')


# ---- exact arithmetic versions for monotonicity tests -------------------------
def _F_exact(pv, t):
    return Fr(sum(1 for p in pv if p <= t), len(pv))


def kround_exact(pv, alpha, M_list):
    pv = sorted(pv); L, U = Fr(0), alpha; tau = Fr(0)
    for M in M_list:
        Z = [L + Fr(k, M - 1) * (U - L) for k in range(M)]
        F = [_F_exact(pv, z) for z in Z]
        I = max(k for k in range(M) if F[k] >= Z[k] / alpha)
        tau = alpha * F[I]
        if I == M - 1: break
        L, U = Z[I], Z[I + 1]
    return tau


def hetero_exact(node_pv, alpha, M1s, M2s):
    m_i = [len(p) for p in node_pv]; m = sum(m_i)
    w = [Fr(mi, m) for mi in m_i]

    def floor_pooled(grids, t):
        s = Fr(0)
        for i, p in enumerate(node_pv):
            below = [z for z in grids[i] if z <= t]
            if below:
                s += w[i] * _F_exact(p, max(below))
        return s
    g1 = [[Fr(k, M - 1) * alpha for k in range(M)] for M in M1s]
    U1 = sorted(set(z for g in g1 for z in g))
    cross = [z for z in U1 if floor_pooled(g1, z) >= z / alpha]
    zj = max(cross); j = U1.index(zj)
    if j == len(U1) - 1:
        return alpha * floor_pooled(g1, zj)
    L, U = U1[j], U1[j + 1]
    g2 = [[L + Fr(k, M - 1) * (U - L) for k in range(M)] for M in M2s]
    U2 = sorted(set(z for g in g2 for z in g))
    cross2 = [z for z in U2 if floor_pooled(g2, z) >= z / alpha]
    if not cross2:
        return alpha * floor_pooled(g1, zj)
    zI = max(cross2)
    return alpha * floor_pooled(g2, zI)


def test_mono(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    alpha = Fr(1, 5); v1 = v2 = 0
    for t in range(n):
        m = int(rng.integers(2, 8))
        pv = [Fr(int(x), 10**6) for x in rng.integers(1, 10**6, size=m)]
        K = int(rng.integers(1, 5)); M = [int(rng.integers(2, 6)) for _ in range(K)]
        i = int(rng.integers(0, m)); pv2 = list(pv); pv2[i] = pv[i] * Fr(int(rng.integers(1, 1000)), 1000)
        if kround_exact(pv2, alpha, M) < kround_exact(pv, alpha, M):
            v1 += 1
            if v1 <= 2: print('KROUND VIOLATION', [float(p) for p in pv], i, float(pv2[i]), M)
        # hetero: split into nodes
        N = int(rng.integers(1, 4)); cuts = sorted(rng.choice(np.arange(1, m), size=min(N - 1, m - 1), replace=False)) if N > 1 and m > 1 else []
        idx = [0] + list(cuts) + [m]
        nodes = [pv[idx[a]:idx[a + 1]] for a in range(len(idx) - 1)]
        nodes2 = [pv2[idx[a]:idx[a + 1]] for a in range(len(idx) - 1)]
        nodes = [nd for nd in nodes if nd]; nodes2 = [nd for nd in nodes2 if nd]
        M1s = [int(rng.integers(2, 6)) for _ in nodes]; M2s = [int(rng.integers(2, 6)) for _ in nodes]
        if hetero_exact(nodes2, alpha, M1s, M2s) < hetero_exact(nodes, alpha, M1s, M2s):
            v2 += 1
            if v2 <= 2: print('HETERO VIOLATION', [[float(p) for p in nd] for nd in nodes], M1s, M2s)
    print(f'monotonicity: K-round violations {v1}, hetero violations {v2}, in {n} trials')


def test_fdr(n_trials=1500, seed=2):
    rng = np.random.default_rng(seed)
    alpha = 0.2
    for eps in [0.1, 0.3, 1.0]:
        f_stop = f_full = 0.0; rounds = 0.0; m0_over_m = 0.0
        for _ in range(n_trials):
            nodes = generate_star_network_weak(60, 3, rng)
            p, nl = sample_pvals_independent(nodes, rng)
            allp = np.concatenate(p); alln = np.concatenate(nl)
            rs = kround_star(p, alpha, [4, 4, 4, 4], eps=eps)
            rf = kround_star(p, alpha, [4, 4, 4, 4], eps=None)
            for r, acc in ((rs, 'stop'), (rf, 'full')):
                rej = allp <= r['tau']
                fdp = (rej & alln).sum() / max(rej.sum(), 1)
                if acc == 'stop': f_stop += fdp
                else: f_full += fdp
            rounds += rs['rounds']; m0_over_m += alln.mean()
        print(f'eps={eps}: FDR(stopped)={f_stop/n_trials:.4f}  FDR(full K)={f_full/n_trials:.4f}  '
              f'bound (1+eps)*alpha*m0/m={(1+eps)*alpha*m0_over_m/n_trials:.4f}  alpha*m0/m={alpha*m0_over_m/n_trials:.4f}  '
              f'mean rounds={rounds/n_trials:.2f}')


if __name__ == '__main__' and sys.argv[1] != 'ident':
    {'cert': test_certificate, 'mono': test_mono, 'fdr': test_fdr}[sys.argv[1]]()


def test_identity_and_hetero(n=3000, seed=11):
    """(1) identity mC = (m/alpha)(tau_bar - tau_hat) for the star certificate, all variants;
       (2) validity of the heterogeneous-grid certificate: tau_hat <= tau_BH <= tau_bar,
           R_BH - R_hat <= mC, and the same identity; empty nodes allowed."""
    from sim.journal_algorithms import hetero_star_cert
    rng = np.random.default_rng(seed)
    bad_id = bad_h = 0; worst = 0.0
    for t in range(n):
        alpha = float(rng.choice([0.05, 0.1, 0.2, 0.3]))
        N = int(rng.integers(1, 6))
        pv = [rng.random(int(rng.integers(0, 30))) ** float(rng.choice([1, 2, 4])) for _ in range(N)]
        if sum(len(x) for x in pv) == 0:
            pv[0] = rng.random(3)
        allp = np.concatenate(pv); m = len(allp)
        R_bh = int(bh_reject(allp, alpha).sum()); tau_bh = alpha * R_bh / m
        res = kround_star(pv, alpha, [int(rng.integers(2, 7)) for _ in range(int(rng.integers(1, 5)))],
                          eps=None if rng.random() < 0.5 else 0.2, refute=int(rng.integers(0, 3)),
                          redirect=bool(rng.random() < 0.5), uncertified=str(rng.choice(['zoom', 'largest', 'abstain'])))
        d = abs(m * res['C'] - (m / alpha) * (res['tau_bar'] - res['tau']))
        worst = max(worst, d); bad_id += d > 1e-7
        M1 = [int(rng.integers(2, 8)) for _ in pv]; M2 = [int(rng.integers(2, 8)) for _ in pv]
        th, tb, C = hetero_star_cert(pv, alpha, M1, M2)
        R_hat = int((allp <= th).sum())
        ok = (th <= tau_bh + 1e-12) and (tau_bh <= tb + 1e-12) and (R_bh - R_hat <= m * C + 1e-9) \
            and abs(m * C - (m / alpha) * (tb - th)) < 1e-7
        bad_h += not ok
    print(f'identity (star, all variants): {bad_id} failures in {n} (max |diff| {worst:.2e}); '
          f'heterogeneous certificate: {bad_h} failures in {n}')


if __name__ == '__main__' and sys.argv[1] == 'ident':
    test_identity_and_hetero()
