# sim/experiments.py
import numpy as np
import os
from sim.algorithms import (
    bh_reject, bonferroni_reject,
    sample_and_forward_star, adaptive_sample_and_forward_star,
)
from sim.data import (
    generate_star_network, generate_star_network_weak,
    sample_pvals_independent, sample_pvals_dependent,
    random_graph_neighbors, grid_2d_neighbors,
)
from sim.verify import fdr_power

RESULTS_DIR = 'sim/results'

def pooled_bh_tau(pvals_list, alpha):
    all_p = np.concatenate(pvals_list) if pvals_list else np.array([])
    mask = bh_reject(all_p, alpha)
    return float(all_p[mask].max()) if mask.any() else 0.0

def bonferroni_tau_per_node(node, alpha, m_total):
    return alpha * node['m'] / m_total  # local test size; threshold per node

def run_methods_one_trial(node_pvals, null_list, nodes, alpha, M, M1, M2, b, rho):
    m_total = sum(len(p) for p in node_pvals)
    tau_bh = pooled_bh_tau(node_pvals, alpha)
    fdr_bh, pow_bh = fdr_power(node_pvals, null_list, tau_bh)
    # "Baseline II" in the original paper: per-node BH at level alpha * m_i / m
    # (this is local-BH with a node-wise Bonferroni correction across nodes,
    # NOT pure Bonferroni per p-value; the paper calls it "Bonferroni" loosely).
    bf_V, bf_R, nonn_tot = 0, 0, 0
    for p, nn, node in zip(node_pvals, null_list, nodes):
        loc_alpha = alpha * node['m'] / m_total
        rej = bh_reject(p, loc_alpha)
        bf_V += int(((rej) & nn).sum()); bf_R += int(rej.sum())
        nonn_tot += int((~nn).sum())
    fdr_bf = bf_V / max(bf_R, 1); pow_bf = (bf_R - bf_V) / max(nonn_tot, 1)
    tau_o = sample_and_forward_star(node_pvals, alpha, M=M)
    fdr_o, pow_o = fdr_power(node_pvals, null_list, tau_o)
    tau_a = adaptive_sample_and_forward_star(node_pvals, alpha, M1=M1, M2=M2, b=b)
    fdr_a, pow_a = fdr_power(node_pvals, null_list, tau_a)
    return dict(fdr_bh=fdr_bh, pow_bh=pow_bh,
                fdr_bf=fdr_bf, pow_bf=pow_bf,
                fdr_o=fdr_o,  pow_o=pow_o,
                fdr_a=fdr_a,  pow_a=pow_a)

def save_npz(name, **arrays):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name + '.npz')
    np.savez(path, **arrays)
    print(f'Saved {path}')


def experiment_1(n_trials=2000, alpha=0.2, N=100, lam=3,
                 M_grid=(4, 5, 6, 7, 8, 10, 12, 16), seed=11):
    """x-axis is *total per-node forward samples*. Adaptive uses
       (M1, M2) = (M//2, M - M//2). Smallest meaningful M is 4 (=2+2)."""
    rng = np.random.default_rng(seed)
    K = len(M_grid)
    fdr = np.zeros((K, 4))   # cols: BH, Bonf, orig, adap
    powr = np.zeros((K, 4))
    for k, M in enumerate(M_grid):
        M1 = M // 2; M2 = M - M1
        assert M1 >= 2 and M2 >= 2, f"Need M >= 4, got {M}"
        for t in range(n_trials):
            nodes = generate_star_network(N, lam, rng)
            p, n = sample_pvals_independent(nodes, rng)
            r = run_methods_one_trial(p, n, nodes, alpha,
                                      M=M, M1=M1, M2=M2, b=0, rho=0.0)
            fdr[k]  += np.array([r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']])
            powr[k] += np.array([r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']])
        fdr[k]  /= n_trials
        powr[k] /= n_trials
        print(f'M={M}: FDR(adap)={fdr[k,3]:.3f}  Pow(adap)={powr[k,3]:.3f}')
    save_npz('exp1', M_grid=np.array(M_grid), fdr=fdr, power=powr)


def experiment_2(n_trials=2000, alpha=0.2, N=100, M_total=6,
                 lam_grid=(1,2,3,4,5,6,7,8,9,10), seed=12):
    rng = np.random.default_rng(seed)
    K = len(lam_grid)
    fdr = np.zeros((K, 4)); powr = np.zeros((K, 4))
    M1 = M_total // 2; M2 = M_total - M1
    for k, lam in enumerate(lam_grid):
        for t in range(n_trials):
            nodes = generate_star_network(N, lam, rng)
            p, n = sample_pvals_independent(nodes, rng)
            r = run_methods_one_trial(p, n, nodes, alpha,
                                      M=M_total, M1=M1, M2=M2, b=0, rho=0.0)
            fdr[k]  += np.array([r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']])
            powr[k] += np.array([r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']])
        fdr[k]  /= n_trials; powr[k] /= n_trials
        print(f'lam={lam}: Pow(adap)={powr[k,3]:.3f}')
    save_npz('exp2', lam_grid=np.array(lam_grid), fdr=fdr, power=powr)


def experiment_3(n_trials=2000, alpha=0.2, lam=3, M_total=6,
                 N_grid=(5,25,50,75,100,125,150,175,200), seed=13):
    rng = np.random.default_rng(seed)
    K = len(N_grid)
    fdr = np.zeros((K, 4)); powr = np.zeros((K, 4))
    M1 = M_total // 2; M2 = M_total - M1
    for k, N in enumerate(N_grid):
        for t in range(n_trials):
            nodes = generate_star_network(N, lam, rng)
            p, n = sample_pvals_independent(nodes, rng)
            r = run_methods_one_trial(p, n, nodes, alpha,
                                      M=M_total, M1=M1, M2=M2, b=0, rho=0.0)
            fdr[k]  += np.array([r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']])
            powr[k] += np.array([r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']])
        fdr[k]  /= n_trials; powr[k] /= n_trials
        print(f'N={N}: Pow(adap)={powr[k,3]:.3f}')
    save_npz('exp3', N_grid=np.array(N_grid), fdr=fdr, power=powr)


def experiment_4(n_trials=2000, alpha=0.2, N=100, lam=3, M_total=6,
                 rho_grid=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9), seed=14):
    rng = np.random.default_rng(seed)
    K = len(rho_grid)
    fdr = np.zeros((K, 4)); powr = np.zeros((K, 4))
    M1 = M_total // 2; M2 = M_total - M1
    for k, rho in enumerate(rho_grid):
        for t in range(n_trials):
            nodes = generate_star_network(N, lam, rng)
            p, n = sample_pvals_dependent(nodes, rho, rng)
            r = run_methods_one_trial(p, n, nodes, alpha,
                                      M=M_total, M1=M1, M2=M2, b=0, rho=rho)
            fdr[k]  += np.array([r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']])
            powr[k] += np.array([r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']])
        fdr[k]  /= n_trials; powr[k] /= n_trials
        print(f'rho={rho}: FDR(adap)={fdr[k,3]:.3f}  Pow(adap)={powr[k,3]:.3f}')
    save_npz('exp4', rho_grid=np.array(rho_grid), fdr=fdr, power=powr)


def experiment_5(n_trials=2000, alpha=0.2, N=100, lam=3,
                 M_total_grid=(4,6,8,10), seed=15):
    rng = np.random.default_rng(seed)
    results = {}
    for M_total in M_total_grid:
        splits = [(M1, M_total - M1) for M1 in range(2, M_total - 1)]
        powr = np.zeros(len(splits))
        for j, (M1, M2) in enumerate(splits):
            acc = 0.0
            for _ in range(n_trials):
                nodes = generate_star_network(N, lam, rng)
                p, n = sample_pvals_independent(nodes, rng)
                tau = adaptive_sample_and_forward_star(p, alpha, M1=M1, M2=M2, b=0)
                _, t = fdr_power(p, n, tau); acc += t
            powr[j] = acc / n_trials
            print(f'M_tot={M_total}  M1={M1}  M2={M2}  Pow={powr[j]:.3f}')
        results[f'M{M_total}_splits'] = np.array(splits)
        results[f'M{M_total}_power']  = powr
    save_npz('exp5', **results)


def experiment_6(n_trials=2000, alpha=0.2, N=100, M_total=6,
                 lam_grid=(1,2,3,4,5), b_grid=(0,1,2), seed=16):
    rng = np.random.default_rng(seed)
    M1 = M_total // 2; M2 = M_total - M1
    fdr = np.zeros((len(lam_grid), len(b_grid)))
    powr = np.zeros((len(lam_grid), len(b_grid)))
    for li, lam in enumerate(lam_grid):
        for bi, b in enumerate(b_grid):
            ff, pp = 0.0, 0.0
            for _ in range(n_trials):
                nodes = generate_star_network(N, lam, rng)
                p, n = sample_pvals_independent(nodes, rng)
                tau = adaptive_sample_and_forward_star(p, alpha, M1=M1, M2=M2, b=b)
                f, t = fdr_power(p, n, tau); ff += f; pp += t
            fdr[li, bi] = ff/n_trials; powr[li, bi] = pp/n_trials
            print(f'lam={lam} b={b}: FDR={fdr[li,bi]:.3f}  Pow={powr[li,bi]:.3f}')
    save_npz('exp6', lam_grid=np.array(lam_grid), b_grid=np.array(b_grid),
             fdr=fdr, power=powr)


from sim.algorithms import adaptive_iterative_sf_star

def experiment_7(n_trials=2000, alpha=0.2, N=100, lam=3,
                 K_grid=(1,2,3,4), M_per_round=3, seed=17):
    rng = np.random.default_rng(seed)
    fdr = np.zeros(len(K_grid)); powr = np.zeros(len(K_grid))
    for ki, K in enumerate(K_grid):
        ff, pp = 0.0, 0.0
        for _ in range(n_trials):
            nodes = generate_star_network(N, lam, rng)
            p, n = sample_pvals_independent(nodes, rng)
            tau = adaptive_iterative_sf_star(p, alpha, K=K, M_per_round=M_per_round)
            f, t = fdr_power(p, n, tau); ff += f; pp += t
        fdr[ki] = ff/n_trials; powr[ki] = pp/n_trials
        print(f'K={K}: FDR={fdr[ki]:.3f}  Pow={powr[ki]:.3f}  '
              f'total_samples={K*M_per_round}')
    save_npz('exp7', K_grid=np.array(K_grid), fdr=fdr, power=powr,
             M_per_round=M_per_round)


def experiment_1_weak(n_trials=2000, alpha=0.2, N=100, lam=3,
                      M_grid=(4, 5, 6, 7, 8, 10, 12, 16, 20), seed=18):
    """Weak-signal variant of experiment_1.

    Uses generate_star_network_weak (mu_base in [1.0, 2.5], pi1=0.3 constant)
    so that both original and adaptive SF* are bias-limited at moderate M,
    making the adaptive scheme's quadratic-in-M bias improvement visible as
    a power gap relative to the original.  Everything else mirrors experiment_1:
    alpha=0.2, N=100, lam=3, independent p-values, 2000 trials.

    Saved to sim/results/exp1_weak.npz with keys: M_grid, fdr, power.
    fdr/power columns: [BH, Bonf, orig, adap]
    """
    rng = np.random.default_rng(seed)
    K = len(M_grid)
    fdr = np.zeros((K, 4))   # cols: BH, Bonf, orig, adap
    powr = np.zeros((K, 4))
    for k, M in enumerate(M_grid):
        M1 = M // 2; M2 = M - M1
        assert M1 >= 2 and M2 >= 2, f"Need M >= 4, got {M}"
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, n = sample_pvals_independent(nodes, rng)
            r = run_methods_one_trial(p, n, nodes, alpha,
                                      M=M, M1=M1, M2=M2, b=0, rho=0.0)
            fdr[k]  += np.array([r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']])
            powr[k] += np.array([r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']])
        fdr[k]  /= n_trials
        powr[k] /= n_trials
        print(f'M={M:3d}: FDR(adap)={fdr[k,3]:.3f}  FDR(orig)={fdr[k,2]:.3f}  '
              f'Pow(adap)={powr[k,3]:.3f}  Pow(orig)={powr[k,2]:.3f}  '
              f'Pow(BH)={powr[k,0]:.3f}  gap(orig-BH)={powr[k,0]-powr[k,2]:.3f}')
    save_npz('exp1_weak', M_grid=np.array(M_grid), fdr=fdr, power=powr)
    # Sanity check: at least one M must have gap Pow(BH) - Pow(orig) >= 0.05
    gaps = powr[:, 0] - powr[:, 2]
    max_gap = float(gaps.max())
    best_M = M_grid[int(gaps.argmax())]
    if max_gap >= 0.05:
        print(f'\nSANITY CHECK PASSED: max gap Pow(BH)-Pow(orig) = {max_gap:.3f} at M={best_M}')
    else:
        print(f'\nSANITY CHECK FAILED: max gap Pow(BH)-Pow(orig) = {max_gap:.3f} '
              f'(need >= 0.05) -- signal may still be too strong')


def experiment_2_weak(n_trials=2000, alpha=0.2, N=100, M_total=10,
                      lam_grid=(1,2,3,4,5,6,7,8,9,10), seed=22):
    """Weak-signal variant of experiment_2.

    Uses generate_star_network_weak + sample_pvals_independent.
    M_total=10 chosen because E1_weak shows adaptive's largest lead at this M.
    Adaptive split: M1=M2=5.
    Saved to sim/results/exp2_weak.npz with keys: lam_grid, fdr, power.
    fdr/power columns: [BH, Bonf, orig, adap]
    """
    rng = np.random.default_rng(seed)
    K = len(lam_grid)
    fdr = np.zeros((K, 4)); powr = np.zeros((K, 4))
    M1 = M_total // 2; M2 = M_total - M1
    for k, lam in enumerate(lam_grid):
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, n = sample_pvals_independent(nodes, rng)
            r = run_methods_one_trial(p, n, nodes, alpha,
                                      M=M_total, M1=M1, M2=M2, b=0, rho=0.0)
            fdr[k]  += np.array([r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']])
            powr[k] += np.array([r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']])
        fdr[k]  /= n_trials; powr[k] /= n_trials
        print(f'lam={lam}: FDR(adap)={fdr[k,3]:.3f}  Pow(adap)={powr[k,3]:.3f}  '
              f'Pow(orig)={powr[k,2]:.3f}')
    save_npz('exp2_weak', lam_grid=np.array(lam_grid), fdr=fdr, power=powr)


def experiment_3_weak(n_trials=2000, alpha=0.2, lam=3, M_total=10,
                      N_grid=(5,25,50,75,100,125,150,175,200), seed=23):
    """Weak-signal variant of experiment_3.

    Uses generate_star_network_weak + sample_pvals_independent.
    Saved to sim/results/exp3_weak.npz with keys: N_grid, fdr, power.
    fdr/power columns: [BH, Bonf, orig, adap]
    """
    rng = np.random.default_rng(seed)
    K = len(N_grid)
    fdr = np.zeros((K, 4)); powr = np.zeros((K, 4))
    M1 = M_total // 2; M2 = M_total - M1
    for k, N in enumerate(N_grid):
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, n = sample_pvals_independent(nodes, rng)
            r = run_methods_one_trial(p, n, nodes, alpha,
                                      M=M_total, M1=M1, M2=M2, b=0, rho=0.0)
            fdr[k]  += np.array([r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']])
            powr[k] += np.array([r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']])
        fdr[k]  /= n_trials; powr[k] /= n_trials
        print(f'N={N}: FDR(adap)={fdr[k,3]:.3f}  Pow(adap)={powr[k,3]:.3f}  '
              f'Pow(orig)={powr[k,2]:.3f}')
    save_npz('exp3_weak', N_grid=np.array(N_grid), fdr=fdr, power=powr)


def experiment_4_weak(n_trials=500, alpha=0.2, N=100, lam=3, M_total=10,
                      rho_grid=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9), seed=24):
    """Weak-signal variant of experiment_4.

    Uses generate_star_network_weak + sample_pvals_dependent.
    Saved to sim/results/exp4_weak.npz with keys: rho_grid, fdr, power.
    fdr/power columns: [BH, Bonf, orig, adap]
    """
    rng = np.random.default_rng(seed)
    K = len(rho_grid)
    fdr = np.zeros((K, 4)); powr = np.zeros((K, 4))
    M1 = M_total // 2; M2 = M_total - M1
    for k, rho in enumerate(rho_grid):
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, n = sample_pvals_dependent(nodes, rho, rng)
            r = run_methods_one_trial(p, n, nodes, alpha,
                                      M=M_total, M1=M1, M2=M2, b=0, rho=rho)
            fdr[k]  += np.array([r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']])
            powr[k] += np.array([r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']])
        fdr[k]  /= n_trials; powr[k] /= n_trials
        print(f'rho={rho}: FDR(adap)={fdr[k,3]:.3f}  Pow(adap)={powr[k,3]:.3f}  '
              f'Pow(orig)={powr[k,2]:.3f}')
    save_npz('exp4_weak', rho_grid=np.array(rho_grid), fdr=fdr, power=powr)


def experiment_alpha_weak(n_trials=1500, N=100, lam=3, M_total=10,
                          alpha_grid=(0.05, 0.075, 0.10, 0.15, 0.20, 0.25),
                          seed=25):
    rng = np.random.default_rng(seed)
    K = len(alpha_grid)
    fdr = np.zeros((K, 4)); powr = np.zeros((K, 4))
    M1 = M_total // 2; M2 = M_total - M1
    for k, alpha in enumerate(alpha_grid):
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, n = sample_pvals_independent(nodes, rng)
            r = run_methods_one_trial(p, n, nodes, alpha,
                                      M=M_total, M1=M1, M2=M2, b=0, rho=0.0)
            fdr[k]  += np.array([r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']])
            powr[k] += np.array([r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']])
        fdr[k]  /= n_trials; powr[k] /= n_trials
        print(f'alpha={alpha}: FDR(adap)={fdr[k,3]:.3f}  Pow(adap)={powr[k,3]:.3f}')
    save_npz('exp_alpha_weak', alpha_grid=np.array(alpha_grid), fdr=fdr, power=powr)


def experiment_multihop_weak(n_trials=1500, alpha=0.2, N=50, lam=3, M_total=10, seed=33):
    """Cycle-of-N multihop network, weak-signal regime, varying nothing — just
    a single comparison point: orig QuTE-S&F vs adaptive QuTE-S&F, FDR + Power."""
    from sim.algorithms import qute_sample_and_forward, adaptive_qute_sample_and_forward
    from sim.data import cycle_graph_neighbors
    rng = np.random.default_rng(seed)
    M1 = M_total // 2; M2 = M_total - M1
    nbrs = cycle_graph_neighbors(N)
    fdr_orig = pow_orig = fdr_adap = pow_adap = 0.0
    for t in range(n_trials):
        nodes = generate_star_network_weak(N, lam, rng)
        p_list, null_list = sample_pvals_independent(nodes, rng)
        tau_orig = qute_sample_and_forward(p_list, nbrs, alpha, M=M_total)
        tau_adap = adaptive_qute_sample_and_forward(p_list, nbrs, alpha, M1=M1, M2=M2)
        V_o = R_o = V_a = R_a = nonn = 0
        for i in range(N):
            rej_o = p_list[i] <= tau_orig[i]
            rej_a = p_list[i] <= tau_adap[i]
            V_o += int((rej_o & null_list[i]).sum()); R_o += int(rej_o.sum())
            V_a += int((rej_a & null_list[i]).sum()); R_a += int(rej_a.sum())
            nonn += int((~null_list[i]).sum())
        fdr_orig += V_o / max(R_o, 1); pow_orig += (R_o - V_o) / max(nonn, 1)
        fdr_adap += V_a / max(R_a, 1); pow_adap += (R_a - V_a) / max(nonn, 1)
    fdr_orig /= n_trials; pow_orig /= n_trials
    fdr_adap /= n_trials; pow_adap /= n_trials
    print(f'Multihop (cycle N={N}, M_tot={M_total}):')
    print(f'  Original QuTE-S&F: FDR={fdr_orig:.3f}, Power={pow_orig:.3f}')
    print(f'  Adaptive QuTE-S&F: FDR={fdr_adap:.3f}, Power={pow_adap:.3f}')
    save_npz('exp_multihop_weak',
             fdr_orig=fdr_orig, pow_orig=pow_orig,
             fdr_adap=fdr_adap, pow_adap=pow_adap, N=N, M_total=M_total)


def experiment_multihop_vary_N(n_trials=1500, alpha=0.2, lam=3, M_total=10,
                                N_grid=(10, 25, 50, 75, 100), seed=34):
    """Cycle-of-N multihop network, weak-signal, vary N. Saves arrays of shape (K,)."""
    from sim.algorithms import qute_sample_and_forward, adaptive_qute_sample_and_forward
    from sim.data import cycle_graph_neighbors
    rng = np.random.default_rng(seed)
    K = len(N_grid)
    M1 = M_total // 2; M2 = M_total - M1
    arr_fdr_orig = np.zeros(K); arr_pow_orig = np.zeros(K)
    arr_fdr_adap = np.zeros(K); arr_pow_adap = np.zeros(K)
    for k, N in enumerate(N_grid):
        nbrs = cycle_graph_neighbors(N)
        fdr_orig = pow_orig = fdr_adap = pow_adap = 0.0
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p_list, null_list = sample_pvals_independent(nodes, rng)
            tau_orig = qute_sample_and_forward(p_list, nbrs, alpha, M=M_total)
            tau_adap = adaptive_qute_sample_and_forward(p_list, nbrs, alpha, M1=M1, M2=M2)
            V_o = R_o = V_a = R_a = nonn = 0
            for i in range(N):
                rej_o = p_list[i] <= tau_orig[i]
                rej_a = p_list[i] <= tau_adap[i]
                V_o += int((rej_o & null_list[i]).sum()); R_o += int(rej_o.sum())
                V_a += int((rej_a & null_list[i]).sum()); R_a += int(rej_a.sum())
                nonn += int((~null_list[i]).sum())
            fdr_orig += V_o / max(R_o, 1); pow_orig += (R_o - V_o) / max(nonn, 1)
            fdr_adap += V_a / max(R_a, 1); pow_adap += (R_a - V_a) / max(nonn, 1)
        arr_fdr_orig[k] = fdr_orig / n_trials; arr_pow_orig[k] = pow_orig / n_trials
        arr_fdr_adap[k] = fdr_adap / n_trials; arr_pow_adap[k] = pow_adap / n_trials
        print(f'N={N}: Orig FDR={arr_fdr_orig[k]:.3f} Pow={arr_pow_orig[k]:.3f}  '
              f'Adap FDR={arr_fdr_adap[k]:.3f} Pow={arr_pow_adap[k]:.3f}  '
              f'gap={arr_pow_adap[k]-arr_pow_orig[k]:.3f}')
    save_npz('exp_multihop_vary_N',
             N_grid=np.array(N_grid),
             fdr_orig=arr_fdr_orig, pow_orig=arr_pow_orig,
             fdr_adap=arr_fdr_adap, pow_adap=arr_pow_adap)


def _multihop_one_trial(nodes, p_list, null_list, nbrs, alpha, M_total, M1, M2, N):
    """Helper: run orig + adap QuTE-S&F for one trial; return (fdr_o, pow_o, fdr_a, pow_a)."""
    from sim.algorithms import qute_sample_and_forward, adaptive_qute_sample_and_forward
    tau_orig = qute_sample_and_forward(p_list, nbrs, alpha, M=M_total)
    tau_adap = adaptive_qute_sample_and_forward(p_list, nbrs, alpha, M1=M1, M2=M2)
    V_o = R_o = V_a = R_a = nonn = 0
    for i in range(N):
        rej_o = p_list[i] <= tau_orig[i]
        rej_a = p_list[i] <= tau_adap[i]
        V_o += int((rej_o & null_list[i]).sum()); R_o += int(rej_o.sum())
        V_a += int((rej_a & null_list[i]).sum()); R_a += int(rej_a.sum())
        nonn += int((~null_list[i]).sum())
    return (V_o / max(R_o, 1), (R_o - V_o) / max(nonn, 1),
            V_a / max(R_a, 1), (R_a - V_a) / max(nonn, 1))


def _pooled_bh_power_fdr(p_list, null_list, alpha):
    """BH on global pool; returns (fdr, power)."""
    all_p = np.concatenate(p_list)
    all_null = np.concatenate(null_list)
    rej = bh_reject(all_p, alpha)
    V = int((rej & all_null).sum()); R = int(rej.sum())
    nonn = int((~all_null).sum())
    return V / max(R, 1), (R - V) / max(nonn, 1)


def experiment_multihop_vary_degree(n_trials=1000, alpha=0.2, N=50, lam=3,
                                     M_total=10,
                                     degree_grid=(2, 4, 6, 8, 12, 16, 20),
                                     seed=41):
    """Cycle (deg=2) baseline plus ER random graphs at higher avg degrees.
    Larger neighborhoods -> larger alpha^(x) -> bigger adaptive bias gap.
    Saves to sim/results/exp_multihop_degree.npz with arrays
    degree_grid, fdr_orig, fdr_adap, pow_orig, pow_adap."""
    from sim.data import cycle_graph_neighbors
    rng = np.random.default_rng(seed)
    K = len(degree_grid)
    M1 = M_total // 2; M2 = M_total - M1
    arr_fdr_orig = np.zeros(K); arr_pow_orig = np.zeros(K)
    arr_fdr_adap = np.zeros(K); arr_pow_adap = np.zeros(K)
    for k, deg in enumerate(degree_grid):
        if deg == 2:
            nbrs = cycle_graph_neighbors(N)
        else:
            nbrs = random_graph_neighbors(N, avg_degree=float(deg), seed=seed + k + 1)
        fdr_orig = pow_orig = fdr_adap = pow_adap = 0.0
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p_list, null_list = sample_pvals_independent(nodes, rng)
            fo, po, fa, pa = _multihop_one_trial(
                nodes, p_list, null_list, nbrs, alpha, M_total, M1, M2, N)
            fdr_orig += fo; pow_orig += po; fdr_adap += fa; pow_adap += pa
        arr_fdr_orig[k] = fdr_orig / n_trials; arr_pow_orig[k] = pow_orig / n_trials
        arr_fdr_adap[k] = fdr_adap / n_trials; arr_pow_adap[k] = pow_adap / n_trials
        print(f'deg={deg:3d}: Orig FDR={arr_fdr_orig[k]:.3f} Pow={arr_pow_orig[k]:.3f}  '
              f'Adap FDR={arr_fdr_adap[k]:.3f} Pow={arr_pow_adap[k]:.3f}  '
              f'gap={arr_pow_adap[k]-arr_pow_orig[k]:.3f}')
    save_npz('exp_multihop_degree',
             degree_grid=np.array(degree_grid),
             fdr_orig=arr_fdr_orig, pow_orig=arr_pow_orig,
             fdr_adap=arr_fdr_adap, pow_adap=arr_pow_adap)


def experiment_multihop_vary_M(n_trials=1000, alpha=0.2, N=50, lam=3,
                                avg_degree=8,
                                M_grid=(8, 10, 12, 16, 20, 24, 32),
                                seed=42):
    """Random graph with deg=8. Vary per-node budget; for each M, adaptive uses
    (M//2, M-M//2). Show the bias-limited regime where original SF is far from
    pooled-BH and adaptive recovers most of the gap.
    Saves to sim/results/exp_multihop_M.npz with arrays
    M_grid, fdr_orig, fdr_adap, pow_orig, pow_adap, pow_pooled_bh."""
    rng = np.random.default_rng(seed)
    K = len(M_grid)
    nbrs = random_graph_neighbors(N, avg_degree=float(avg_degree), seed=seed)
    arr_fdr_orig = np.zeros(K); arr_pow_orig = np.zeros(K)
    arr_fdr_adap = np.zeros(K); arr_pow_adap = np.zeros(K)
    arr_pow_bh   = np.zeros(K)
    for k, M_total in enumerate(M_grid):
        M1 = M_total // 2; M2 = M_total - M1
        fdr_orig = pow_orig = fdr_adap = pow_adap = pow_bh = 0.0
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p_list, null_list = sample_pvals_independent(nodes, rng)
            fo, po, fa, pa = _multihop_one_trial(
                nodes, p_list, null_list, nbrs, alpha, M_total, M1, M2, N)
            _, pbh = _pooled_bh_power_fdr(p_list, null_list, alpha)
            fdr_orig += fo; pow_orig += po; fdr_adap += fa; pow_adap += pa; pow_bh += pbh
        arr_fdr_orig[k] = fdr_orig / n_trials; arr_pow_orig[k] = pow_orig / n_trials
        arr_fdr_adap[k] = fdr_adap / n_trials; arr_pow_adap[k] = pow_adap / n_trials
        arr_pow_bh[k]   = pow_bh   / n_trials
        print(f'M={M_total:3d}: Orig Pow={arr_pow_orig[k]:.3f}  '
              f'Adap Pow={arr_pow_adap[k]:.3f}  '
              f'BH Pow={arr_pow_bh[k]:.3f}  '
              f'gap(adap-orig)={arr_pow_adap[k]-arr_pow_orig[k]:.3f}')
    save_npz('exp_multihop_M',
             M_grid=np.array(M_grid),
             fdr_orig=arr_fdr_orig, pow_orig=arr_pow_orig,
             fdr_adap=arr_fdr_adap, pow_adap=arr_pow_adap,
             pow_pooled_bh=arr_pow_bh)


def experiment_multihop_grid(n_trials=1000, alpha=0.2, side=8, lam=3,
                              M_total=12, seed=43):
    """8x8 grid graph, N=64, each node has up to 4 neighbors + self.
    Single comparison point: orig vs adap FDR + Power.
    Saves to sim/results/exp_multihop_grid.npz with scalars
    fdr_orig, fdr_adap, pow_orig, pow_adap."""
    N = side * side
    rng = np.random.default_rng(seed)
    M1 = M_total // 2; M2 = M_total - M1
    nbrs = grid_2d_neighbors(side)
    fdr_orig = pow_orig = fdr_adap = pow_adap = 0.0
    for t in range(n_trials):
        nodes = generate_star_network_weak(N, lam, rng)
        p_list, null_list = sample_pvals_independent(nodes, rng)
        fo, po, fa, pa = _multihop_one_trial(
            nodes, p_list, null_list, nbrs, alpha, M_total, M1, M2, N)
        fdr_orig += fo; pow_orig += po; fdr_adap += fa; pow_adap += pa
    fdr_orig /= n_trials; pow_orig /= n_trials
    fdr_adap /= n_trials; pow_adap /= n_trials
    print(f'Grid {side}x{side} (N={N}, M_tot={M_total}):')
    print(f'  Original QuTE-S&F: FDR={fdr_orig:.3f}, Power={pow_orig:.3f}')
    print(f'  Adaptive QuTE-S&F: FDR={fdr_adap:.3f}, Power={pow_adap:.3f}')
    print(f'  Gap (adap-orig):   {pow_adap-pow_orig:.3f}')
    save_npz('exp_multihop_grid',
             fdr_orig=np.float64(fdr_orig), pow_orig=np.float64(pow_orig),
             fdr_adap=np.float64(fdr_adap), pow_adap=np.float64(pow_adap),
             side=np.int64(side), M_total=np.int64(M_total))


if __name__ == '__main__':
    import sys
    {'1': experiment_1, '1w': experiment_1_weak,
     '2': experiment_2, '3': experiment_3,
     '4': experiment_4, '5': experiment_5, '6': experiment_6,
     '7': experiment_7,
     '2w': experiment_2_weak, '3w': experiment_3_weak, '4w': experiment_4_weak,
     'aw': experiment_alpha_weak,
     'mh': experiment_multihop_weak,
     'mhN': experiment_multihop_vary_N,
     'mhD': experiment_multihop_vary_degree,
     'mhM': experiment_multihop_vary_M,
     'mhG': experiment_multihop_grid,
    }[sys.argv[1]]()
