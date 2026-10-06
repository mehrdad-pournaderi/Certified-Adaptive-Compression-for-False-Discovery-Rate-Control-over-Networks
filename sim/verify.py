# sim/verify.py
"""Run before any experiment. Confirms FDR is controlled in the simplest setting."""
import numpy as np
from sim.algorithms import (
    bh_reject, sample_and_forward_star, adaptive_sample_and_forward_star,
)
from sim.data import generate_star_network, sample_pvals_independent

def fdr_power(pvals_list, is_null_list, tau):
    rej = [(p <= tau) for p in pvals_list]
    V = sum(int(((r) & nn).sum()) for r, nn in zip(rej, is_null_list))
    R = sum(int(r.sum()) for r in rej)
    nn_total = sum(int(nn.sum()) for nn in is_null_list)
    nonn_total = sum(int((~nn).sum()) for nn in is_null_list)
    fdp = V / max(R, 1)
    tdp = (R - V) / max(nonn_total, 1)
    return fdp, tdp

def main():
    alpha, N, lam, trials = 0.2, 100, 3, 2000
    rng = np.random.default_rng(0)
    fdr_orig, pow_orig = [], []
    fdr_adap, pow_adap = [], []
    for _ in range(trials):
        nodes = generate_star_network(N, lam, rng)
        p_list, null_list = sample_pvals_independent(nodes, rng)
        tau_o = sample_and_forward_star(p_list, alpha, M=6)
        tau_a = adaptive_sample_and_forward_star(p_list, alpha, M1=4, M2=4)
        f_o, t_o = fdr_power(p_list, null_list, tau_o)
        f_a, t_a = fdr_power(p_list, null_list, tau_a)
        fdr_orig.append(f_o); pow_orig.append(t_o)
        fdr_adap.append(f_a); pow_adap.append(t_a)
    print(f'orig:  FDR={np.mean(fdr_orig):.4f}  Power={np.mean(pow_orig):.4f}')
    print(f'adap:  FDR={np.mean(fdr_adap):.4f}  Power={np.mean(pow_adap):.4f}')
    assert np.mean(fdr_orig) <= alpha + 0.02
    assert np.mean(fdr_adap) <= alpha + 0.02
    assert np.mean(pow_adap) >= np.mean(pow_orig) - 0.01
    print('Verification passed.')

if __name__ == '__main__':
    main()
