"""Run all three weak-signal experiments and save results."""
import numpy as np
import time
import os

from sim.data import generate_star_network_weak, sample_pvals_independent, sample_pvals_dependent
from sim.experiments import run_methods_one_trial, save_npz

os.makedirs('sim/results', exist_ok=True)

def run_exp2_weak():
    n_trials=2000; alpha=0.2; N=100; M_total=10
    lam_grid=(1,2,3,4,5,6,7,8,9,10); seed=22
    rng = np.random.default_rng(seed)
    K = len(lam_grid)
    fdr = np.zeros((K, 4)); powr = np.zeros((K, 4))
    M1 = M_total // 2; M2 = M_total - M1
    t0 = time.time()
    for k, lam in enumerate(lam_grid):
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, n = sample_pvals_independent(nodes, rng)
            r = run_methods_one_trial(p, n, nodes, alpha, M=M_total, M1=M1, M2=M2, b=0, rho=0.0)
            fdr[k]  += [r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']]
            powr[k] += [r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']]
        fdr[k]  /= n_trials; powr[k] /= n_trials
        print(f'lam={lam}: FDR(adap)={fdr[k,3]:.3f}  Pow(adap)={powr[k,3]:.3f}  Pow(orig)={powr[k,2]:.3f}', flush=True)
    save_npz('exp2_weak', lam_grid=np.array(lam_grid), fdr=fdr, power=powr)
    print(f'exp2_weak done in {time.time()-t0:.1f}s', flush=True)

def run_exp3_weak():
    n_trials=2000; alpha=0.2; lam=3; M_total=10
    N_grid=(5,25,50,75,100,125,150,175,200); seed=23
    rng = np.random.default_rng(seed)
    K = len(N_grid)
    fdr = np.zeros((K, 4)); powr = np.zeros((K, 4))
    M1 = M_total // 2; M2 = M_total - M1
    t0 = time.time()
    for k, N in enumerate(N_grid):
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, n = sample_pvals_independent(nodes, rng)
            r = run_methods_one_trial(p, n, nodes, alpha, M=M_total, M1=M1, M2=M2, b=0, rho=0.0)
            fdr[k]  += [r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']]
            powr[k] += [r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']]
        fdr[k]  /= n_trials; powr[k] /= n_trials
        print(f'N={N}: FDR(adap)={fdr[k,3]:.3f}  Pow(adap)={powr[k,3]:.3f}  Pow(orig)={powr[k,2]:.3f}', flush=True)
    save_npz('exp3_weak', N_grid=np.array(N_grid), fdr=fdr, power=powr)
    print(f'exp3_weak done in {time.time()-t0:.1f}s', flush=True)

def run_exp4_weak():
    n_trials=500; alpha=0.2; N=100; lam=3; M_total=10
    rho_grid=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9); seed=24
    rng = np.random.default_rng(seed)
    K = len(rho_grid)
    fdr = np.zeros((K, 4)); powr = np.zeros((K, 4))
    M1 = M_total // 2; M2 = M_total - M1
    t0 = time.time()
    for k, rho in enumerate(rho_grid):
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, n = sample_pvals_dependent(nodes, rho, rng)
            r = run_methods_one_trial(p, n, nodes, alpha, M=M_total, M1=M1, M2=M2, b=0, rho=rho)
            fdr[k]  += [r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']]
            powr[k] += [r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']]
        fdr[k]  /= n_trials; powr[k] /= n_trials
        print(f'rho={rho}: FDR(adap)={fdr[k,3]:.3f}  Pow(adap)={powr[k,3]:.3f}  Pow(orig)={powr[k,2]:.3f}', flush=True)
    save_npz('exp4_weak', rho_grid=np.array(rho_grid), fdr=fdr, power=powr)
    print(f'exp4_weak done in {time.time()-t0:.1f}s', flush=True)

if __name__ == '__main__':
    import sys
    funcs = {'2w': run_exp2_weak, '3w': run_exp3_weak, '4w': run_exp4_weak}
    funcs[sys.argv[1]]()
