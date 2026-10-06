"""Run a chunk of an experiment. Called as:
  python3 run_chunk.py <exp> <k_start> <k_end>
Where k_start..k_end-1 are the parameter indices to compute.
Saves partial results to sim/results/<exp>_chunk_<k_start>.npz
"""
import sys, numpy as np, os
from sim.data import generate_star_network_weak, sample_pvals_independent, sample_pvals_dependent
from sim.experiments import run_methods_one_trial

os.makedirs('sim/results', exist_ok=True)

exp = sys.argv[1]
k_start = int(sys.argv[2])
k_end   = int(sys.argv[3])

if exp == '2w':
    n_trials=2000; alpha=0.2; N=100; M_total=10
    lam_grid=(1,2,3,4,5,6,7,8,9,10); seed=22
    param_grid = list(lam_grid)
    param_name = 'lam_grid'
    M1 = M_total // 2; M2 = M_total - M1

    rng = np.random.default_rng(seed + k_start * 1000)
    chunk_size = k_end - k_start
    fdr = np.zeros((chunk_size, 4)); powr = np.zeros((chunk_size, 4))
    for ci, k in enumerate(range(k_start, k_end)):
        lam = param_grid[k]
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, n = sample_pvals_independent(nodes, rng)
            r = run_methods_one_trial(p, n, nodes, alpha, M=M_total, M1=M1, M2=M2, b=0, rho=0.0)
            fdr[ci]  += [r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']]
            powr[ci] += [r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']]
        fdr[ci]  /= n_trials; powr[ci] /= n_trials
        print(f'lam={lam}: FDR(adap)={fdr[ci,3]:.3f}  Pow(adap)={powr[ci,3]:.3f}  Pow(orig)={powr[ci,2]:.3f}', flush=True)
    out = f'sim/results/exp2_weak_chunk_{k_start}.npz'
    np.savez(out, indices=np.arange(k_start, k_end), fdr=fdr, power=powr,
             param_vals=np.array(param_grid[k_start:k_end]))
    print(f'Saved {out}')

elif exp == '3w':
    n_trials=2000; alpha=0.2; lam=3; M_total=10
    N_grid=(5,25,50,75,100,125,150,175,200); seed=23
    param_grid = list(N_grid)
    M1 = M_total // 2; M2 = M_total - M1

    rng = np.random.default_rng(seed + k_start * 1000)
    chunk_size = k_end - k_start
    fdr = np.zeros((chunk_size, 4)); powr = np.zeros((chunk_size, 4))
    for ci, k in enumerate(range(k_start, k_end)):
        N = param_grid[k]
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, n = sample_pvals_independent(nodes, rng)
            r = run_methods_one_trial(p, n, nodes, 0.2, M=M_total, M1=M1, M2=M2, b=0, rho=0.0)
            fdr[ci]  += [r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']]
            powr[ci] += [r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']]
        fdr[ci]  /= n_trials; powr[ci] /= n_trials
        print(f'N={N}: FDR(adap)={fdr[ci,3]:.3f}  Pow(adap)={powr[ci,3]:.3f}  Pow(orig)={powr[ci,2]:.3f}', flush=True)
    out = f'sim/results/exp3_weak_chunk_{k_start}.npz'
    np.savez(out, indices=np.arange(k_start, k_end), fdr=fdr, power=powr,
             param_vals=np.array(param_grid[k_start:k_end]))
    print(f'Saved {out}')

elif exp == '4w':
    n_trials=500; alpha=0.2; N=100; lam=3; M_total=10
    rho_grid=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9); seed=24
    param_grid = list(rho_grid)
    M1 = M_total // 2; M2 = M_total - M1

    rng = np.random.default_rng(seed + k_start * 1000)
    chunk_size = k_end - k_start
    fdr = np.zeros((chunk_size, 4)); powr = np.zeros((chunk_size, 4))
    for ci, k in enumerate(range(k_start, k_end)):
        rho = param_grid[k]
        for t in range(n_trials):
            nodes = generate_star_network_weak(N, lam, rng)
            p, n = sample_pvals_dependent(nodes, rho, rng)
            r = run_methods_one_trial(p, n, nodes, alpha, M=M_total, M1=M1, M2=M2, b=0, rho=rho)
            fdr[ci]  += [r['fdr_bh'], r['fdr_bf'], r['fdr_o'], r['fdr_a']]
            powr[ci] += [r['pow_bh'], r['pow_bf'], r['pow_o'], r['pow_a']]
        fdr[ci]  /= n_trials; powr[ci] /= n_trials
        print(f'rho={rho}: FDR(adap)={fdr[ci,3]:.3f}  Pow(adap)={powr[ci,3]:.3f}  Pow(orig)={powr[ci,2]:.3f}', flush=True)
    out = f'sim/results/exp4_weak_chunk_{k_start}.npz'
    np.savez(out, indices=np.arange(k_start, k_end), fdr=fdr, power=powr,
             param_vals=np.array(param_grid[k_start:k_end]))
    print(f'Saved {out}')
