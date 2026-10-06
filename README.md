# Certified Adaptive Compression for False Discovery Rate Control over Networks

Simulation code and saved results for

> M. Pournaderi, *Certified Adaptive Compression for False Discovery Rate Control over Networks*, 2026.

The paper studies pooled Benjamini–Hochberg (BH) over a network when each node
can only forward a few counts of its p-values. It gives

* a **certificate** (an interval for the BH threshold, equivalently a bound on
  the number of missed BH rejections) computable from the counts the center
  already holds, for any grid and any joint law of the p-values;
* a **certified K-round zoom** that refines the grid where the BH threshold
  lies and stops when the certificate says so, with FDR control at level
  `(1+eps) alpha m0/m` (Theorem 2);
* extensions to **multihop** networks (QuTE-style exchanges), **sparse**
  regimes with geometric grids, and **heterogeneous** per-node budgets.

## Layout

```
sim/                       simulation code (numpy + scipy + matplotlib)
  algorithms.py              BH, sample-and-forward (PX23), QuTE baselines
  journal_algorithms.py      certificate(), kround_star(), kround_qute(), hetero_*(), allocate()
  journal_experiments_r5.py  experiments S1–S7 reported in the paper
  journal_plots_r5.py        figures and LaTeX tables from sim/results/*.npz
  journal_tests.py           sanity tests (certificate identity, monotonicity, FDR, exact identity)
  data.py, verify.py, experiments.py, review_experiments.py, journal_experiments.py, plot.py
  results/                   saved per-trial outcomes (expS*.npz) used for every number in the paper
scripts/                   runners for the legacy weak-regime sweeps (Appendix D of the journal version)
output/                    created by the scripts: regenerated figures (PDF) and LaTeX tables
```

## Reproducing the results

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash reproduce.sh            # tests, experiments, figures/tables
```

`reproduce.sh` runs the steps below; each can be run on its own from the
repository root (the code uses `sim.` imports and relative paths).

1. **Tests** — all four should report 0 violations.
   ```bash
   for t in cert mono fdr ident; do python3 -m sim.journal_tests $t; done
   ```
2. **Experiments** — each call writes `sim/results/expS*.npz` with per-trial
   outcomes (power, FDP, counts, exchanges, certificate, …). Seeds are fixed
   in the function signatures, so the files shipped in `sim/results/` are
   reproduced exactly. Total running time is a few hours on a laptop; S1 and
   S5 are the slowest.
   ```bash
   for pi in 0.01 0.03 0.1 0.3; do python3 -m sim.journal_experiments_r5 S1 $pi; done
   for S in 6 8 10 12 16 20;    do python3 -m sim.journal_experiments_r5 S2 $S;  done
   for M in 3 4 5;              do python3 -m sim.journal_experiments_r5 S3 $M;  done
   python3 -m sim.journal_experiments_r5 S3u; python3 -m sim.journal_experiments_r5 S3n
   for v in fixed redirect_abstain fixed_levelcorr; do python3 -m sim.journal_experiments_r5 S4 $v; done
   for d in 2 4 8 16;           do python3 -m sim.journal_experiments_r5 S5 $d;  done
   for B in 8 10 14;            do python3 -m sim.journal_experiments_r5 S6 $B;  done
   python3 -m sim.journal_experiments_r5 S7
   ```
3. **Figures and tables** — writes `output/figs/*.pdf` and `output/tab_*.tex`
   (set `CAC_OUT` to choose another directory).
   ```bash
   python3 -m sim.journal_plots_r5
   ```
   The legacy sweeps of Appendix D (`fig_headline.pdf`) come from
   `scripts/run_weak_exps.py` followed by `python3 -m sim.plot`.
The paper itself (arXiv version and the IEEE TSP submission) is not part of this
repository; the arXiv identifier will be added here once available.

## Using the certificate in your own code

```python
import numpy as np
from sim.journal_algorithms import kround_star

# node_pvals: list of 1-d arrays, one per node
out = kround_star(node_pvals, alpha=0.2, M_list=[4, 4, 4], eps=0.2)
out['tau']        # threshold used for rejection
out['tau_bar']    # certified upper bound on the pooled-BH threshold
out['C']          # certificate: m*C bounds the number of BH rejections possibly missed
out['certified']  # True if tau_bar <= (1+eps) * tau held at stopping (relative certificate)
out['samples'], out['exchanges']
```

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
