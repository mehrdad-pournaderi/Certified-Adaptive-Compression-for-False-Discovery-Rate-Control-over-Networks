"""Runner script for all experiments. Call with experiment number as argument.
Each experiment saves its .npz incrementally so partial runs can be resumed."""
import sys
import numpy as np
import os
import time

sys.path.insert(0, '/sessions/awesome-sweet-einstein/mnt/outputs')

from sim.experiments import (
    experiment_1, experiment_2, experiment_3, experiment_4,
    experiment_5, experiment_6, experiment_7,
    generate_star_network, sample_pvals_independent, sample_pvals_dependent,
    run_methods_one_trial, save_npz, RESULTS_DIR, pooled_bh_tau,
    adaptive_sample_and_forward_star,
)
from sim.verify import fdr_power
from sim.algorithms import adaptive_iterative_sf_star

DISPATCH = {
    '1': lambda: experiment_1(n_trials=2000),
    '2': lambda: experiment_2(n_trials=2000),
    '3': lambda: experiment_3(n_trials=2000),
    '4': lambda: experiment_4(n_trials=500),
    '5': lambda: experiment_5(n_trials=1500),
    '6': lambda: experiment_6(n_trials=1500),
    '7': lambda: experiment_7(n_trials=1500),
}

if __name__ == '__main__':
    exp = sys.argv[1]
    t0 = time.time()
    DISPATCH[exp]()
    print(f'Elapsed: {time.time()-t0:.1f}s')
