# sim/data.py
import numpy as np
from scipy.stats import norm, multivariate_normal

def generate_star_network(N: int, lam: float, rng: np.random.Generator):
    """Return: list of (m_i, pi1_i, mu_base_i, is_null mask per node) tuples."""
    nodes = []
    for i in range(1, N + 1):
        m_i = rng.poisson(lam)
        if m_i == 0:
            m_i = 1
        pi1 = 0.5 - 0.4 * (i / N)
        mu_base = 2.0 + 4.0 * (i / N)
        is_null = rng.random(m_i) >= pi1
        nodes.append({'m': m_i, 'pi1': pi1, 'mu_base': mu_base, 'is_null': is_null})
    return nodes

def generate_star_network_weak(N: int, lam: float, rng: np.random.Generator):
    """Weak-signal variant of generate_star_network.

    Design choice (option a): separate function rather than adding optional
    parameters to generate_star_network.  This keeps the original function
    untouched and makes the weak-signal experiment self-contained.

    mu_base = 1.0 + 1.5*(i/N)  -> range [1.01, 2.5]  (vs [2.04, 6.0] original)
    pi1     = 0.3 constant      (vs 0.5 - 0.4*(i/N) original)
    """
    nodes = []
    for i in range(1, N + 1):
        m_i = rng.poisson(lam)
        if m_i == 0:
            m_i = 1
        pi1 = 0.3
        mu_base = 1.0 + 1.5 * (i / N)
        is_null = rng.random(m_i) >= pi1
        nodes.append({'m': m_i, 'pi1': pi1, 'mu_base': mu_base, 'is_null': is_null})
    return nodes


def two_sided_pvals(z: np.ndarray) -> np.ndarray:
    return 2.0 * (1.0 - norm.cdf(np.abs(z)))

def sample_pvals_independent(nodes, rng):
    """Returns: list of arrays of p-values, parallel list of is_null masks."""
    pvals, nulls = [], []
    for node in nodes:
        m, mu_base, is_null = node['m'], node['mu_base'], node['is_null']
        mu = np.where(is_null, 0.0,
                      rng.uniform(mu_base - 0.5, mu_base + 0.5, size=m))
        z = rng.normal(loc=mu, scale=1.0)
        pvals.append(two_sided_pvals(z))
        nulls.append(is_null)
    return pvals, nulls

def cycle_graph_neighbors(N: int) -> list:
    """Cycle of N nodes; each neighborhood = {self, left, right}."""
    return [[i, (i - 1) % N, (i + 1) % N] for i in range(N)]


def random_graph_neighbors(N: int, avg_degree: float, seed: int) -> list:
    """Hamiltonian cycle augmented with random edges (NOT a standard
    Erdos-Renyi graph): every pair not already adjacent is joined
    independently with probability p = (avg_degree - 2)/(N - 1), so the
    expected average degree is 2 + (avg_degree - 2)(N - 3)/(N - 1).
    The cycle makes the graph connected.  Returns list of neighborhoods
    (each is a list of indices including self)."""
    rng = np.random.default_rng(seed)
    nbrs = [{i} for i in range(N)]
    # Spine: cycle
    for i in range(N):
        j = (i + 1) % N
        nbrs[i].add(j); nbrs[j].add(i)
    # Random edges
    p = max(0.0, (avg_degree - 2) / (N - 1))
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p:
                nbrs[i].add(j); nbrs[j].add(i)
    return [sorted(s) for s in nbrs]


def grid_2d_neighbors(side: int) -> list:
    """side x side 2D grid graph. Each node has up to 4 neighbors (N/S/E/W) plus self."""
    N = side * side
    nbrs = []
    for idx in range(N):
        row, col = divmod(idx, side)
        nb = {idx}
        if row > 0:          nb.add((row - 1) * side + col)
        if row < side - 1:   nb.add((row + 1) * side + col)
        if col > 0:          nb.add(row * side + col - 1)
        if col < side - 1:   nb.add(row * side + col + 1)
        nbrs.append(sorted(nb))
    return nbrs


def sample_pvals_dependent(nodes, rho: float, rng):
    """Tapering covariance Sigma_ij = rho^|i-j| across the *flattened* p-values."""
    m_total = sum(node['m'] for node in nodes)
    if rho == 0.0:
        return sample_pvals_independent(nodes, rng)
    # Build a Toeplitz covariance
    idx = np.arange(m_total)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    # Cholesky once
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(m_total))
    # Mean vector
    mu = np.zeros(m_total)
    cursor = 0
    for node in nodes:
        m, mu_base, is_null = node['m'], node['mu_base'], node['is_null']
        non_null_mu = rng.uniform(mu_base - 0.5, mu_base + 0.5, size=m)
        mu[cursor:cursor + m] = np.where(is_null, 0.0, non_null_mu)
        cursor += m
    z = mu + L @ rng.standard_normal(m_total)
    p_flat = two_sided_pvals(z)
    # Split back per node
    pvals, nulls = [], []
    cursor = 0
    for node in nodes:
        m = node['m']
        pvals.append(p_flat[cursor:cursor + m])
        nulls.append(node['is_null'])
        cursor += m
    return pvals, nulls
