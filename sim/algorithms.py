# sim/algorithms.py
import numpy as np

def bh_reject(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Standard BH step-up. Returns boolean rejection mask over `pvals`."""
    m = len(pvals)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    thresholds = alpha * (np.arange(1, m + 1)) / m
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool)
    k_hat = np.max(np.where(below)[0]) + 1  # 1-indexed BH index
    tau = sorted_p[k_hat - 1]
    return pvals <= tau

def bonferroni_reject(pvals: np.ndarray, alpha_local: float) -> np.ndarray:
    return pvals <= alpha_local


def _empirical_cdf_samples(local_pvals: np.ndarray, locations: np.ndarray) -> np.ndarray:
    """F_i evaluated at each location: fraction of local_pvals <= each location."""
    if len(local_pvals) == 0:
        return np.zeros_like(locations, dtype=float)
    return np.searchsorted(np.sort(local_pvals), locations, side='right') / len(local_pvals)

def sample_and_forward_star(node_pvals: list, alpha: float, M: int) -> float:
    """
    Original Sample-and-Forward on a star network.
    `node_pvals[i]` is a 1D array of node i's p-values.
    Returns a single threshold tau_hat applied globally; caller can reject per node.
    """
    if M < 2:
        raise ValueError("M must be >= 2")
    locations = np.linspace(0, alpha, M)
    m_i = np.array([len(p) for p in node_pvals], dtype=float)
    m_total = m_i.sum()
    if m_total == 0:
        return 0.0
    F_samples = np.zeros(M)
    for i, p in enumerate(node_pvals):
        F_samples += (m_i[i] / m_total) * _empirical_cdf_samples(p, locations)
    # tau_hat = alpha * F_hat(Z_I) where I = max k with F_hat(Z_k) >= Z_k/alpha
    # (matches the original paper: tau = t_I + alpha*(S_I - t_I/alpha) = alpha*S_I).
    crossing = F_samples >= (locations / alpha)
    if not crossing.any():
        return 0.0
    I = int(np.max(np.where(crossing)[0]))
    return float(alpha * F_samples[I])


def adaptive_sample_and_forward_star(
    node_pvals: list,
    alpha: float,
    M1: int,
    M2: int,
    b: int = 0,
) -> float:
    """
    Two-pass adaptive Sample-and-Forward on a star network.
      M1: coarse-grid samples per node over [0, alpha]
      M2: refined-grid samples per node within the chosen bin
      b:  window inflation in bins (0 or 1)
    Returns the final threshold tau_hat.
    """
    if M1 < 2 or M2 < 2:
        raise ValueError("M1, M2 must be >= 2")

    # Round 1: coarse pass
    Z1 = np.linspace(0, alpha, M1)
    m_i = np.array([len(p) for p in node_pvals], dtype=float)
    m_total = m_i.sum()
    if m_total == 0:
        return 0.0
    F1 = np.zeros(M1)
    for i, p in enumerate(node_pvals):
        F1 += (m_i[i] / m_total) * _empirical_cdf_samples(p, Z1)
    # Find j_star = max k with F1[k] >= Z1[k]/alpha.
    # Z1[0]=0 trivially satisfies 0>=0; later indices dominate if they cross too.
    crossing1 = F1 >= (Z1 / alpha)
    if not crossing1.any():
        return 0.0
    j_star = int(np.max(np.where(crossing1)[0]))   # 0-indexed
    if j_star == M1 - 1:
        # threshold lies at the rightmost grid point; nothing to refine
        return float(alpha * F1[j_star])

    # Window
    lo_idx = max(0, j_star - b)
    hi_idx = min(M1 - 1, j_star + 1 + b)
    L, U = Z1[lo_idx], Z1[hi_idx]

    # Round 2: refined pass
    Z2 = np.linspace(L, U, M2)
    F2 = np.zeros(M2)
    for i, p in enumerate(node_pvals):
        F2 += (m_i[i] / m_total) * _empirical_cdf_samples(p, Z2)
    crossing2 = F2 >= (Z2 / alpha)
    if not crossing2.any():
        # Fallback: use round-1 threshold at j_star
        return float(alpha * F1[j_star])
    I = int(np.max(np.where(crossing2)[0]))
    return float(alpha * F2[I])


def qute_sample_and_forward(node_pvals, neighbors, alpha, M):
    """
    Original QuTE-Sample-and-Forward on a multihop graph.
      node_pvals[i]: array of node i's p-values
      neighbors[i]: list of indices of i's neighbors (including i itself, by convention)
      Returns: a list of length N of thresholds tau_hat[i] (one per node).

    Per [PX23]: for each node x the local test size is alpha_x = (n_x/m)*alpha,
    the grid lives in [0, alpha_x], and the crossing condition uses
    F_pool >= locations/alpha_x  (NOT the global alpha).
    """
    N = len(node_pvals)
    m_i = np.array([len(p) for p in node_pvals], dtype=float)
    m_total = m_i.sum()
    # For each node x, pool over its neighborhood using a per-node grid [0, alpha_x]
    tau0 = np.zeros(N)
    for x in range(N):
        nbrs = neighbors[x]
        n_x = sum(m_i[i] for i in nbrs)  # number of p-values in neighborhood
        if n_x <= 0:
            tau0[x] = 0.0; continue
        # Local test size alpha_x = (n_x / m_total) * alpha  [PX23 eq. (4)]
        alpha_x = (n_x / m_total) * alpha
        # Per-node grid lives in [0, alpha_x] for best resolution
        locations = np.linspace(0, alpha_x, M)
        # Each neighbor samples F_i on this node-specific grid
        F_pool = np.zeros(M)
        for i in nbrs:
            F_pool += (m_i[i] / n_x) * _empirical_cdf_samples(node_pvals[i], locations)
        # Crossing condition uses local alpha_x: F_pool(t) >= t/alpha_x
        crossing = F_pool >= (locations / alpha_x)
        if not crossing.any():
            tau0[x] = 0.0
        else:
            I = int(np.max(np.where(crossing)[0]))
            tau0[x] = float(alpha_x * F_pool[I])
    # Exchange: each node takes the max threshold over its neighborhood
    tau_final = np.zeros(N)
    for x in range(N):
        tau_final[x] = max(tau0[i] for i in neighbors[x])
    return tau_final


def adaptive_qute_sample_and_forward(node_pvals, neighbors, alpha, M1, M2, b=0):
    """
    Adaptive QuTE-Sample-and-Forward. Each node performs the adaptive procedure
    on its own neighborhood; thresholds are exchanged via max.

    Per [PX23]: for each node x the local test size is alpha_x = (n_x/m)*alpha.
    Round-1 grid lives in [0, alpha_x]; crossing condition uses F_pool >= Z1/alpha_x.
    Round-2 window [L,U] is taken from the round-1 grid (scaled to alpha_x).
    """
    N = len(node_pvals)
    m_i = np.array([len(p) for p in node_pvals], dtype=float)
    m_total = m_i.sum()
    # Each node pools over its neighborhood with a per-node grid
    tau0 = np.zeros(N)
    for x in range(N):
        nbrs = neighbors[x]
        n_x = sum(m_i[i] for i in nbrs)
        alpha_x = (n_x / m_total) * alpha
        if alpha_x <= 0:
            tau0[x] = 0.0; continue
        # Round 1: coarse grid in [0, alpha_x]
        Z1 = np.linspace(0, alpha_x, M1)
        F1_pool = np.zeros(M1)
        for i in nbrs:
            F1_pool += (m_i[i] / n_x) * _empirical_cdf_samples(node_pvals[i], Z1)
        # Crossing uses local alpha_x
        crossing1 = F1_pool >= (Z1 / alpha_x)
        if not crossing1.any():
            tau0[x] = 0.0; continue
        j_star = int(np.max(np.where(crossing1)[0]))
        if j_star == M1 - 1:
            tau0[x] = float(alpha_x * F1_pool[j_star]); continue
        lo_idx = max(0, j_star - b)
        hi_idx = min(M1 - 1, j_star + 1 + b)
        L, U = Z1[lo_idx], Z1[hi_idx]
        # Round 2: refined grid within [L, U] (both endpoints from the alpha_x-scaled grid)
        Z2 = np.linspace(L, U, M2)
        F2_pool = np.zeros(M2)
        for i in nbrs:
            F2_pool += (m_i[i] / n_x) * _empirical_cdf_samples(node_pvals[i], Z2)
        # Crossing still uses alpha_x
        crossing2 = F2_pool >= (Z2 / alpha_x)
        if not crossing2.any():
            tau0[x] = float(alpha_x * F1_pool[j_star])
        else:
            I = int(np.max(np.where(crossing2)[0]))
            tau0[x] = float(alpha_x * F2_pool[I])
    # Exchange (same as original QuTE)
    tau_final = np.zeros(N)
    for x in range(N):
        tau_final[x] = max(tau0[i] for i in neighbors[x])
    return tau_final


def adaptive_iterative_sf_star(node_pvals, alpha, K=3, M_per_round=3):
    """K-round zoom; M_per_round samples each round; window shrinks each round."""
    m_i = np.array([len(p) for p in node_pvals], dtype=float)
    m_total = m_i.sum()
    if m_total == 0:
        return 0.0
    L, U = 0.0, alpha
    tau_hat = 0.0
    for r in range(K):
        Z = np.linspace(L, U, M_per_round)
        F = np.zeros(M_per_round)
        for i, p in enumerate(node_pvals):
            F += (m_i[i] / m_total) * _empirical_cdf_samples(p, Z)
        crossing = F >= (Z / alpha)
        if not crossing.any():
            break  # keep last successful tau_hat
        I = int(np.max(np.where(crossing)[0]))
        tau_hat = alpha * F[I]
        if I == M_per_round - 1:
            # at right boundary; can't zoom further without expanding
            break
        L, U = Z[I], Z[I + 1]
    return float(tau_hat)
