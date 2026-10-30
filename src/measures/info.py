"""Information measures on binary agent states (design notes §4.2 test 1).

For N binary variables X_1..X_N with joint distribution p:

    TC(X)    = sum_i H(X_i) - H(X)                       (total correlation)
    DTC(X)   = sum_i H(X_{-i}) - (N - 1) H(X)            (dual total correlation)
    Omega(X) = TC(X) - DTC(X)                            (O-information, Rosas 2019)

Omega > 0: redundancy dominates (consensus). Omega < 0: synergy dominates,
the only marker of structure beyond mean field (§14.1 A2). Plug-in
estimation with Jeffreys pseudo-counts on the joint table, feasible only for
small N (2^N cells); for larger N estimate on random subsets of agents.
"""
from __future__ import annotations

import itertools

import numpy as np

from .order import shannon_entropy


def _joint_table(states: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Jeffreys-smoothed joint pmf of N binary columns from samples (rows)."""
    states = np.asarray(states)
    n_samples, N = states.shape
    if N > 20:
        raise ValueError("joint table too large; subsample agents")
    bits = ((states + 1) // 2).astype(np.int64)
    codes = bits @ (1 << np.arange(N))
    counts = np.bincount(codes, minlength=1 << N).astype(float)
    counts += alpha
    return counts / counts.sum()


def _marginal(p: np.ndarray, keep: tuple[int, ...], N: int) -> np.ndarray:
    p = p.reshape([2] * N)
    axes = tuple(i for i in range(N) if i not in keep)
    return p.sum(axis=axes).ravel() if axes else p.ravel()


def total_correlation_binary(states: np.ndarray, alpha: float = 0.5) -> float:
    """TC in bits from samples of N binary agents (rows = samples)."""
    N = states.shape[1]
    p = _joint_table(states, alpha)
    H_joint = shannon_entropy(p)
    H_marg = sum(shannon_entropy(_marginal(p, (i,), N)) for i in range(N))
    return H_marg - H_joint


def o_information_binary(states: np.ndarray, alpha: float = 0.5) -> float:
    """O-information Omega = TC - DTC in bits, N binary agents."""
    N = states.shape[1]
    p = _joint_table(states, alpha)
    H_joint = shannon_entropy(p)
    H_marg = sum(shannon_entropy(_marginal(p, (i,), N)) for i in range(N))
    H_minus = sum(shannon_entropy(_marginal(p, tuple(j for j in range(N) if j != i), N))
                  for i in range(N))
    tc = H_marg - H_joint
    dtc = H_minus - (N - 1) * H_joint
    return tc - dtc


def o_information_subsets(states: np.ndarray, subset_size: int, n_subsets: int,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """Omega on random agent subsets, for N too large for the joint table."""
    rng = rng or np.random.default_rng(0)
    N = states.shape[1]
    vals = []
    for _ in range(n_subsets):
        idx = rng.choice(N, size=subset_size, replace=False)
        vals.append(o_information_binary(states[:, idx]))
    return np.array(vals)


__all__ = ["total_correlation_binary", "o_information_binary", "o_information_subsets",
           "itertools"]
