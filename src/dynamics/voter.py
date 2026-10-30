"""Voter model on the complete graph: the drift control (design notes §10.21).

With J = 0 and h = 0 in the sense of an Ising policy there is no ordering
force, yet a finite population reaches consensus by sampling noise alone:
each update copies a random other agent, the magnetisation is a martingale
and fixation happens in a time that grows as N^2 in single updates (N in
population rounds). This is the "memetic drift" of Tanaka (arXiv:2603.24676)
and the reference against which a finite-N consensus must be checked
before it is called order.
"""
from __future__ import annotations

import numpy as np


def voter_consensus_time(N: int, seed: int, max_steps: int | None = None) -> int:
    """Number of single-agent updates until all N agents agree, starting
    from a balanced random configuration. Returns -1 if max_steps is hit."""
    rng = np.random.default_rng(seed)
    s = np.ones(N, dtype=np.int8)
    s[: N // 2] = -1
    rng.shuffle(s)
    n_plus = int((s == 1).sum())
    max_steps = max_steps or 200 * N * N
    for t in range(1, max_steps + 1):
        i, j = rng.integers(N, size=2)
        if s[i] != s[j]:
            n_plus += int(s[j]) if s[j] == 1 else -1
            s[i] = s[j]
            if n_plus == 0 or n_plus == N:
                return t
    return -1
