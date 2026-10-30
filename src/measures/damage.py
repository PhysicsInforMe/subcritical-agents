"""Damage spreading (Derrida and Stanley 1987), design notes §4.2 test 3.

Two replicas evolve under the same sequence of random numbers, starting from
configurations that differ in a single agent. The Hamming distance
    D(t) = (1/N) sum_i [s_i^A(t) != s_i^B(t)]
grows (chaotic phase) or decays to zero (frozen phase). The growth rate is
estimated from the early-time log-linear regime.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def damage_spreading(step: Callable[[np.ndarray, np.random.Generator], None],
                     s0: np.ndarray, flip_index: int, n_steps: int, seed: int
                     ) -> dict[str, np.ndarray | float]:
    """Run two replicas with identical random streams.

    ``step(s, rng)`` advances a configuration in place using ``rng`` for all
    its randomness; two generators with the same seed give the same random
    sequence, which is the definition of the protocol.
    """
    a = np.array(s0, copy=True)
    b = np.array(s0, copy=True)
    b.flat[flip_index] *= -1
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed)
    D = np.empty(n_steps + 1)
    D[0] = np.mean(a != b)
    for t in range(1, n_steps + 1):
        step(a, rng_a)
        step(b, rng_b)
        D[t] = np.mean(a != b)
    # growth rate from log D on the window where 0 < D < 1/2
    mask = (D > 0) & (D < 0.5)
    if mask.sum() >= 3:
        tt = np.arange(D.size)[mask]
        rate = float(np.polyfit(tt, np.log(D[mask]), 1)[0])
    else:
        rate = float("nan")
    return {"hamming": D, "rate": rate}
