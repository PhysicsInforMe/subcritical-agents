"""Hysteresis with stubborn agents (De Marzo et al. protocol, §4.2 test 4).

A fraction z of agents is pinned to a fixed opinion, z is swept up and then
down, and the stationary magnetisation of the free agents is recorded on
both branches. The gap between branches is the hysteresis; the tipping
point z_c is where the branch jumps. With the mean-field policy
P(+ | k) = sigma(2(h + J k)) the spinodal of De Marzo gives the prediction.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def stubborn_sweep(run_stationary: Callable[[float, np.ndarray, np.random.Generator], float],
                   z_values: np.ndarray, s0: np.ndarray, seed: int) -> dict[str, np.ndarray | float]:
    """Sweep z up and down.

    ``run_stationary(z, s, rng)`` must evolve ``s`` in place with a fraction
    z of stubborn agents (positive z pins to +1, negative to -1) and return
    the stationary magnetisation of the free agents. The state is carried
    from one z to the next, which is what creates the memory of the branch.
    """
    rng = np.random.default_rng(seed)
    s = np.array(s0, copy=True)
    up = np.array([run_stationary(z, s, rng) for z in z_values])
    down = np.array([run_stationary(z, s, rng) for z in z_values[::-1]])[::-1]
    gap = np.abs(up - down)
    return {"z": np.asarray(z_values), "up": up, "down": down, "gap": gap,
            "max_gap": float(gap.max()), "z_at_max_gap": float(z_values[int(np.argmax(gap))])}
