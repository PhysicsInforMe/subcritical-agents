"""Asynchronous dynamics on a graph driven by a tabulated policy.

The policy gives P(s_i' = +1 | k_i, y_i) where k_i is the sum of the peer
states the agent observes and y_i is the item truth (+-1, or 0 when there
is no designed field). Flint et al. (PNAS 2026) precompute such policies
from logits and simulate at zero LLM cost; De Nobili et al. do the same in
sampled form. The dynamics is heat-bath (Glauber): at each step one agent is
chosen at random and resampled from the policy. With the logistic form
P(+ | k) = sigma(2(h + J k)) this is the Glauber dynamics of an Ising model
with field h and coupling J, and satisfies detailed balance. A table read
from an LLM need not have that form (§14.1 B3), which is the point of
tabulating it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit


@dataclass
class TabulatedPolicy:
    """P(+ | k, y) as a lookup on integer k in [-z, z] and y in {-1, 0, +1}."""
    z: int
    table: dict[tuple[int, int], float] = field(default_factory=dict)

    def p_plus(self, k: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array([self.table[(int(kk), int(yy))] for kk, yy in zip(k, y)])

    def with_global_flip(self) -> "TabulatedPolicy":
        """The Z2-conjugate table: P'(+ | k, y) = 1 - P(+ | -k, -y).

        Averaging a run and its flipped run cancels the label bias in odd
        moments (De Nobili global-flip, §14.1 B6).
        """
        flipped = {(-k, -y): 1.0 - p for (k, y), p in self.table.items()}
        return TabulatedPolicy(self.z, flipped)


def glauber_policy_from_params(z: int, J: float, h_bias: float, h_spec: float = 0.0
                               ) -> TabulatedPolicy:
    """Logistic policy sigma(2(h_bias + h_spec y + J k)) on all (k, y)."""
    table = {}
    for k in range(-z, z + 1):
        for y in (-1, 0, 1):
            table[(k, y)] = float(expit(2.0 * (h_bias + h_spec * y + J * k)))
    return TabulatedPolicy(z, table)


def run_tabulated(policy: TabulatedPolicy, adjacency: list[np.ndarray], s0: np.ndarray,
                  y: np.ndarray, n_sweeps: int, seed: int, record_every: int = 1,
                  exposure: int | None = None) -> dict[str, np.ndarray]:
    """Asynchronous heat-bath dynamics on the graph.

    ``adjacency[i]`` is the array of neighbours of agent i. ``exposure``
    limits the number of peers shown at each update to a random subset of
    that size (the exposure knob, design notes §5 Fase A); peers are drawn
    afresh at every update, which also randomises their order (§14.1 B2).
    Returns the magnetisation per recorded sweep and the final state.
    """
    rng = np.random.default_rng(seed)
    s = np.array(s0, dtype=np.int8, copy=True)
    N = s.size
    n_rec = n_sweeps // record_every
    m = np.empty(n_rec)
    states = np.empty((n_rec, N), dtype=np.int8)
    for sweep in range(n_sweeps):
        for _ in range(N):
            i = int(rng.integers(N))
            nb = adjacency[i]
            if exposure is not None and nb.size > exposure:
                nb = rng.choice(nb, size=exposure, replace=False)
            k = int(s[nb].sum())
            p = policy.table[(k, int(y[i]))]
            s[i] = 1 if rng.random() < p else -1
        if sweep % record_every == 0:
            m[sweep // record_every] = s.mean()
            states[sweep // record_every] = s
    return {"m": m, "states": states, "final": s}
