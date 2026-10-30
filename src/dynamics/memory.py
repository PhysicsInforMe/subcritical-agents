"""Asynchronous heat-bath dynamics on a policy that depends on the observed
peers and on the agent's own memory of its last k_mem states.

The policy is a dict keyed by (k, memory tuple) giving P(+). Memory is the
tuple of the agent's own previous states, oldest first, initialised by
repeating the initial state (declared modelling choice: an agent starts
with a consistent history rather than an empty one, because the prompt
always shows k_mem past states). Exposure draws a fresh random subset of
peers at every update, which also randomises their order (§14.1 B2).

Every update is logged as (round, agent, k_seen, memory_sum, new_state) so
that the De Nobili regression can be re-run per round on the simulated
logs (§14.1 B14) and compared with the tabulated parameters.
"""
from __future__ import annotations

import numpy as np


def run_memory_dynamics(policy: dict, adjacency: list[np.ndarray], s0: np.ndarray, k_mem: int,
                        exposure: int | None, n_rounds: int, seed: int,
                        stubborn: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return states per round (n_rounds + 1, N), magnetisation, update log.

    ``stubborn`` is an optional int8 array: 0 free, +1 or -1 pinned (test 4).
    """
    rng = np.random.default_rng(seed)
    N = s0.size
    s = np.array(s0, dtype=np.int8, copy=True)
    mem = [tuple([int(s[i])] * k_mem) for i in range(N)]
    states = np.empty((n_rounds + 1, N), dtype=np.int8)
    states[0] = s
    log = []
    for r in range(1, n_rounds + 1):
        for _ in range(N):
            i = int(rng.integers(N))
            if stubborn is not None and stubborn[i] != 0:
                s[i] = stubborn[i]
                continue
            nb = adjacency[i]
            if exposure is not None and nb.size > exposure:
                nb = rng.choice(nb, size=exposure, replace=False)
            k = int(s[nb].sum())
            p = policy[(k, mem[i])]
            new = 1 if rng.random() < p else -1
            log.append((r, i, k, sum(mem[i]), new))
            if k_mem:
                mem[i] = tuple(list(mem[i][1:]) + [int(s[i])])
            s[i] = new
        states[r] = s
    return {"states": states, "m": states.mean(axis=1), "log": np.array(log, dtype=np.int64),
            "final": s}
