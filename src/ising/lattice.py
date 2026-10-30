"""2D Ising model, periodic square lattice, in units J = k_B = 1.

Hamiltonian
    H = -J sum_<ij> s_i s_j,      s_i in {-1, +1},

with the exact critical temperature (Onsager 1944)
    T_c = 2 / ln(1 + sqrt 2) = 2.26918...
and the 2D exponents gamma/nu = 7/4, beta/nu = 1/8, nu = 1.

Two update schemes are provided and both are used in Phase 0:

* ``metropolis_sweep``: single-spin checkerboard Metropolis. It is the
  local dynamics whose stochastic version (heat bath, Glauber 1963) is the
  model of the LLM update rule, so the measurement pipeline must be tested
  on it. Near T_c it suffers critical slowing down, tau ~ L^z with z ~ 2.2,
  which makes finite-size scaling on a CPU laptop unreliable for L >= 24.
* ``swendsen_wang_sweep``: cluster update (Swendsen and Wang 1987), same
  equilibrium distribution, z ~ 0.3. It is used only to sample the
  equilibrium observables for the calibration (T_c, gamma/nu, beta/nu,
  Binder). Clusters are found with scipy's connected_components on the
  sparse bond graph, which handles the periodic wrap that ndimage.label
  would not.

Only even L are used (De Nobili et al., §10.2 and §11.5): even periodic
lattices are bipartite, which keeps the checkerboard update exact and the
finite-size corrections clean.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

T_C_ONSAGER: float = 2.0 / math.log(1.0 + math.sqrt(2.0))


def random_state(L: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random spin configuration, shape (L, L), values +-1."""
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))


def neighbour_sum(s: np.ndarray) -> np.ndarray:
    """Sum of the four periodic neighbours of every site, k_i = sum_j s_j."""
    return (np.roll(s, 1, 0) + np.roll(s, -1, 0)
            + np.roll(s, 1, 1) + np.roll(s, -1, 1)).astype(np.int16)


def magnetization(s: np.ndarray) -> float:
    """m = (1/N) sum_i s_i, signed."""
    return float(s.mean())


def energy_per_spin(s: np.ndarray) -> float:
    """E/N = -(1/N) sum_<ij> s_i s_j, each bond counted once."""
    return float(-(s * (np.roll(s, 1, 0) + np.roll(s, 1, 1))).mean())


def metropolis_sweep(s: np.ndarray, beta: float, rng: np.random.Generator) -> None:
    """One checkerboard Metropolis sweep in place.

    A flip of spin i changes the energy by dE = 2 s_i k_i. Metropolis accepts
    with probability min(1, exp(-beta dE)). Updating one sublattice at a
    time keeps the neighbour sums fixed during the vectorised update, so the
    move is exactly the sequential algorithm restricted to a colour class.
    """
    L = s.shape[0]
    if L % 2:
        raise ValueError("checkerboard update needs even L")
    ii, jj = np.indices(s.shape)
    parity = (ii + jj) % 2
    for colour in (0, 1):
        mask = parity == colour
        k = neighbour_sum(s)
        dE = 2.0 * s * k
        accept = (dE <= 0) | (rng.random(s.shape) < np.exp(-beta * np.clip(dE, 0, None)))
        s[mask & accept] *= -1


def swendsen_wang_sweep(s: np.ndarray, beta: float, rng: np.random.Generator) -> None:
    """One Swendsen-Wang cluster sweep in place.

    Bonds between parallel neighbours are opened with p = 1 - exp(-2 beta J);
    every connected cluster of open bonds is flipped with probability 1/2.
    """
    L = s.shape[0]
    N = L * L
    idx = np.arange(N).reshape(L, L)
    p = 1.0 - math.exp(-2.0 * beta)
    right = np.roll(idx, -1, 1)
    down = np.roll(idx, -1, 0)
    par_r = (s == np.roll(s, -1, 1)) & (rng.random(s.shape) < p)
    par_d = (s == np.roll(s, -1, 0)) & (rng.random(s.shape) < p)
    rows = np.concatenate([idx[par_r], idx[par_d]])
    cols = np.concatenate([right[par_r], down[par_d]])
    graph = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(N, N))
    n_comp, labels = connected_components(graph, directed=False)
    flip = rng.random(n_comp) < 0.5
    s[flip[labels].reshape(L, L)] *= -1


def simulate(L: int, T: float, seed: int, n_equil: int, n_meas: int,
             algorithm: str = "sw", thin: int = 1) -> dict[str, np.ndarray]:
    """Equilibrate and record m and E per measured sweep.

    Returns a dict with arrays ``m`` (signed magnetisation per spin) and
    ``e`` (energy per spin), length n_meas // thin.
    """
    rng = np.random.default_rng(seed)
    beta = 1.0 / T
    s = random_state(L, rng)
    step = swendsen_wang_sweep if algorithm == "sw" else metropolis_sweep
    for _ in range(n_equil):
        step(s, beta, rng)
    n_rec = n_meas // thin
    m = np.empty(n_rec)
    e = np.empty(n_rec)
    for t in range(n_meas):
        step(s, beta, rng)
        if t % thin == 0:
            m[t // thin] = magnetization(s)
            e[t // thin] = energy_per_spin(s)
    return {"m": m, "e": e}
