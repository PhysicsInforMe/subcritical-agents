"""Ising sampler and dynamics tests against exact values."""
import math

import numpy as np
import pytest

from src.dynamics import glauber_policy_from_params, run_tabulated, voter_consensus_time
from src.ising import T_C_ONSAGER, energy_per_spin, simulate
from src.topology import adjacency_lists, lattice_2d


def test_onsager_constant():
    assert T_C_ONSAGER == pytest.approx(2.269185, abs=1e-6)


@pytest.mark.parametrize("algorithm", ["sw", "metropolis"])
def test_high_temperature_energy_matches_series(algorithm):
    # High-T expansion: e = -2 tanh(beta) + O(beta^3) for 2D square lattice.
    T = 8.0
    out = simulate(L=16, T=T, seed=11, n_equil=200, n_meas=1000, algorithm=algorithm)
    assert out["e"].mean() == pytest.approx(-2 * math.tanh(1 / T), abs=0.02)


def test_low_temperature_ordered():
    out = simulate(L=8, T=1.0, seed=12, n_equil=200, n_meas=300, algorithm="sw")
    assert np.abs(out["m"]).mean() > 0.98
    assert out["e"].mean() == pytest.approx(-2.0, abs=0.02)


def test_energy_per_spin_ground_state():
    assert energy_per_spin(np.ones((6, 6), dtype=np.int8)) == -2.0


def test_tabulated_policy_matches_ising_lattice():
    # Glauber policy with J = 1/T, h = 0 on the lattice is the Ising heat bath:
    # at T = 1.5 the lattice is ordered, |m| large; at T = 4 disordered.
    L = 8
    adj = adjacency_lists(lattice_2d(L))
    y = np.zeros(L * L, dtype=int)
    rng = np.random.default_rng(0)
    s0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=L * L)
    for T, expect_ordered in ((1.5, True), (4.0, False)):
        pol = glauber_policy_from_params(z=4, J=1.0 / T, h_bias=0.0)
        out = run_tabulated(pol, adj, s0, y, n_sweeps=300, seed=1)
        abs_m = np.abs(out["m"][150:]).mean()
        assert (abs_m > 0.8) == expect_ordered


def test_global_flip_conjugate():
    pol = glauber_policy_from_params(z=4, J=0.2, h_bias=0.5)
    flipped = pol.with_global_flip()
    for (k, y), p in pol.table.items():
        assert flipped.table[(-k, -y)] == pytest.approx(1 - p)


def test_voter_consensus_scales_as_N_squared():
    # mean fixation time from a balanced start ~ c N^2 single updates
    Ns = [8, 16, 32]
    means = [np.mean([voter_consensus_time(N, seed) for seed in range(40)]) for N in Ns]
    slope = np.polyfit(np.log(Ns), np.log(means), 1)[0]
    assert 1.6 < slope < 2.4
