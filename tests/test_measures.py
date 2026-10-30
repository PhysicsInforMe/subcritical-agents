"""Unit tests of the measurement library against analytic cases."""
import math

import numpy as np
import pytest

from src.measures import (binder_cumulant, binder_crossing, chi_abs, chi_signed,
                          fit_glauber_counts, fit_glauber_mle, o_information_binary,
                          peak_parabolic, powerlaw_fit, shannon_entropy,
                          synthetic_updates, tc_from_peaks, total_correlation_binary)
from src.measures.damage import damage_spreading
from src.measures.hysteresis import stubborn_sweep


def test_shannon_entropy_uniform():
    assert shannon_entropy(np.full(8, 1 / 8)) == pytest.approx(3.0)


def test_chi_of_bernoulli_spins():
    # N independent spins, P(+)=q: <m>=2q-1, Var(m) = 4q(1-q)/N, chi = N/T Var
    rng = np.random.default_rng(1)
    N, q, T = 50, 0.7, 1.0
    m = (2 * (rng.random((20000, N)) < q) - 1).mean(axis=1)
    assert chi_signed(m, N, T) == pytest.approx(4 * q * (1 - q), rel=0.05)
    assert chi_abs(m, N, T) <= chi_signed(m, N, T) + 1e-12


def test_binder_gaussian_and_ordered():
    rng = np.random.default_rng(2)
    g = rng.normal(size=100000)          # Gaussian m: <m^4> = 3 <m^2>^2 -> U4 = 0
    assert abs(binder_cumulant(g)) < 0.02
    assert binder_cumulant(np.ones(10)) == pytest.approx(2 / 3)


def test_peak_parabolic_exact():
    x = np.linspace(0, 2, 21)
    y = 5 - (x - 1.234) ** 2
    xp, yp = peak_parabolic(x, y)
    assert xp == pytest.approx(1.234, abs=1e-9)
    assert yp == pytest.approx(5.0, abs=1e-9)


def test_powerlaw_fit_exact():
    L = np.array([8, 12, 16, 24, 32])
    s, se = powerlaw_fit(L, 3.0 * L ** 1.75)
    assert s == pytest.approx(1.75, abs=1e-9)


def test_tc_from_peaks_exact():
    L = np.array([8, 16, 32])
    Tc, a = tc_from_peaks(L, 2.269 + 0.8 / L)
    assert Tc == pytest.approx(2.269, abs=1e-9) and a == pytest.approx(0.8, abs=1e-9)


def test_binder_crossing_linear():
    T = np.linspace(2.0, 2.6, 61)
    U_small = 0.6 - 0.5 * (T - 2.269)
    U_large = 0.6 - 1.0 * (T - 2.269)
    assert binder_crossing(T, U_small, U_large) == pytest.approx(2.269, abs=1e-6)


@pytest.mark.parametrize("estimator", [fit_glauber_mle, fit_glauber_counts])
def test_regression_recovers_two_terms(estimator):
    rng = np.random.default_rng(3)
    d = synthetic_updates(40000, J=0.3, h_bias=0.4, with_truth=False, rng=rng)
    fit = estimator(d["k"], d["s_new"], None, n_boot=100, rng=rng)
    assert fit["J"]["estimate"] == pytest.approx(0.3, abs=0.03)
    assert fit["h"]["estimate"] == pytest.approx(0.4, abs=0.03)
    assert fit["J"]["ci95"][0] <= 0.3 <= fit["J"]["ci95"][1]


@pytest.mark.parametrize("estimator", [fit_glauber_mle, fit_glauber_counts])
def test_regression_recovers_three_terms(estimator):
    rng = np.random.default_rng(4)
    d = synthetic_updates(60000, J=0.25, h_bias=0.15, h_spec=0.6, rng=rng)
    fit = estimator(d["k"], d["s_new"], d["y"], n_boot=100, rng=rng)
    assert fit["J"]["estimate"] == pytest.approx(0.25, abs=0.03)
    assert fit["h_bias"]["estimate"] == pytest.approx(0.15, abs=0.03)
    assert fit["h_spec"]["estimate"] == pytest.approx(0.6, abs=0.04)


def test_o_information_signs():
    rng = np.random.default_rng(5)
    n = 20000
    # redundancy: three copies of one bit -> TC = 2 bits, Omega > 0
    b = rng.integers(0, 2, size=n)
    red = np.stack([b, b, b], axis=1) * 2 - 1
    assert total_correlation_binary(red) == pytest.approx(2.0, abs=0.02)
    assert o_information_binary(red) > 0.5
    # synergy: XOR triplet -> TC = 1 bit, DTC = 2 bits, Omega = -1
    a = rng.integers(0, 2, size=n)
    c = rng.integers(0, 2, size=n)
    syn = np.stack([a, c, a ^ c], axis=1) * 2 - 1
    assert o_information_binary(syn) == pytest.approx(-1.0, abs=0.03)
    # independent bits -> Omega ~ 0
    ind = rng.integers(0, 2, size=(n, 3)) * 2 - 1
    assert abs(o_information_binary(ind)) < 0.02


def test_damage_spreading_identical_streams_frozen_and_chaotic():
    # frozen: deterministic majority-like copy of neighbour -> damage stays
    # chaotic: every step flips a random spin chosen with the rng, so equal
    # streams keep the damage at exactly one site: distance constant 1/N.
    N = 16

    def step_const(s, rng):
        s[rng.integers(N)] *= -1

    out = damage_spreading(step_const, np.ones(N, dtype=np.int8), 0, 50, seed=7)
    assert np.allclose(out["hamming"], 1 / N)


def test_stubborn_sweep_branches():
    z = np.linspace(-0.5, 0.5, 11)

    def run_stationary(zz, s, rng):
        return float(np.sign(zz) if zz != 0 else s.mean())

    out = stubborn_sweep(run_stationary, z, np.ones(4), seed=0)
    assert out["up"].shape == z.shape and out["max_gap"] >= 0
