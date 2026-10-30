"""Unit tests for the decomposition of e J~ - F'(0) (scripts/phase_a_F0_decomposition.py).

Exact checks: (1) the closed-form slope equals the numerical derivative of the map
of phase_a_meanfield_stability.py; (2) for the additive unbiased policy at 4 peers
the slope is tanh(2 J) + tanh(4 J) / 2; (3) for an additive biased table the odd
part recovers the unbiased slope and the symmetrisation in probability keeps the
bias in the slope, F_sym'(0) = F_A'(0) < F_odd'(0); (4) the sequential steps sum
to the total in both orders.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_F0_decomposition import contributions, policies, sigma, slope_at_zero, sym_prob  # noqa: E402
from scripts.phase_a_meanfield_stability import mf_map  # noqa: E402


def additive_table(J: float, h: float, e: int = 4) -> dict[int, np.ndarray]:
    """One ordering per k with gap exactly 2(h + J k)."""
    return {k: np.array([2 * (h + J * k)]) for k in range(-e, e + 1, 2)}


def test_closed_form_slope_matches_numerical_derivative():
    p = {k: 1 / (1 + math.exp(-0.9 * k + 0.4)) for k in range(-4, 5, 2)}
    _, dF = mf_map(p, 4)
    assert abs(slope_at_zero(p, 4) - dF(0.0)) < 1e-6


def test_additive_slope_is_tanh_formula_at_4_peers():
    for J in (0.1, 0.4, 1.0):
        p = {k: float(sigma(2 * J * k)) for k in range(-4, 5, 2)}
        assert abs(slope_at_zero(p, 4) - (math.tanh(2 * J) + 0.5 * math.tanh(4 * J))) < 1e-12


def test_bias_stays_in_the_probability_symmetrised_slope():
    J, h = 0.4, 0.8
    r = policies(additive_table(J, h), 4)
    assert abs(r["J"] - J) < 1e-9 and abs(r["h"] - h) < 1e-9
    assert abs(r["F0_odd"] - r["F0_add"]) < 1e-9          # odd part removes the bias exactly
    assert r["F0_even"] < r["F0_odd"] - 0.05               # the bias flattens the slope through sigma'
    pA = {k: float(sigma(2 * (h + J * k))) for k in range(-4, 5, 2)}
    _, dFA = mf_map(pA, 4)
    assert abs(slope_at_zero(sym_prob(pA), 4) - dFA(0.0)) < 1e-6   # F_sym'(0) = F_A'(0)


def test_steps_sum_to_the_total_in_both_orders():
    rng = np.random.default_rng(1)
    gaps = {k: 0.6 * k + 0.5 + rng.normal(0, 0.7, size=4) for k in range(-4, 5, 2)}
    r = policies(gaps, 4); c = contributions(r)
    total = r["F0_paper"] - r["eJ"]
    o1 = c["saturation"] + c["shape_k"] + c["order_bias_then_disp"]["bias"] + c["order_bias_then_disp"]["dispersion"]
    o2 = c["saturation"] + c["shape_k"] + c["order_disp_then_bias"]["dispersion"] + c["order_disp_then_bias"]["bias"]
    assert abs(o1 - total) < 1e-12 and abs(o2 - total) < 1e-12
    assert abs(c["shapley"]["bias"] + c["shapley"]["dispersion"] - (r["F0_paper"] - r["F0_odd"])) < 1e-12
