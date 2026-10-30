"""Unit tests for the mean-field criterion and the gap estimator (revision R4-12).

Two exact checks: (1) for an unbiased logistic policy P(+ | k) = sigma(2 J k) with
small J the slope of the map at the disordered point is F'(0) = e J; (2) for the
deterministic majority rule the slope equals the combinatorial ceiling E_0|k|
(1.000, 1.500, 2.188, 2.461, 4.334 at 2, 4, 7, 10, 30 peers). A third check: the gap
estimator recovers J and h from gaps that are exactly 2(h + J k).
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_format_check import j_gap  # noqa: E402
from scripts.phase_a_meanfield_stability import fixed_points, mf_map  # noqa: E402


def ceiling(e: int) -> float:
    return sum(math.comb(e, n) / 2 ** e * abs(2 * n - e) for n in range(e + 1))


def test_logistic_policy_slope_is_eJ():
    for e in (2, 4, 7):
        for J in (0.005, 0.02):
            p = {k: 1 / (1 + math.exp(-2 * J * k)) for k in range(-e, e + 1, 2)}
            _, dF = mf_map(p, e)
            assert abs(dF(0.0) - e * J) < 1e-2 * e * J   # the logistic map deviates from e J at order (e J)^3


def test_majority_rule_reaches_the_ceiling():
    for e, c in ((2, 1.000), (4, 1.500), (7, 2.188), (10, 2.461), (30, 4.334)):
        p = {k: (1.0 if k > 0 else 0.0 if k < 0 else 0.5) for k in range(-e, e + 1, 2)}
        _, dF = mf_map(p, e)
        assert abs(dF(0.0) - ceiling(e)) < 2e-3
        assert abs(ceiling(e) - c) < 6e-4


def test_fixed_points_of_a_bistable_map():
    e = 4
    p = {k: 1 / (1 + math.exp(-2 * 0.6 * k)) for k in range(-e, e + 1, 2)}   # e J = 2.4 > 1: three fixed points
    F, dF = mf_map(p, e)
    fps = fixed_points(F, dF)
    stable = [f for f in fps if f["stable"]]
    assert len(stable) == 2 and stable[0]["m"] < 0 < stable[1]["m"]
    assert any(abs(f["m"]) < 1e-3 and not f["stable"] for f in fps)


def test_gap_estimator_recovers_J():
    J, h = 0.7, 0.3
    g = {k: 2 * (h + J * k) for k in range(-4, 5, 2)}
    assert abs(j_gap(g) - J) < 1e-9
