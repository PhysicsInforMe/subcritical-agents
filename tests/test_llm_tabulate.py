"""Tests of the pure (model-free) parts of the tabulation module."""
import math

import numpy as np
import pytest
from scipy.special import expit

from src.llm import cycle_test, enumerate_orderings, fit_table, policy_at_temperature


def test_orderings_count():
    assert len(enumerate_orderings(4, 0)) == 6
    assert len(enumerate_orderings(4, 2)) == 4
    assert len(enumerate_orderings(4, 4)) == 1
    with pytest.raises(ValueError):
        enumerate_orderings(4, 1)


def test_temperature_rescales_gap():
    assert policy_at_temperature(2.0, 0.0, 1.0) == pytest.approx(expit(2.0))
    assert policy_at_temperature(2.0, 0.0, 2.0) == pytest.approx(expit(1.0))


def test_fit_table_recovers_logistic():
    h, J = 0.7, 0.15
    gaps = {str(k): [2 * (h + J * k)] * len(enumerate_orderings(4, k)) for k in (-4, -2, 0, 2, 4)}
    fit = fit_table(gaps, T=1.0, n_boot=20)
    assert fit["h"]["estimate"] == pytest.approx(h, abs=1e-9)
    assert fit["J"]["estimate"] == pytest.approx(J, abs=1e-9)


def test_set_labels_changes_prompts_and_restores_default():
    from src.llm.phase_a import build_prompt_a
    from src.llm.tabulate import LABELS, build_prompt, set_labels
    wording = "States {a} or {b}. Peers: {peers}.{memory} Answer."
    default = build_prompt_a(wording, "list", (1, -1, 1), (), LABELS[0], "comma")
    assert default == "States A or B. Peers: A, B, A. Answer."
    assert build_prompt((1, -1), "A", "{a}/{b}: {states}") == "A/B: A, B"
    set_labels("X", "Y")
    try:
        assert build_prompt_a(wording, "list", (1, -1, 1), (), LABELS[0], "comma") == "States X or Y. Peers: X, Y, X. Answer."
        assert build_prompt((1, -1), "X", "{a}/{b}: {states}") == "X/Y: X, Y"
    finally:
        set_labels("A", "B")
    assert build_prompt_a(wording, "list", (1, -1, 1), (), LABELS[0], "comma") == default


def test_cycle_test_zero_for_logistic_and_nonzero_otherwise():
    logistic = {k: float(expit(2 * (0.3 + 0.4 * k))) for k in range(-4, 5)}
    out = cycle_test(logistic, n_samples=500)
    assert out["max_abs_cycle_sum"] < 1e-9
    # majority-vote-like table: logit not linear in k -> detailed balance broken
    majority = {k: 0.9 if k > 0 else 0.1 if k < 0 else 0.5 for k in range(-4, 5)}
    out2 = cycle_test(majority, n_samples=500)
    assert out2["mean_abs_cycle_sum"] > 0.1
