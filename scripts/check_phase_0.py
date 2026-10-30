#!/usr/bin/env python
"""Definition of done for Phase 0 (design notes §5, week 1; §14.2).

This script is the gate between Phase 0 and Phase A. It does not run
simulations itself: it reads the result manifests that the Phase 0 scripts
must write under ``results/phase0/`` and fails (exit code 1) unless every
exit criterion of §5 is met with the required uncertainty.

Why a reader and not a runner: §14.2 requires that measures are always
computed downstream from raw results, never inline, and that every run
leaves a manifest with config, seeds, commit hash and timestamp. Keeping the
check separate from the producers also lets it be run on any machine in a
second, which is the point of a gate.

Exit criteria (§5, "Week 1 — Fase 0", updated 2026-09-08) and the file
each one is read from:

A. Ising 2D calibration                    results/phase0/ising_calibration.json
   A1. T_c estimate consistent with Onsager, T_c = 2 / ln(1 + sqrt 2).
   A2. gamma/nu consistent with 7/4 and beta/nu with 1/8 (finite-size
       scaling of chi_|m| peak and |m| at T_c over at least four even L).
   A3. Binder crossing at h = 0 within tolerance of T_c (Binder only here,
       §14.1 B9).
B. Synthetic recovery of J and h           results/phase0/regression_recovery.json
   B1. Two-term fit logit P(+|k) = 2(h + J k) recovers known (J, h) on
       Glauber-generated data, within the bootstrap interval.
   B2. Three-term fit logit P(+|k, y) = 2(h_bias + h_spec y + J k)
       recovers known (J, h_bias, h_spec) on synthetic data with a
       truth-aligned random field (§14.1 B1).
   B3. Recovery study reported per parameter with bias and coverage over
       repeated synthetic draws (standard of Brockers et al., §10.12).
C. Drift control                           results/phase0/drift_control.json
   C1. Voter model (J = 0, h = 0, finite N) reaches consensus by noise
       alone with t_cons growing as N^2 in interaction steps (§10.21).
       Fitted exponent must be consistent with 2.
D. De Nobili check with logits             results/phase0/denobili_check.json
   D1. Policies P(+ | k), k in {-4, -2, 0, 2, 4}, tabulated from Qwen3
       logits with constrained prefix and summed token variants
       (§14.1 B5), under the global-flip protocol (§14.1 B6).
   D2. Two-term fit gives h_tilde >> J_tilde (ratio above threshold) for
       at least one model and one temperature, reproducing the
       field-dominated regime of De Nobili (§10.2).
   D3. Order-dependence diagnostic: variance of P(+) across neighbour
       orderings at fixed k reported (§14.1 B3).
   D4. Detailed-balance cycle test on the tabulated table: sum over closed
       3-cycles of log ratios of transition probabilities reported, with
       the deviation attributable to order dependence (§10.28).
   D5. Numpy dynamics on the tabulated policy reproduces the chi_|m|
       curve shape of De Nobili with a field-dominated crossover, no
       Ising universality claimed.

Every criterion must carry a bootstrap or seed-based uncertainty; a point
estimate without an interval fails (§14.2, "nessuna soglia riportata senza
intervallo"). Every manifest must carry the keys listed in REQUIRED_MANIFEST.

JSON schemas expected (all numbers are floats unless stated):

results/phase0/ising_calibration.json
  {
    "manifest": {...},
    "L_values": [int, ...],                # even L only, at least 4 values
    "Tc": {"estimate": x, "ci95": [lo, hi]},
    "gamma_over_nu": {"estimate": x, "ci95": [lo, hi]},
    "beta_over_nu": {"estimate": x, "ci95": [lo, hi]},
    "binder_crossing": {"estimate": x, "ci95": [lo, hi]}
  }

results/phase0/regression_recovery.json
  {
    "manifest": {...},
    "two_term": {
      "true": {"J": x, "h": x},
      "fit":  {"J": {"estimate": x, "ci95": [lo, hi]},
               "h": {"estimate": x, "ci95": [lo, hi]}}
    },
    "three_term": {
      "true": {"J": x, "h_bias": x, "h_spec": x},
      "fit":  {"J": {...}, "h_bias": {...}, "h_spec": {...}}
    },
    "recovery_study": {
      "n_draws": int,
      "coverage95": {"J": x, "h_bias": x, "h_spec": x},
      "relative_bias": {"J": x, "h_bias": x, "h_spec": x}
    }
  }

results/phase0/drift_control.json
  {
    "manifest": {...},
    "N_values": [int, ...],                # at least 4 values
    "t_cons_exponent": {"estimate": x, "ci95": [lo, hi]}
  }

results/phase0/denobili_check.json
  {
    "manifest": {...},
    "cells": [
      {
        "model": "Qwen/Qwen3-0.6B", "revision": "...", "temperature": x,
        "policy": {"-4": x, "-2": x, "0": x, "2": x, "4": x},
        "policy_flipped": {...},           # global-flip sector
        "fit": {"J": {"estimate": x, "ci95": [lo, hi]},
                "h": {"estimate": x, "ci95": [lo, hi]}},
        "order_variance": {"-4": x, ..., "4": x},
        "cycle_test": {"mean_abs_cycle_sum": x, "n_cycles": int},
        "chi_abs_curve": {"T": [...], "chi_abs": [...], "chi_abs_err": [...]}
      }, ...
    ]
  }

Run:  python scripts/check_phase_0.py [--results results/phase0] [--strict]
Exit: 0 if every criterion passes, 1 otherwise. Prints one line per
criterion with PASS / FAIL / MISSING and the numbers used.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Reference values and tolerances
# ---------------------------------------------------------------------------

# Onsager (1944): T_c = 2 / ln(1 + sqrt(2)) in units of J / k_B.
TC_ONSAGER = 2.0 / math.log(1.0 + math.sqrt(2.0))          # 2.26918...
GAMMA_OVER_NU = 7.0 / 4.0                                    # 1.75
BETA_OVER_NU = 1.0 / 8.0                                     # 0.125

# Tolerances are deliberately loose for CPU-sized lattices (L <= 32, a few
# thousand sweeps). They are absolute windows around the exact values, used
# in addition to the requirement that the exact value lies inside the CI.
TOL = {
    "Tc": 0.03,             # |Tc_hat - Tc| <= 0.03  (about 1.3 %)
    "gamma_over_nu": 0.15,  # half a decade of L gives ~ +-0.1 to 0.15
    "beta_over_nu": 0.05,
    "binder_crossing": 0.05,
    "drift_exponent": 0.3,  # t_cons ~ N^2, effective exponent within +-0.3
}

# De Nobili (§10.2): h_tilde >> J_tilde. We require the ratio to exceed 3
# in at least one cell; De Nobili reports ~8 at worst (llama, lowest T).
H_OVER_J_MIN = 3.0
MIN_EVEN_L = 4
MIN_N_VALUES = 4
MIN_RECOVERY_DRAWS = 100
COVERAGE95_MIN = 0.85        # nominal 0.95, allow finite-draw slack
RELATIVE_BIAS_MAX = 0.10

REQUIRED_MANIFEST = ("config", "seeds", "commit", "timestamp")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class Report:
    """Collects PASS/FAIL/MISSING lines and the overall verdict."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failed = 0

    def ok(self, code: str, msg: str) -> None:
        self.lines.append(f"PASS     {code}  {msg}")

    def fail(self, code: str, msg: str) -> None:
        self.failed += 1
        self.lines.append(f"FAIL     {code}  {msg}")

    def missing(self, code: str, msg: str) -> None:
        self.failed += 1
        self.lines.append(f"MISSING  {code}  {msg}")

    def dump(self) -> None:
        print("\n".join(self.lines))
        verdict = "PHASE 0 NOT DONE" if self.failed else "PHASE 0 DONE"
        print(f"\n{verdict}: {self.failed} criteria failing.")


def load(path: Path, rep: Report, code: str) -> dict | None:
    if not path.exists():
        rep.missing(code, f"{path} not found")
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        rep.fail(code, f"{path} is not valid JSON: {exc}")
        return None


def check_manifest(data: dict, rep: Report, code: str) -> None:
    man = data.get("manifest")
    if not isinstance(man, dict):
        rep.fail(code, "manifest missing")
        return
    absent = [k for k in REQUIRED_MANIFEST if k not in man]
    if absent:
        rep.fail(code, f"manifest lacks {absent}")
    else:
        rep.ok(code, f"manifest complete (commit {str(man['commit'])[:8]})")


def estimate_with_ci(obj: dict | None, code: str, rep: Report,
                     name: str) -> tuple[float, float, float] | None:
    """Return (estimate, lo, hi) or None; a value without a CI fails."""
    if not isinstance(obj, dict) or "estimate" not in obj or "ci95" not in obj:
        rep.fail(code, f"{name}: estimate or ci95 missing (no threshold without interval)")
        return None
    est = float(obj["estimate"])
    lo, hi = (float(v) for v in obj["ci95"])
    if not (lo <= est <= hi):
        rep.fail(code, f"{name}: estimate {est:.4g} outside its own ci95 [{lo:.4g}, {hi:.4g}]")
        return None
    return est, lo, hi


def check_against_reference(obj: dict | None, ref: float, tol: float,
                            code: str, rep: Report, name: str) -> None:
    """Pass if |est - ref| <= tol and the reference lies within tol of the
    95 % interval.

    The interval is statistical (bootstrap over seeds); the tolerance is the
    allowance for systematic corrections to scaling that a fit on L <= 32
    cannot resolve (effective exponents, De Nobili §10.2). Requiring the
    reference strictly inside a narrow statistical interval would fail a
    correct simulation for the wrong reason, so the two are combined:
    the distance from the reference to the interval must not exceed tol.
    """
    got = estimate_with_ci(obj, code, rep, name)
    if got is None:
        return
    est, lo, hi = got
    dist_to_interval = max(lo - ref, ref - hi, 0.0)
    close = abs(est - ref) <= tol
    msg = f"{name} = {est:.4f} [{lo:.4f}, {hi:.4f}] vs {ref:.4f} (tol {tol})"
    if close and dist_to_interval <= tol:
        note = "" if dist_to_interval == 0 else f" (reference {dist_to_interval:.4f} outside ci95, within tol)"
        rep.ok(code, msg + note)
    else:
        why = []
        if dist_to_interval > tol:
            why.append("reference farther than tol from ci95")
        if not close:
            why.append("outside tolerance")
        rep.fail(code, msg + " -> " + ", ".join(why))


# ---------------------------------------------------------------------------
# A. Ising calibration
# ---------------------------------------------------------------------------

def check_ising(results: Path, rep: Report) -> None:
    data = load(results / "ising_calibration.json", rep, "A")
    if data is None:
        return
    check_manifest(data, rep, "A")
    L = data.get("L_values", [])
    if len(L) < MIN_EVEN_L or any(int(l) % 2 for l in L):
        rep.fail("A0", f"need at least {MIN_EVEN_L} even L values, got {L} (§11.5)")
    else:
        rep.ok("A0", f"L values {L}")
    check_against_reference(data.get("Tc"), TC_ONSAGER, TOL["Tc"], "A1", rep, "T_c")
    check_against_reference(data.get("gamma_over_nu"), GAMMA_OVER_NU,
                            TOL["gamma_over_nu"], "A2", rep, "gamma/nu")
    check_against_reference(data.get("beta_over_nu"), BETA_OVER_NU,
                            TOL["beta_over_nu"], "A2", rep, "beta/nu")
    check_against_reference(data.get("binder_crossing"), TC_ONSAGER,
                            TOL["binder_crossing"], "A3", rep, "Binder crossing")


# ---------------------------------------------------------------------------
# B. Regression recovery
# ---------------------------------------------------------------------------

def _check_block(block: dict | None, params: tuple[str, ...], code: str,
                 rep: Report, label: str) -> None:
    if not isinstance(block, dict) or "true" not in block or "fit" not in block:
        rep.fail(code, f"{label}: 'true' or 'fit' missing")
        return
    for name in params:
        ref = block["true"].get(name)
        if ref is None:
            rep.fail(code, f"{label}: true {name} missing")
            continue
        got = estimate_with_ci(block["fit"].get(name), code, rep, f"{label} {name}")
        if got is None:
            continue
        est, lo, hi = got
        if lo <= float(ref) <= hi:
            rep.ok(code, f"{label} {name} = {est:.4f} [{lo:.4f}, {hi:.4f}] covers {float(ref):.4f}")
        else:
            rep.fail(code, f"{label} {name} = {est:.4f} [{lo:.4f}, {hi:.4f}] misses {float(ref):.4f}")


def check_regression(results: Path, rep: Report) -> None:
    data = load(results / "regression_recovery.json", rep, "B")
    if data is None:
        return
    check_manifest(data, rep, "B")
    _check_block(data.get("two_term"), ("J", "h"), "B1", rep, "two-term")
    _check_block(data.get("three_term"), ("J", "h_bias", "h_spec"), "B2", rep, "three-term")
    rs = data.get("recovery_study")
    if not isinstance(rs, dict):
        rep.fail("B3", "recovery_study missing")
        return
    n = int(rs.get("n_draws", 0))
    if n < MIN_RECOVERY_DRAWS:
        rep.fail("B3", f"recovery study has {n} draws, need >= {MIN_RECOVERY_DRAWS}")
    for name in ("J", "h_bias", "h_spec"):
        cov = rs.get("coverage95", {}).get(name)
        bias = rs.get("relative_bias", {}).get(name)
        if cov is None or bias is None:
            rep.fail("B3", f"{name}: coverage95 or relative_bias missing")
            continue
        cov, bias = float(cov), abs(float(bias))
        msg = f"{name}: coverage95 {cov:.3f} (>= {COVERAGE95_MIN}), |rel bias| {bias:.3f} (<= {RELATIVE_BIAS_MAX})"
        if cov >= COVERAGE95_MIN and bias <= RELATIVE_BIAS_MAX:
            rep.ok("B3", msg)
        else:
            rep.fail("B3", msg)


# ---------------------------------------------------------------------------
# C. Drift control (voter model, §10.21)
# ---------------------------------------------------------------------------

def check_drift(results: Path, rep: Report) -> None:
    data = load(results / "drift_control.json", rep, "C")
    if data is None:
        return
    check_manifest(data, rep, "C")
    N = data.get("N_values", [])
    if len(N) < MIN_N_VALUES:
        rep.fail("C0", f"need at least {MIN_N_VALUES} N values, got {N}")
    else:
        rep.ok("C0", f"N values {N}")
    check_against_reference(data.get("t_cons_exponent"), 2.0, TOL["drift_exponent"],
                            "C1", rep, "t_cons exponent")


# ---------------------------------------------------------------------------
# D. De Nobili check with logits (§10.2, §4.3)
# ---------------------------------------------------------------------------

K_BINS = ("-4", "-2", "0", "2", "4")


def check_denobili(results: Path, rep: Report) -> None:
    data = load(results / "denobili_check.json", rep, "D")
    if data is None:
        return
    check_manifest(data, rep, "D")
    cells = data.get("cells", [])
    if not cells:
        rep.fail("D1", "no cells")
        return
    any_field_dominated = False
    for idx, cell in enumerate(cells):
        tag = f"cell {idx} ({cell.get('model', '?')}, T={cell.get('temperature', '?')})"
        # D1: tabulated policy in both flip sectors, all bins in (0, 1).
        ok_policy = True
        for key in ("policy", "policy_flipped"):
            pol = cell.get(key, {})
            if any(b not in pol for b in K_BINS):
                rep.fail("D1", f"{tag}: {key} lacks bins {K_BINS}")
                ok_policy = False
                continue
            if any(not (0.0 <= float(pol[b]) <= 1.0) for b in K_BINS):
                rep.fail("D1", f"{tag}: {key} has probabilities outside [0, 1]")
                ok_policy = False
            saturated = [b for b in K_BINS if float(pol[b]) in (0.0, 1.0)]
            if saturated:
                # a gap of tens of logits underflows to exactly 0 or 1: the
                # policy is deterministic at this T (the saturation case of
                # §14.1 B4), reported but not a failure
                rep.lines.append(f"INFO     D1  {tag}: {key} saturated at bins {saturated}")
        if ok_policy:
            rep.ok("D1", f"{tag}: policy tabulated in both flip sectors")
        if "revision" not in cell:
            rep.fail("D1", f"{tag}: HF revision not pinned (§14.1 B5)")
        # D2: h_tilde >> J_tilde with intervals.
        fit = cell.get("fit", {})
        gotJ = estimate_with_ci(fit.get("J"), "D2", rep, f"{tag} J")
        goth = estimate_with_ci(fit.get("h"), "D2", rep, f"{tag} h")
        if gotJ and goth:
            J, h = gotJ[0], goth[0]
            ratio = abs(h) / abs(J) if abs(J) > 1e-12 else math.inf
            if ratio >= H_OVER_J_MIN:
                any_field_dominated = True
                rep.ok("D2", f"{tag}: |h|/|J| = {ratio:.2f} (h={h:.3f}, J={J:.3f}) field-dominated")
            else:
                rep.lines.append(f"INFO     D2  {tag}: |h|/|J| = {ratio:.2f} not field-dominated")
        # D3: order-dependence diagnostic reported.
        ov = cell.get("order_variance")
        if isinstance(ov, dict) and all(b in ov for b in K_BINS):
            rep.ok("D3", f"{tag}: order variance reported, max {max(float(ov[b]) for b in K_BINS):.4f}")
        else:
            rep.fail("D3", f"{tag}: order_variance missing (§14.1 B3)")
        # D4: detailed-balance cycle test reported.
        ct = cell.get("cycle_test")
        if isinstance(ct, dict) and "mean_abs_cycle_sum" in ct and int(ct.get("n_cycles", 0)) > 0:
            rep.ok("D4", f"{tag}: cycle test on {ct['n_cycles']} cycles, mean |sum| {float(ct['mean_abs_cycle_sum']):.4f}")
        else:
            rep.fail("D4", f"{tag}: cycle_test missing (§10.28)")
        # D5: chi_|m| curve with errors.
        curve = cell.get("chi_abs_curve", {})
        T, chi, err = curve.get("T", []), curve.get("chi_abs", []), curve.get("chi_abs_err", [])
        if len(T) >= 5 and len(T) == len(chi) == len(err):
            rep.ok("D5", f"{tag}: chi_|m| curve on {len(T)} temperatures with errors")
        else:
            rep.fail("D5", f"{tag}: chi_abs_curve needs >= 5 temperatures with errors")
    if any_field_dominated:
        rep.ok("D2", "field-dominated regime (h >> J) reproduced in at least one cell")
    else:
        rep.fail("D2", f"no cell with |h|/|J| >= {H_OVER_J_MIN}; if this persists it is a finding (§11.7), not a bug")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results/phase0",
                        help="directory with the Phase 0 result manifests")
    parser.add_argument("--strict", action="store_true",
                        help="reserved: currently identical to the default")
    args = parser.parse_args(argv)
    results = Path(args.results)
    rep = Report()
    print(f"Phase 0 gate on {results.resolve()}\n")
    check_ising(results, rep)
    check_regression(results, rep)
    check_drift(results, rep)
    check_denobili(results, rep)
    rep.dump()
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
