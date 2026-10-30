#!/usr/bin/env python
"""Decomposition of the gap between the linear criterion e J~ and the mean-field
slope F'(0) at 4 peers (round 5, colleague's doubt of 2026-10-24; declared in
the project log (not included) the same day; correction row in the results notes (not included)).

Every quantity is a slope at m = 0 of the binomial mean-field map of
phase_a_meanfield_stability.py, F(m) = 2 sum_n Bin(n; e, (1+m)/2) P(+ | k=2n-e) - 1,
computed for a different policy P(+ | k) built from the same table:

  eJ        the linear criterion e J~, J~ the slope over two of the regression of the
            mean gap on k (the paper's gap estimator), h~ its intercept over two
            (gap = 2(h~ + J~ k): h~ and J~ share units, the bias in logit is 2 h~);
  add       P = sigma(2 J~ k): the additive, unbiased policy. At 4 peers
            F'(0) = tanh(2 J~) + tanh(4 J~) / 2 (the colleague's formula; in general
            F'(0) = E_0[k tanh(J~ k)]). eJ - add is the saturation;
  add_bias  P_A = sigma(2 h~ + 2 J~ k), then symmetrised in probability as the paper
            does (the colleague's second calculation);
  odd       P = sigma((g(k) - g(-k)) / 2), g(k) the mean gap over the orderings with
            sum k: the odd part of the table in logit space, bias and even part
            removed, shape in k kept, no dispersion. add - odd is the shape in k;
  even      P_A = sigma(g(k)), then symmetrised in probability: adds back the even
            part of the table (label bias h~ and its dependence on k). odd - even is
            the bias step, which acts through sigma'(h~) (sech^2(h~) in the linear
            regime: the mean-field criterion of the random-field Ising model, since
            the symmetrised map is the exact mean-field map of a population half in
            sector A and half in sector B);
  disp      P = mean over pairs of orderings (i at k, j at -k) of
            sigma((g_i(k) - g_j(-k)) / 2): dispersion without bias;
  paper     P_A(k) = mean over orderings of sigma(g_i(k)), symmetrised in probability:
            the value of phase_a_meanfield_stability.py (dF0_sym). even - paper is the
            dispersion between orderings at fixed k (Jensen on sigma), the policy of
            an agent that draws the peers and their order at every update.

Identity used: F_B(m) = -F_A(-m), so the symmetrised map is (F_A + F_B)/2 and
F_sym'(0) = F_A'(0). The symmetrisation moves the fixed point back to m = 0 but does
not remove the bias from the slope.

The decomposition is sequential and therefore depends on the order of the steps.
Both orders of the last two steps are reported (shape, bias, dispersion) and
(shape, dispersion, bias), with the Shapley mean of the two and the interaction
term. Declared rule: a contribution is quoted as a fraction of the residual in
the paper only if it changes by less than a quarter of the residual between the
two orders; otherwise both orders or the Shapley mean are reported. Counts of
formulations with F'(0) > 1 are reported with the formulations within 0.02 of
the threshold set apart.

Sensitivity: F'(0) is linear in the ordering-averaged probabilities, so a Gaussian
perturbation of standard deviation s on every gap gives
std F'(0) <= s / 4 at 4 peers (sigma' <= 1/4); the exact value for the table and
the fraction of 400 draws above 1 are computed for the paper quantity and for the
odd part (seed 0). At 4 peers every ordering is enumerated, so there is no sampling
error and this is the only uncertainty; s is the precision difference of Table 7
(0.6B: bfloat16 against float32 at 2 peers, a perturbation the float32 tables do
not have; 1.7B: mean of the two standard deviations, the error of its bfloat16
tables).

Output: results/phase_a/F0_decomposition_e4.json and F0_decomposition_e4.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402

MARGIN = 0.02


def sigma(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def slope_at_zero(p: dict[int, float], e: int) -> float:
    """F'(0) = 2 E_0[k P(+ | k)] for the binomial map at m = 0 (exact, closed form:
    d/dm Bin(n; e, (1+m)/2) at m = 0 equals Bin(n; e, 1/2) (2n - e))."""
    ns = np.arange(e + 1)
    ks = 2 * ns - e
    w = binom.pmf(ns, e, 0.5)
    return float(np.sum(w * ks * np.array([2.0 * p[int(k)] - 1.0 for k in ks])))


def sym_prob(pA: dict[int, float]) -> dict[int, float]:
    return {k: 0.5 * (pA[k] + 1.0 - pA[-k]) for k in pA}


def gap_fit(gaps: dict[int, np.ndarray]) -> tuple[float, float]:
    """J~ = slope/2 and h~ = intercept/2 of the regression of the mean gap on k."""
    ks = np.array(sorted(gaps), float)
    g = np.array([gaps[int(k)].mean() for k in ks])
    b, a = np.polyfit(ks, g, 1)
    return float(b / 2), float(a / 2)


def policies(gaps: dict[int, np.ndarray], e: int) -> dict[str, float]:
    ks = sorted(gaps)
    J, h = gap_fit(gaps)
    gbar = {k: float(gaps[k].mean()) for k in ks}
    p_add = {k: float(sigma(2 * J * k)) for k in ks}
    p_add_bias = sym_prob({k: float(sigma(2 * h + 2 * J * k)) for k in ks})
    p_odd = {k: float(sigma(0.5 * (gbar[k] - gbar[-k]))) for k in ks}
    p_even = sym_prob({k: float(sigma(gbar[k])) for k in ks})
    p_disp = {k: float(np.mean([sigma(0.5 * (a - b)) for a, b in product(gaps[k], gaps[-k])])) for k in ks}
    p_paper = sym_prob({k: float(sigma(gaps[k]).mean()) for k in ks})
    out = {"J": J, "h": h, "eJ": e * J, "sech2_h": float(1.0 / np.cosh(h) ** 2),
           "even_part": {str(k): 0.5 * (gbar[k] + gbar[-k]) for k in ks if k >= 0}}
    for name, p in (("add", p_add), ("add_bias", p_add_bias), ("odd", p_odd), ("even", p_even),
                    ("disp", p_disp), ("paper", p_paper)):
        out[f"F0_{name}"] = slope_at_zero(p, e)
    return out


def contributions(r: dict) -> dict:
    """Sequential steps in both orders and the Shapley mean of bias and dispersion."""
    sat = r["F0_add"] - r["eJ"]
    shape = r["F0_odd"] - r["F0_add"]
    bias_first = r["F0_even"] - r["F0_odd"]; disp_after_bias = r["F0_paper"] - r["F0_even"]
    disp_first = r["F0_disp"] - r["F0_odd"]; bias_after_disp = r["F0_paper"] - r["F0_disp"]
    residual = r["F0_paper"] - r["F0_add"]
    return {"saturation": sat, "shape_k": shape, "residual_after_saturation": residual,
            "order_bias_then_disp": {"bias": bias_first, "dispersion": disp_after_bias},
            "order_disp_then_bias": {"dispersion": disp_first, "bias": bias_after_disp},
            "shapley": {"bias": 0.5 * (bias_first + bias_after_disp), "dispersion": 0.5 * (disp_first + disp_after_bias)},
            "interaction_bias_x_disp": bias_after_disp - bias_first}


def sensitivity(gaps: dict[int, np.ndarray], e: int, s: float, n_draws: int, rng: np.random.Generator) -> dict:
    """Exact std of F'(0) under independent Gaussian errors of sd s on every gap
    (linear propagation, dF'(0)/dg_i = 2 w_k k sigma'(g_i) / n_k) and Monte Carlo
    fractions of draws with F'(0) > 1, for the paper quantity and the odd part."""
    ns = np.arange(e + 1); ks = 2 * ns - e; w = dict(zip((int(k) for k in ks), binom.pmf(ns, e, 0.5)))
    var = 0.0
    for k, g in gaps.items():
        pr = sigma(g); var += float(np.sum((2 * w[k] * k * pr * (1 - pr) / len(g)) ** 2))
    above_paper = above_odd = 0
    for _ in range(n_draws):
        noisy = {k: g + rng.normal(0.0, s, size=g.shape) for k, g in gaps.items()}
        gbar = {k: float(v.mean()) for k, v in noisy.items()}
        above_paper += slope_at_zero(sym_prob({k: float(sigma(v).mean()) for k, v in noisy.items()}), e) > 1.0
        above_odd += slope_at_zero({k: float(sigma(0.5 * (gbar[k] - gbar[-k]))) for k in gbar}, e) > 1.0
    return {"s": s, "std_F0_linear": float(np.sqrt(s * s * var)), "bound_s_over_4": s / 4,
            "frac_above_1_paper": above_paper / n_draws, "frac_above_1_odd": above_odd / n_draws}


def counts(rows: list[dict], key: str) -> dict:
    v = np.array([r[key] for r in rows])
    return {"above_1": int((v > 1.0 + MARGIN).sum()), "within_margin": int((np.abs(v - 1.0) <= MARGIN).sum()),
            "below_1": int((v < 1.0 - MARGIN).sum()), "above_1_point": int((v > 1.0).sum()), "mean": float(v.mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="qwen0.6b,qwen1.7b")
    ap.add_argument("--exposure", type=int, default=4)
    ap.add_argument("--noise-sd", default="qwen0.6b=0.12,qwen1.7b=0.37",
                    help="precision perturbation per model, logit (Table 7 of the paper)")
    ap.add_argument("--n-draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    noise = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.noise_sd.split(",")}
    e = a.exposure
    out = {"exposure": e, "margin": MARGIN, "models": {}}
    csv_rows = []
    keys = ("eJ", "F0_add", "F0_add_bias", "F0_odd", "F0_even", "F0_disp", "F0_paper")
    for model in a.models.split(","):
        rng = np.random.default_rng(a.seed)
        cell = f"A_{model}_list_k0_e{e}_Tan_f0_all_N8"
        d = json.load(open(Path(a.results) / cell / "gaps.json", encoding="utf-8"))
        rows = []
        for tab in d["tables"]:
            gaps: dict[int, list[float]] = {}
            for x in tab["entries"]:
                gaps.setdefault(x["k"], []).append(x["gap"])
            g = {k: np.array(v) for k, v in gaps.items()}
            r = {"cell": cell, "wording": tab["wording"], "format": tab["format"],
                 "n_orderings": {str(k): len(v) for k, v in g.items()}, **policies(g, e)}
            r["contributions"] = contributions(r)
            r["sensitivity"] = sensitivity(g, e, noise[model], a.n_draws, rng)
            rows.append(r)
            csv_rows.append({"model": model, "wording": tab["wording"], "format": tab["format"],
                             "h_tilde": round(r["h"], 4), **{k: round(r[k], 4) for k in keys},
                             "std_F0_noise": round(r["sensitivity"]["std_F0_linear"], 4)})
        strong = [r for r in rows if r["F0_odd"] > 0.9]
        summary = {"means": {k: float(np.mean([r[k] for r in rows])) for k in keys},
                   "counts": {k: counts(rows, k) for k in keys},
                   "contributions_mean": {
                       "saturation": float(np.mean([r["contributions"]["saturation"] for r in rows])),
                       "shape_k": float(np.mean([r["contributions"]["shape_k"] for r in rows])),
                       "residual_after_saturation": float(np.mean([r["contributions"]["residual_after_saturation"] for r in rows])),
                       "bias_then_disp": {q: float(np.mean([r["contributions"]["order_bias_then_disp"][q] for r in rows])) for q in ("bias", "dispersion")},
                       "disp_then_bias": {q: float(np.mean([r["contributions"]["order_disp_then_bias"][q] for r in rows])) for q in ("bias", "dispersion")},
                       "shapley": {q: float(np.mean([r["contributions"]["shapley"][q] for r in rows])) for q in ("bias", "dispersion")},
                       "interaction_bias_x_disp": float(np.mean([r["contributions"]["interaction_bias_x_disp"] for r in rows]))},
                   "n_strong_odd_above_0.9": len(strong),
                   "strong_F0_odd_range": [float(min(r["F0_odd"] for r in strong)), float(max(r["F0_odd"] for r in strong))] if strong else None,
                   "strong_F0_paper_range": [float(min(r["F0_paper"] for r in strong)), float(max(r["F0_paper"] for r in strong))] if strong else None,
                   "sech2_h_range": [float(min(r["sech2_h"] for r in rows)), float(max(r["sech2_h"] for r in rows))],
                   "sensitivity": {"s": noise[model],
                                   "std_F0_max": float(max(r["sensitivity"]["std_F0_linear"] for r in rows)),
                                   "n_firm_above_1_odd": int(sum(r["sensitivity"]["frac_above_1_odd"] >= 0.975 for r in rows)),
                                   "n_firm_below_1_odd": int(sum(r["sensitivity"]["frac_above_1_odd"] <= 0.025 for r in rows)),
                                   "n_firm_above_1_paper": int(sum(r["sensitivity"]["frac_above_1_paper"] >= 0.975 for r in rows)),
                                   "n_firm_below_1_paper": int(sum(r["sensitivity"]["frac_above_1_paper"] <= 0.025 for r in rows))}}
        c = summary["contributions_mean"]; res = c["residual_after_saturation"]
        b1, b2 = c["bias_then_disp"]["bias"], c["disp_then_bias"]["bias"]
        summary["bias_share_of_residual"] = {"bias_then_disp": b1 / res, "disp_then_bias": b2 / res,
                                             "order_robust": bool(abs(b1 - b2) < 0.25 * abs(res))}
        out["models"][model] = {"rows": rows, "summary": summary}
        m = summary["means"]
        print(f"== {model}, {len(rows)} formulations at {e} peers: eJ {m['eJ']:.3f} -> add {m['F0_add']:.3f} -> odd {m['F0_odd']:.3f} "
              f"-> even {m['F0_even']:.3f} -> paper {m['F0_paper']:.3f} (add_bias {m['F0_add_bias']:.3f}, disp {m['F0_disp']:.3f})")
        print(f"   steps: saturation {c['saturation']:+.3f}, shape {c['shape_k']:+.3f}; bias {b1:+.3f}/{b2:+.3f}, "
              f"dispersion {c['bias_then_disp']['dispersion']:+.3f}/{c['disp_then_bias']['dispersion']:+.3f}, "
              f"interaction {c['interaction_bias_x_disp']:+.3f}; bias share {b1 / res:.2f}/{b2 / res:.2f} "
              f"({'order-robust' if summary['bias_share_of_residual']['order_robust'] else 'order-dependent'})")
        for k in keys:
            q = summary["counts"][k]
            print(f"   {k:12s} mean {q['mean']:.3f}: above {q['above_1']}, within {MARGIN} {q['within_margin']}, below {q['below_1']} (point count above 1: {q['above_1_point']})")
        s = summary["sensitivity"]
        print(f"   sensitivity s={s['s']}: max std {s['std_F0_max']:.3f}; odd firm above/below {s['n_firm_above_1_odd']}/{s['n_firm_below_1_odd']}; paper firm above/below {s['n_firm_above_1_paper']}/{s['n_firm_below_1_paper']}")
    res_dir = Path(a.results)
    write_json(res_dir / "F0_decomposition_e4.json", {"manifest": build_manifest(vars(a), []), **out})
    with open(res_dir / "F0_decomposition_e4.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys())); wr.writeheader(); wr.writerows(csv_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
