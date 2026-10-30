#!/usr/bin/env python
"""Shape of the response to exposure (the results notes (not included), protocol
written before the data, 2026-09-12).

For the list channel with no memory, all tabulated exposures (2, 4, 7 in
results/phase_a, 10-30 in the extended cells) and all formulations:
    * unanimity jump  gap(+e) - gap(-e)  from the mean sector-A gap per k;
    * ratio jump(e) / jump(e_ref) per formulation, bootstrap over
      formulations, against the predictions of the three hypotheses
      (fraction: 1; count: e / e_ref; sub-linear: in between);
    * collapse test: mean gap against the fraction f = k / e for all
      exposures, with the model comparison
          M_frac  gap = a + b f + c u
          M_count gap = a + b k + c u
          M_both  gap = a + b f + d k + c u
      (u = sign(k) at unanimity, 0 otherwise) per formulation, AIC and R^2;
    * gap-based estimator J_gap and z_bar * J_gap per exposure.
Output: results/phase_a/exposure_shape.json and a text summary. The
decision rule of §21 is applied to the ratio at the largest exposure
available against e_ref = 7.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402


def _boot(d: np.ndarray, rng: np.random.Generator, n: int = 4000) -> tuple[float, float, float]:
    b = [np.mean(rng.choice(d, size=d.size, replace=True)) for _ in range(n)]
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(((y - X @ coef) ** 2).sum()); n = len(y)
    aic = n * np.log(max(rss, 1e-12) / n) + 2 * X.shape[1]
    tss = float(((y - y.mean()) ** 2).sum())
    return coef, (1 - rss / tss if tss > 0 else float("nan")), aic


def load_mean_gaps(results: Path, model: str) -> dict[tuple, dict[int, float]]:
    """{(wording, format, e): {k: mean gap}} for list, k_mem = 0 cells."""
    out: dict[tuple, dict[int, float]] = {}
    for path in glob.glob(str(results / f"A_{model}_list_k0_e*_Tan_f0_all_N*/gaps.json")):
        m = re.search(r"_e(\d+)_", Path(path).parent.name)
        e = int(m.group(1))
        for tab in json.load(open(path, encoding="utf-8"))["tables"]:
            byk: dict[int, list[float]] = {}
            for x in tab["entries"]:
                byk.setdefault(x["k"], []).append(x["gap"])
            out[(tab["wording"], tab["format"], e)] = {k: float(np.mean(v)) for k, v in byk.items()}
    return out


def analyse(model: str, G: dict, e_ref: int, rng: np.random.Generator) -> dict:
    exposures = sorted({e for (_, _, e) in G})
    forms = sorted({(w, f) for (w, f, e) in G if e == e_ref})
    res: dict = {"exposures": exposures, "e_ref": e_ref, "n_formulations": len(forms), "jump": {}, "ratio": {},
                 "J_gap": {}, "zJ_gap": {}, "collapse": {}}
    for e in exposures:
        keys = [(w, f) for (w, f) in forms if (w, f, e) in G]
        jumps = np.array([G[(w, f, e)][e] - G[(w, f, e)][-e] for (w, f) in keys])
        ks = np.arange(-e, e + 1, 2, dtype=float)
        Js = []
        for (w, f) in keys:
            y = np.array([G[(w, f, e)][int(k)] for k in ks])
            Js.append(_fit(np.c_[np.ones_like(ks), ks], y)[0][1] / 2)
        res["jump"][str(e)] = {"mean": float(jumps.mean()), "sd": float(jumps.std(ddof=1)) if jumps.size > 1 else None,
                               "by_formulation": jumps.tolist()}
        res["J_gap"][str(e)] = float(np.mean(Js)); res["zJ_gap"][str(e)] = float(e * np.mean(Js))
        if e != e_ref:
            r = np.array([(G[(w, f, e)][e] - G[(w, f, e)][-e]) / (G[(w, f, e_ref)][e_ref] - G[(w, f, e_ref)][-e_ref])
                          for (w, f) in keys if (G[(w, f, e_ref)][e_ref] - G[(w, f, e_ref)][-e_ref]) > 0.5])
            if r.size:
                m_, lo, hi = _boot(r, rng)
                res["ratio"][str(e)] = {"mean": m_, "ci95": [lo, hi], "n": int(r.size),
                                        "pred_fraction": 1.0, "pred_count": e / e_ref,
                                        "by_formulation": r.tolist()}
    # power-law exponent of the jump for e >= e_ref, per formulation (0 = fraction, 1 = count),
    # bootstrap over formulations; formulations with a jump below 0.5 logit at any e are excluded
    big = [e for e in exposures if e >= e_ref]
    if len(big) >= 3:
        expo = []
        for (w, f) in forms:
            ys = [G[(w, f, e)][e] - G[(w, f, e)][-e] for e in big if (w, f, e) in G]
            if len(ys) == len(big) and min(ys) > 0.5:
                expo.append(float(np.polyfit(np.log(big), np.log(ys), 1)[0]))
        if expo:
            m_, lo, hi = _boot(np.array(expo), rng)
            res["exponent_e_ge_ref"] = {"mean": m_, "ci95": [lo, hi], "n": len(expo), "by_formulation": expo,
                                        "min": float(min(expo)), "max": float(max(expo))}
    # collapse test over all exposures per formulation
    wins, dd, bb, r2 = 0, [], [], {"frac": [], "count": [], "both": []}
    for (w, f) in forms:
        pts = [(e, k, G[(w, f, e)][k]) for e in exposures if (w, f, e) in G for k in sorted(G[(w, f, e)])]
        e_ = np.array([p[0] for p in pts], float); k_ = np.array([p[1] for p in pts], float); y = np.array([p[2] for p in pts])
        fr = k_ / e_; u = np.sign(k_) * (np.abs(k_) == e_); one = np.ones_like(y)
        cf, r2f, af = _fit(np.c_[one, fr, u], y); cc, r2c, ac = _fit(np.c_[one, k_, u], y); cb, r2b, ab = _fit(np.c_[one, fr, k_, u], y)
        wins += af < ac; dd.append(cb[2]); bb.append(cb[1]); r2["frac"].append(r2f); r2["count"].append(r2c); r2["both"].append(r2b)
    md, ld, hd = _boot(np.array(dd), rng); mb, lb, hb = _boot(np.array(bb), rng)
    res["collapse"] = {"frac_beats_count_aic": f"{wins}/{len(forms)}",
                       "r2_mean": {k: float(np.mean(v)) for k, v in r2.items()},
                       "coef_fraction_in_both": {"mean": mb, "ci95": [lb, hb]},
                       "coef_count_in_both": {"mean": md, "ci95": [ld, hd]}}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="qwen0.6b,qwen1.7b")
    ap.add_argument("--e-ref", type=int, default=7)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(5)
    out = {"models": {}}
    for model in a.models.split(","):
        G = load_mean_gaps(Path(a.results), model)
        r = analyse(model, G, a.e_ref, rng)
        out["models"][model] = r
        print(f"== {model}: exposures {r['exposures']}, {r['n_formulations']} formulations, e_ref = {a.e_ref}")
        for e in r["exposures"]:
            j = r["jump"][str(e)]
            line = f"  e={e:2d}: jump {j['mean']:+6.2f} (sd {j['sd']:.2f})  J_gap {r['J_gap'][str(e)]:+.2f}  zJ_gap {r['zJ_gap'][str(e)]:.2f}"
            if str(e) in r["ratio"]:
                q = r["ratio"][str(e)]
                line += f"  | jump(e)/jump({a.e_ref}) = {q['mean']:.2f} [{q['ci95'][0]:.2f}, {q['ci95'][1]:.2f}]  (fraction 1.00, count {q['pred_count']:.2f})"
            print(line)
        if "exponent_e_ge_ref" in r:
            x = r["exponent_e_ge_ref"]
            print(f"  exponent (e >= {a.e_ref}): {x['mean']:.2f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}] n={x['n']} min {x['min']:.2f} max {x['max']:.2f}")
        c = r["collapse"]
        print(f"  collapse: frac beats count (AIC) {c['frac_beats_count_aic']}; R2 frac {c['r2_mean']['frac']:.3f} count {c['r2_mean']['count']:.3f} both {c['r2_mean']['both']:.3f}; "
              f"in M_both coef(fraction) {c['coef_fraction_in_both']['mean']:+.2f} [{c['coef_fraction_in_both']['ci95'][0]:+.2f}, {c['coef_fraction_in_both']['ci95'][1]:+.2f}], "
              f"coef(count) {c['coef_count_in_both']['mean']:+.3f} [{c['coef_count_in_both']['ci95'][0]:+.3f}, {c['coef_count_in_both']['ci95'][1]:+.3f}]")
    write_json(Path(a.results) / "exposure_shape.json", {"manifest": build_manifest(vars(a), [5]), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
