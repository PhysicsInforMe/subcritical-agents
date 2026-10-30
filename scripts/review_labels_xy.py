#!/usr/bin/env python
"""Token-identity test of the label bias (declared 2026-09-30 in the project log (not included),
correction row of 2026-09-30 in the results notes (not included)).

The cells *_allxy_N8 (five original wordings), *_allxyba_N8 (same, labels
named in the reverse order), *_allnwxy_N8 and *_allnwxyba_N8 (five new
wordings, both orders) repeat the 4-peer, k = 0 tabulation with the label
tokens X and Y in place of A and B (YAML key ``labels: [X, Y]``; X is the +
label). Per formulation the bias h toward X is the intercept over two of the
regression of the mean gap on k. For every wording there is a version that
names X first and one that names Y first:
  X first: allxy (wordings 0-4), allnwxy (6, 8), allnwxyba (5, 7, 9)
  Y first: allxyba (0-4), allnwxy (5, 7, 9), allnwxyba (6, 8)
Quantities per wording (mean over the two formats): d = h(X first) - h(Y first)
and m = (h(X first) + h(Y first)) / 2, each with a t interval on the ten
per-wording means (and on the five original and the five new alone).

Declared rule, per model, on the ten wordings:
  order        d excludes zero (positive) and m includes zero;
  token        m excludes zero (a preference for one token whatever the order);
  A-specific   d and m both include zero and |mean h| below 0.5 (one logit of
               bias, since gap = 2(h + J k)): the
               bias measured with A/B does not appear with X/Y.
The comparison with A/B (h and J~ of the same wordings) is descriptive.
Output: results/phase_a/review_labels_xy.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.review_r2_runs_analysis import fit_k, tables  # noqa: E402
from src.common import build_manifest, write_json  # noqa: E402

R = Path("results/phase_a")
NEW_B_FIRST = {0, 2, 4}   # wordings 5, 7, 9 of the new set name the second label first in the base file


def t_int(values: dict) -> dict:
    m = np.array(list(values.values()), float); n = m.size
    tq = float(stats.t.ppf(0.975, n - 1)); se = m.std(ddof=1) / np.sqrt(n)
    ci = [float(m.mean() - tq * se), float(m.mean() + tq * se)]
    return {"mean": float(m.mean()), "ci95": ci, "n_wordings": int(n), "n_positive": int((m > 0).sum()),
            "excludes_zero": bool(ci[0] > 0 or ci[1] < 0), "per_wording": {str(k): float(v) for k, v in values.items()}}


def fits(cell: str) -> dict[tuple, tuple[float, float]]:
    return {key: fit_k(t["entries"]) for key, t in tables(cell).items()}


def by_order(m: str, base: str, swapped: str, second_first: set[int], offset: int) -> tuple[dict, dict, dict, dict]:
    fb, fs = fits(f"A_{m}_list_k0_e4_Tan_f0_{base}_N8"), fits(f"A_{m}_list_k0_e4_Tan_f0_{swapped}_N8")
    hX, hY, JX, JY = {}, {}, {}, {}
    for key in sorted(set(fb) & set(fs)):
        w, f = key; lab = (w + offset, f)
        (Jb, hb), (Js, hs) = fb[key], fs[key]
        if w in second_first:
            hX[lab], hY[lab], JX[lab], JY[lab] = hs, hb, Js, Jb
        else:
            hX[lab], hY[lab], JX[lab], JY[lab] = hb, hs, Jb, Js
    return hX, hY, JX, JY


def summarise(hX: dict, hY: dict, JX: dict, JY: dict) -> dict:
    rows = [{"wording": k[0], "format": k[1], "h_X_first": hX[k], "h_Y_first": hY[k], "J_X_first": JX[k], "J_Y_first": JY[k]} for k in sorted(hX)]
    d_w: dict[int, list[float]] = {}; m_w: dict[int, list[float]] = {}
    for r in rows:
        d_w.setdefault(r["wording"], []).append(r["h_X_first"] - r["h_Y_first"])
        m_w.setdefault(r["wording"], []).append(0.5 * (r["h_X_first"] + r["h_Y_first"]))
    d = t_int({w: float(np.mean(v)) for w, v in d_w.items()}); mm = t_int({w: float(np.mean(v)) for w, v in m_w.items()})
    mean_h = float(np.mean([r["h_X_first"] for r in rows] + [r["h_Y_first"] for r in rows]))
    verdict = []
    if d["excludes_zero"] and d["mean"] > 0 and not mm["excludes_zero"]:
        verdict.append("order")
    if mm["excludes_zero"]:
        verdict.append("token")
    if not d["excludes_zero"] and not mm["excludes_zero"] and abs(mean_h) < 0.5:
        verdict.append("A-specific")
    return {"rows": rows, "d_t": d, "m_t": mm, "h_mean_X_first": float(np.mean([r["h_X_first"] for r in rows])),
            "h_mean_Y_first": float(np.mean([r["h_Y_first"] for r in rows])), "h_mean_both_orders": mean_h,
            "J_mean_X_first": float(np.mean([r["J_X_first"] for r in rows])), "J_mean_Y_first": float(np.mean([r["J_Y_first"] for r in rows])),
            "n_toward_X_when_X_first": int(sum(r["h_X_first"] > 0 for r in rows)), "n_toward_X_when_Y_first": int(sum(r["h_Y_first"] > 0 for r in rows)),
            "n_formulations": len(rows), "verdict": verdict or ["undetermined"]}


def main() -> int:
    out: dict = {"models": {}}
    for m in ("qwen0.6b", "qwen1.7b"):
        needed = [f"A_{m}_list_k0_e4_Tan_f0_{s}_N8" for s in ("allxy", "allxyba", "allnwxy", "allnwxyba")]
        if not all((R / c / "gaps.json").exists() for c in needed):
            print(f"missing cells for {m}"); continue
        orig = by_order(m, "allxy", "allxyba", set(), 0)
        new = by_order(m, "allnwxy", "allnwxyba", NEW_B_FIRST, 5)
        ten = tuple({**a, **b} for a, b in zip(orig, new))
        res = {"original": summarise(*orig), "new": summarise(*new), "ten": summarise(*ten)}
        # descriptive comparison with the A/B tabulation of the same wordings (h toward A, J~)
        ab = {}
        for grp, base, swapped, sf, off in (("original", "all", "allba", set(), 0), ("new", "allnw", "allnwba", NEW_B_FIRST, 5)):
            hA, hB, JA, JB = by_order(m, base, swapped, sf, off)
            ab[grp] = {"h_mean_first": float(np.mean(list(hA.values()))), "h_mean_second": float(np.mean(list(hB.values()))),
                       "J_mean_first": float(np.mean(list(JA.values()))), "J_mean_second": float(np.mean(list(JB.values())))}
        res["AB_reference"] = ab
        out["models"][m] = res
        for g in ("original", "new", "ten"):
            s = res[g]
            print(f"{m} {g:8s}: h X-first {s['h_mean_X_first']:+.2f} (toward X {s['n_toward_X_when_X_first']}/{s['n_formulations']}), "
                  f"Y-first {s['h_mean_Y_first']:+.2f} (toward X {s['n_toward_X_when_Y_first']}/{s['n_formulations']}); "
                  f"d {s['d_t']['mean']:+.2f} {[round(x, 2) for x in s['d_t']['ci95']]}; m {s['m_t']['mean']:+.2f} {[round(x, 2) for x in s['m_t']['ci95']]}; "
                  f"J X-first {s['J_mean_X_first']:.2f} Y-first {s['J_mean_Y_first']:.2f}; verdict {s['verdict']}")
        print(f"{m} A/B reference: {json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in ab.items()})}")
    write_json(R / "review_labels_xy.json", {"manifest": build_manifest({"script": "review_labels_xy"}, []), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
