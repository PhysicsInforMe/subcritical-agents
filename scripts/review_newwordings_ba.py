#!/usr/bin/env python
"""Order of mention on the five new wordings (correction of 2026-09-30,
the results notes (not included); hypothesis and rule declared in the project log (not included) before
the data).

The share of the label bias due to the order of mention (§14.1 B4, R2-06)
was measured on the five original wordings with the two labels named in the
order B then A (*_allba_N8 against *_all_N8). Three of the five new wordings
(5, 7, 9) name B first and keep a bias toward A, so the cells *_allnwba_N8
tabulate the new wordings with the naming order swapped inside each wording
(5, 7, 9 now A first; 6, 8 now B first), 4 peers, k = 0, both models.
Paired against *_allnw_N8 every new wording has both orders.

Per formulation (wording, format): h_gap = intercept over two of the
regression of the mean gap on k (gap = 2(h + J k)), in the A-first and in
the B-first version. Quantity: d = h(A-first) - h(B-first), mean over the
two formats per wording, t interval on the five per-wording means (the
wording is the independent unit, §3 of the paper), and the share
d / h(A-first) on the means. The same quantities on the five original
wordings (from *_all_N8 and *_allba_N8) and pooled on the ten wordings.

Declared rule: order of mention is a general effect on the new wordings if
d > 0 in at least 4 of 5 wordings per model and the t interval excludes
zero; otherwise the share is a property of the original structure and the
paper restricts the sentence to the five original wordings.
Output: results/phase_a/review_newwordings_ba.json.
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
NEW_B_FIRST = {0, 2, 4}   # wordings 5, 7, 9 of the new set name B first in *_allnw_N8


def t_int(values: dict) -> dict:
    m = np.array(list(values.values()), float); n = m.size
    tq = float(stats.t.ppf(0.975, n - 1)); se = m.std(ddof=1) / np.sqrt(n)
    ci = [float(m.mean() - tq * se), float(m.mean() + tq * se)]
    return {"mean": float(m.mean()), "ci95": ci, "n_wordings": int(n), "n_positive": int((m > 0).sum()),
            "excludes_zero": bool(ci[0] > 0 or ci[1] < 0), "per_wording": {str(k): float(v) for k, v in values.items()}}


def h_by_order(cell_a: str, cell_b: str, b_first_in_a: set[int], offset: int) -> tuple[dict, dict]:
    """h per (wording, format) in the A-first and B-first versions.
    cell_a is the base cell, cell_b the swapped one; wordings in b_first_in_a
    name B first in cell_a (so their A-first version is in cell_b)."""
    ta, tb = tables(cell_a), tables(cell_b)
    hA, hB = {}, {}
    for key in sorted(set(ta) & set(tb)):
        w, f = key
        _, ha = fit_k(ta[key]["entries"]); _, hb = fit_k(tb[key]["entries"])
        lab = (w + offset, f)
        if w in b_first_in_a:
            hA[lab], hB[lab] = hb, ha
        else:
            hA[lab], hB[lab] = ha, hb
    return hA, hB


def summarise(hA: dict, hB: dict) -> dict:
    rows = [{"wording": k[0], "format": k[1], "h_A_first": hA[k], "h_B_first": hB[k], "d": hA[k] - hB[k]} for k in sorted(hA)]
    per_w: dict[int, list[float]] = {}
    hA_w: dict[int, list[float]] = {}
    for r in rows:
        per_w.setdefault(r["wording"], []).append(r["d"]); hA_w.setdefault(r["wording"], []).append(r["h_A_first"])
    d_w = {w: float(np.mean(v)) for w, v in per_w.items()}
    ti = t_int(d_w)
    mean_hA = float(np.mean([r["h_A_first"] for r in rows])); mean_hB = float(np.mean([r["h_B_first"] for r in rows]))
    return {"rows": rows, "d_t": ti, "h_mean_A_first": mean_hA, "h_mean_B_first": mean_hB,
            "share": float(ti["mean"] / mean_hA) if mean_hA else None,
            "n_toward_A_when_B_first": int(sum(r["h_B_first"] > 0 for r in rows)), "n_formulations": len(rows),
            "n_wordings_positive": ti["n_positive"], "n_wordings": ti["n_wordings"],
            "rule_general": bool(ti["n_positive"] >= 4 and ti["excludes_zero"])}


def main() -> int:
    out: dict = {"models": {}}
    for m in ("qwen0.6b", "qwen1.7b"):
        new_cell, new_ba = f"A_{m}_list_k0_e4_Tan_f0_allnw_N8", f"A_{m}_list_k0_e4_Tan_f0_allnwba_N8"
        if not (R / new_ba / "gaps.json").exists():
            print(f"missing {new_ba}"); continue
        hA_new, hB_new = h_by_order(new_cell, new_ba, NEW_B_FIRST, offset=5)
        hA_or, hB_or = h_by_order(f"A_{m}_list_k0_e4_Tan_f0_all_N8", f"A_{m}_list_k0_e4_Tan_f0_allba_N8", set(), offset=0)
        res = {"new": summarise(hA_new, hB_new), "original": summarise(hA_or, hB_or),
               "ten": summarise({**hA_or, **hA_new}, {**hB_or, **hB_new})}
        out["models"][m] = res
        for g in ("original", "new", "ten"):
            s = res[g]; t = s["d_t"]
            print(f"{m} {g:8s}: h A-first {s['h_mean_A_first']:+.2f}, B-first {s['h_mean_B_first']:+.2f}; d {t['mean']:+.2f} {[round(x, 2) for x in t['ci95']]} "
                  f"positive in {t['n_positive']}/{t['n_wordings']} wordings; share {s['share']:.2f}; toward A when B first {s['n_toward_A_when_B_first']}/{s['n_formulations']}"
                  + (f"; rule general: {s['rule_general']}" if g == "new" else ""))
    write_json(R / "review_newwordings_ba.json", {"manifest": build_manifest({"script": "review_newwordings_ba"}, []), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
