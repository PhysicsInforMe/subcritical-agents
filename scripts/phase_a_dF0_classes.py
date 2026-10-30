#!/usr/bin/env python
"""Interval-based mean-field classification of the formulations (claims.md
B4b), declared on 2026-09-22 (the project log (not included)) to replace the ad hoc band
"within 0.05 of 1" of §24 with a standard statistical criterion.

For each model and exposure, per formulation: F'(0) of the symmetrised
mean-field map (stats of phase_a_orderings_check) with a bootstrap over the
orderings inside each k (the sampling error of the tabulation). Classes:
    ordered      lower bound of the interval above 1 (disordered state unstable)
    disordered   upper bound below 1
    undetermined interval contains 1
Cells: 2, 4, 7 peers from the 8-ordering tables (A_<m>_list_k0_e<e>_Tan_f0_all_N8),
10 and 30 peers from the 32-ordering tables (..._allo32_N32); optionally
the 32-ordering tables at 4 and 7 peers when present (..._allo32_N8).
Output: results/phase_a/dF0_classes.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_orderings_check import stats  # noqa: E402
from src.common import build_manifest, write_json  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="qwen0.6b,qwen1.7b")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--out", default="results/phase_a/dF0_classes.json")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(40)
    out = {"models": {}}
    for model in a.models.split(","):
        out["models"][model] = {}
        cells = [(2, f"A_{model}_list_k0_e2_Tan_f0_all_N8", 8), (4, f"A_{model}_list_k0_e4_Tan_f0_all_N8", 8),
                 (7, f"A_{model}_list_k0_e7_Tan_f0_all_N8", 8), (10, f"A_{model}_list_k0_e10_Tan_f0_allo32_N32", 32),
                 (30, f"A_{model}_list_k0_e30_Tan_f0_allo32_N32", 32),
                 (4, f"A_{model}_list_k0_e4_Tan_f0_allo32_N8", 32), (7, f"A_{model}_list_k0_e7_Tan_f0_allo32_N8", 32)]
        for e, cell, n_ord in cells:
            p = Path(a.results) / cell / "gaps.json"
            if not p.exists():
                continue
            rows = []
            for t in json.load(open(p, encoding="utf-8"))["tables"]:
                s = stats(t["entries"], e)
                boots = [stats(t["entries"], e, rng, sampled_only=True)["dF0_sym"] for _ in range(a.n_boot)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                cls = "ordered" if lo > 1 else ("disordered" if hi < 1 else "undetermined")
                if lo == hi:
                    cls += "_exact"   # every ordering enumerated: no sampling error at all
                rows.append({"wording": t["wording"], "format": t["format"], "dF0": s["dF0_sym"],
                             "ci_orderings": [float(lo), float(hi)], "class": cls, "zJ_gap": s["zJ_gap"]})
            counts = {c: sum(r["class"].startswith(c) for r in rows) for c in ("ordered", "disordered", "undetermined")}
            counts["exact"] = sum(r["class"].endswith("_exact") for r in rows)
            key = f"e{e}_ord{n_ord}"
            out["models"][model][key] = {"exposure": e, "n_orderings": n_ord, "cell": cell, "rows": rows, "counts": counts,
                                         "dF0_mean": float(np.mean([r["dF0"] for r in rows]))}
            print(f"== {model} e={e} ({n_ord} orderings): ordered {counts['ordered']}, disordered {counts['disordered']}, "
                  f"undetermined {counts['undetermined']} of {len(rows)}; mean F'(0) {np.mean([r['dF0'] for r in rows]):.2f}; "
                  f"mean CI half-width {np.mean([(r['ci_orderings'][1]-r['ci_orderings'][0])/2 for r in rows]):.2f}")
    write_json(Path(a.out), {"manifest": build_manifest(vars(a), [40]), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
