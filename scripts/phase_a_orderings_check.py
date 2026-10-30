#!/usr/bin/env python
"""Test 1 of the results notes (not included) (prediction P1, written before the
data): do the conclusions that depend on the sampled orderings change
when 8 orderings per configuration become 32?

For every (model, exposure) with both an 8-ordering cell
(A_<m>_list_k0_e<e>_Tan_f0_all_N32) and a 32-ordering cell (..._allo32_N32),
per formulation:
    * z_bar * J_gap (linear summary), F'(0) of the symmetrised mean-field
      map (§24 test 4), R^2 of the linear fit in k, and the unanimity
      jump (which must be identical: same single ordering);
    * the bootstrap interval of the 8-ordering estimate (resampling gaps
      inside each k) and whether the 32-ordering value falls inside it;
    * the paired difference 32 - 8 across formulations with bootstrap.
P1 holds for a quantity if the 32-ordering value lies inside the
8-ordering bootstrap interval in most formulations and the paired
difference has an interval containing zero. Output:
results/phase_a/orderings_check.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_meanfield_stability import mf_map, policy_from_entries  # noqa: E402
from src.common import build_manifest, write_json  # noqa: E402


def stats(entries: list[dict], e: int, rng: np.random.Generator | None = None, sampled_only: bool = False) -> dict:
    """z_bar J_gap, F'(0), R^2 and unanimity jump of one table; with rng the gaps
    inside each k are resampled (ordering bootstrap). With sampled_only the
    resampling is restricted to the k whose orderings were sampled (fewer
    tabulated than the C(e, n+) distinct ones): where every ordering is
    enumerated there is no sampling error, only order sensitivity, which the
    mean-field map averages exactly (classes of B4b, 2026-09-22)."""
    import math
    byk: dict[int, list[float]] = {}
    for x in entries:
        byk.setdefault(x["k"], []).append(x["gap"])
    if rng is not None:
        def resample(k, v):
            if sampled_only and len(v) >= math.comb(e, (k + e) // 2):
                return list(v)
            return list(rng.choice(v, size=len(v), replace=True))
        byk = {k: resample(k, v) for k, v in byk.items()}
        entries = [{"k": k, "gap": g} for k, v in byk.items() for g in v]
    ks = np.array(sorted(byk), float)
    g = np.array([np.mean(byk[int(k)]) for k in ks])
    coef = np.polyfit(ks, g, 1)
    pred = np.polyval(coef, ks)
    r2 = float(1 - ((g - pred) ** 2).sum() / ((g - g.mean()) ** 2).sum())
    _, ps = policy_from_entries(entries, 1.0)
    _, dF = mf_map(ps, e)
    return {"zJ_gap": float(e * coef[0] / 2), "dF0_sym": float(dF(0.0)), "r2_linear_k": r2,
            "jump": float(g[-1] - g[0])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="qwen0.6b,qwen1.7b")
    ap.add_argument("--exposures", default="10,30")
    ap.add_argument("--n-boot", type=int, default=300)
    a = ap.parse_args(argv)
    out = {"models": {}}
    for model in a.models.split(","):
        out["models"][model] = {}
        for e in (int(x) for x in a.exposures.split(",")):
            # one generator per (model, exposure): the intervals of a cell must not
            # depend on which other cells are analysed in the same call (2026-09-22)
            rng = np.random.default_rng(8 + 1000 * e + {"qwen0.6b": 1, "qwen1.7b": 2}.get(model, 3))
            n_agents = 32 if e >= 10 else 8   # cells up to 7 peers live in the N = 8 grid
            p8 = Path(a.results) / f"A_{model}_list_k0_e{e}_Tan_f0_all_N{n_agents}" / "gaps.json"
            p32 = Path(a.results) / f"A_{model}_list_k0_e{e}_Tan_f0_allo32_N{n_agents}" / "gaps.json"
            if not (p8.exists() and p32.exists()):
                print(f"  {model} e={e}: missing cell(s), skipped")
                continue
            t8 = {(t["wording"], t["format"]): t for t in json.load(open(p8, encoding="utf-8"))["tables"]}
            t32 = {(t["wording"], t["format"]): t for t in json.load(open(p32, encoding="utf-8"))["tables"]}
            rows = []
            for key in sorted(set(t8) & set(t32)):
                s8, s32 = stats(t8[key]["entries"], e), stats(t32[key]["entries"], e)
                boots = [stats(t8[key]["entries"], e, rng) for _ in range(a.n_boot)]
                row = {"wording": key[0], "format": key[1]}
                for q in ("zJ_gap", "dF0_sym", "r2_linear_k", "jump"):
                    lo, hi = np.percentile([b[q] for b in boots], [2.5, 97.5])
                    row[q] = {"v8": s8[q], "v32": s32[q], "ci8": [float(lo), float(hi)],
                              "inside": bool(lo <= s32[q] <= hi)}
                rows.append(row)
            summary = {}
            for q in ("zJ_gap", "dF0_sym", "r2_linear_k", "jump"):
                d = np.array([r[q]["v32"] - r[q]["v8"] for r in rows])
                b = [np.mean(rng.choice(d, size=d.size, replace=True)) for _ in range(2000)]
                summary[q] = {"inside_ci8": f"{sum(r[q]['inside'] for r in rows)}/{len(rows)}",
                              "paired_diff_32_minus_8": float(d.mean()),
                              "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                              "mean8": float(np.mean([r[q]["v8"] for r in rows])),
                              "mean32": float(np.mean([r[q]["v32"] for r in rows]))}
            # classification stability of the mean-field criterion
            flips = sum((r["dF0_sym"]["v8"] > 1) != (r["dF0_sym"]["v32"] > 1) for r in rows)
            summary["dF0_class_flips"] = f"{flips}/{len(rows)}"
            out["models"][model][str(e)] = {"rows": rows, "summary": summary}
            print(f"== {model} e={e}: {len(rows)} formulations")
            for q, s in summary.items():
                if isinstance(s, dict):
                    print(f"  {q:12s}: mean 8 -> 32 = {s['mean8']:6.2f} -> {s['mean32']:6.2f}; 32 inside CI(8) in {s['inside_ci8']}; "
                          f"paired diff {s['paired_diff_32_minus_8']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]")
            print(f"  F'(0) > 1 classification flips between 8 and 32 orderings: {summary['dF0_class_flips']}")
    write_json(Path(a.results) / "orderings_check.json", {"manifest": build_manifest(vars(a), [8]), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
