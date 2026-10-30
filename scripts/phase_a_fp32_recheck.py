#!/usr/bin/env python
"""Full float32 recheck of the 1.7B mean-field quantities (declared on
2026-09-21, the project log (not included), after the C3 precision test): the 32-ordering cells
at 10 and 30 peers tabulated in float32 (results/phase_a_fp32check) against
the bfloat16 ones (results/phase_a), same orderings, per formulation.

Per (exposure, formulation): z_bar * J_gap, F'(0) of the symmetrised
mean-field map and R^2 (stats of phase_a_orderings_check) in both
precisions; the ordering bootstrap of the bfloat16 table gives the sampling
interval. Decision rule (log entry): the recheck passes if F'(0) in float32
lies inside the bfloat16 interval in at least 18 formulations out of 20
and the class counts (above 1.05, within 0.05 of 1, below 0.95) change by
at most one formulation per exposure; the paired difference float32 minus
bfloat16 of z_bar * J_gap is reported with a cluster bootstrap over
wordings. Output: results/phase_a/fp32_recheck.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_format_check import boot_mean  # noqa: E402
from scripts.phase_a_orderings_check import stats  # noqa: E402
from src.common import build_manifest, write_json  # noqa: E402


def classes(vals: list[float]) -> dict:
    above = sum(v > 1.05 for v in vals); crit = sum(abs(v - 1) <= 0.05 for v in vals)
    return {"above": above, "critical": crit, "below": len(vals) - above - crit}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", default="results/phase_a")
    ap.add_argument("--fp32", default="results/phase_a_fp32check")
    ap.add_argument("--model", default="qwen1.7b")
    ap.add_argument("--exposures", default="10,30")
    ap.add_argument("--n-boot", type=int, default=300)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(32)
    out = {"model": a.model, "exposures": {}}
    total_inside, total_n = 0, 0
    for e in (int(x) for x in a.exposures.split(",")):
        cell = f"A_{a.model}_list_k0_e{e}_Tan_f0_allo32_N32"
        pb, pf = Path(a.bf16) / cell / "gaps.json", Path(a.fp32) / cell / "gaps.json"
        if not (pb.exists() and pf.exists()):
            print(f"  e={e}: missing cell(s), skipped"); continue
        tb = {(t["wording"], t["format"]): t for t in json.load(open(pb, encoding="utf-8"))["tables"]}
        tf = {(t["wording"], t["format"]): t for t in json.load(open(pf, encoding="utf-8"))["tables"]}
        rows = []
        for key in sorted(set(tb) & set(tf)):
            sb, sf = stats(tb[key]["entries"], e), stats(tf[key]["entries"], e)
            boots = [stats(tb[key]["entries"], e, rng) for _ in range(a.n_boot)]
            row = {"wording": key[0], "format": key[1]}
            for q in ("zJ_gap", "dF0_sym", "r2_linear_k", "jump"):
                lo, hi = np.percentile([b[q] for b in boots], [2.5, 97.5])
                row[q] = {"bf16": sb[q], "fp32": sf[q], "ci_bf16": [float(lo), float(hi)], "inside": bool(lo <= sf[q] <= hi)}
            rows.append(row)
        n_in = sum(r["dF0_sym"]["inside"] for r in rows); total_inside += n_in; total_n += len(rows)
        d = np.array([r["zJ_gap"]["fp32"] - r["zJ_gap"]["bf16"] for r in rows])
        w = np.array([r["wording"] for r in rows])
        cls_b, cls_f = classes([r["dF0_sym"]["bf16"] for r in rows]), classes([r["dF0_sym"]["fp32"] for r in rows])
        max_change = max(abs(cls_b[k] - cls_f[k]) for k in cls_b)
        summ = {"n": len(rows), "dF0_fp32_inside_bf16_ci": f"{n_in}/{len(rows)}",
                "zJ_paired_fp32_minus_bf16": boot_mean(d, w, rng),
                "dF0_mean_bf16": float(np.mean([r["dF0_sym"]["bf16"] for r in rows])),
                "dF0_mean_fp32": float(np.mean([r["dF0_sym"]["fp32"] for r in rows])),
                "zJ_mean_bf16": float(np.mean([r["zJ_gap"]["bf16"] for r in rows])),
                "zJ_mean_fp32": float(np.mean([r["zJ_gap"]["fp32"] for r in rows])),
                "classes_bf16": cls_b, "classes_fp32": cls_f, "max_class_change": int(max_change),
                "class_rule_ok": bool(max_change <= 1)}
        out["exposures"][str(e)] = {"rows": rows, "summary": summ}
        print(f"== {a.model} e={e}: {len(rows)} formulations")
        print(f"  F'(0) fp32 inside bf16 ordering CI: {summ['dF0_fp32_inside_bf16_ci']}; mean F'(0) {summ['dF0_mean_bf16']:.2f} -> {summ['dF0_mean_fp32']:.2f}")
        print(f"  classes (above/critical/below) bf16 {cls_b} fp32 {cls_f}; max change {max_change} -> {'ok' if max_change <= 1 else 'FAIL'}")
        z = summ["zJ_paired_fp32_minus_bf16"]
        print(f"  zJ mean {summ['zJ_mean_bf16']:.2f} -> {summ['zJ_mean_fp32']:.2f}; paired diff {z['mean']:+.3f} [{z['ci95'][0]:+.3f}, {z['ci95'][1]:+.3f}]")
        for r in rows:
            f = r["dF0_sym"]
            flag = "" if (f["bf16"] > 1) == (f["fp32"] > 1) else "FLIP"
            print(f"    w{r['wording']} {r['format']:6s} F'(0) {f['bf16']:.2f} -> {f['fp32']:.2f} ci[{f['ci_bf16'][0]:.2f}, {f['ci_bf16'][1]:.2f}] {'in' if f['inside'] else 'OUT'} {flag}  zJ {r['zJ_gap']['bf16']:.2f} -> {r['zJ_gap']['fp32']:.2f}")
    out["rule"] = {"dF0_inside_total": f"{total_inside}/{total_n}", "inside_ok": bool(total_inside >= 18 and total_n == 20),
                   "classes_ok": bool(all(v["summary"]["class_rule_ok"] for v in out["exposures"].values()))}
    out["rule"]["passes"] = bool(out["rule"]["inside_ok"] and out["rule"]["classes_ok"])
    print(f"RULE: F'(0) inside {out['rule']['dF0_inside_total']} (need 18/20) -> {out['rule']['inside_ok']}; classes -> {out['rule']['classes_ok']}; PASSES: {out['rule']['passes']}")
    write_json(Path(a.bf16) / "fp32_recheck.json", {"manifest": build_manifest(vars(a), [32]), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
