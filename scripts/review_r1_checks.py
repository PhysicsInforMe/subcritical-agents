#!/usr/bin/env python
"""Zero-cost analyses answering the schede of revision round, R1 (the revision round (not included)),
all from existing result files. Output: results/phase_a/review_r1.json.

(a) R1-01  Combinatorial ceiling of the mean-field slope: with k ~ 2 Bin(e, 1/2) - e
           at m = 0, F'(0) = 2 E0[k P(+|k)] <= E0|k|, attained by deterministic
           majority rule. Reported per exposure with the ratio F'(0)/E0|k| per
           formulation (dF0_classes.json) and the "marginal" flag (within 0.01
           of the ceiling).
(b) R1-02  Classes from the labelled sector A (meanfield_stability.json, 8
           orderings): bistable (two stable fixed points), field-dominated (one
           stable fixed point with |m*| > 0.9), disordered (one with |m*| <= 0.9).
(c) R1-03  Consensus time in rounds against the voter reference in rounds on the
           complete graph, N ln 2 (voter.py: N^2 single updates = N rounds), per
           N and per formulation (lattice_control.json size_scan rows).
(d) R1-04  Combined linear gain with memory, e J + k J_mem, per memory cell
           (memory_curve.json at 4 peers; aggregate.json at k = 3, e = 2, 4, 7).
(e) R1-06  Paired differences of z_bar J_gap between 7 and 15 peers and between
           7 and 30 peers (8-ordering cells), cluster bootstrap over wordings.
(g) R1-05  Observed tipping fraction z_c per branch and formulation from the
           hysteresis sweep at T = 1 (measures.json by_T), resolution one agent.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_format_check import boot_mean, j_gap, load_mean_gaps  # noqa: E402
from src.common import build_manifest, write_json  # noqa: E402

R = Path("results/phase_a")


def ceiling(e: int) -> float:
    """E0|k| for k = 2 Bin(e, 1/2) - e."""
    return sum(math.comb(e, n) / 2 ** e * abs(2 * n - e) for n in range(e + 1))


def main() -> int:
    out: dict = {}
    # (a) ceiling and ratio
    classes = json.load(open(R / "dF0_classes.json", encoding="utf-8"))["models"]
    out["ceiling"] = {str(e): ceiling(e) for e in (2, 4, 7, 10, 15, 20, 30)}
    out["ratio"] = {}
    for model, blocks in classes.items():
        out["ratio"][model] = {}
        for key, b in blocks.items():
            e = b["exposure"]; c = ceiling(e)
            ratios = [r["dF0"] / c for r in b["rows"]]
            marginal = sum(abs(r["dF0"] - c) <= 0.01 for r in b["rows"])
            out["ratio"][model][key] = {"exposure": e, "n_orderings": b["n_orderings"], "ceiling": c,
                                        "ratio_mean": float(np.mean(ratios)), "ratio_min": float(min(ratios)),
                                        "ratio_max": float(max(ratios)), "n_marginal": int(marginal),
                                        "per_formulation": [{"wording": r["wording"], "format": r["format"], "dF0": r["dF0"],
                                                             "ratio": r["dF0"] / c} for r in b["rows"]]}
            print(f"(a) {model} e={e} ({b['n_orderings']} ord): ceiling {c:.3f}, F'(0)/ceiling mean {np.mean(ratios):.2f} "
                  f"[{min(ratios):.2f}, {max(ratios):.2f}], marginal {marginal}/{len(ratios)}")
    # (b) sector-A classes
    st = json.load(open(R / "meanfield_stability.json", encoding="utf-8"))["models"]
    out["sector_A"] = {}
    for model, rows in st.items():
        out["sector_A"][model] = {}
        for e in sorted({r["exposure"] for r in rows}):
            sub = [r for r in rows if r["exposure"] == e]
            cls = []
            for r in sub:
                if r["n_stable_A"] >= 2:
                    c = "bistable"
                elif max(abs(m) for m in r["stable_m_A"]) > 0.9:
                    c = "field_dominated"
                else:
                    c = "disordered"
                cls.append({"wording": r["wording"], "format": r["format"], "class": c, "n_stable_A": r["n_stable_A"],
                            "stable_m_A": r["stable_m_A"], "dF0_sym": r["dF0_sym"]})
            counts = {c: sum(x["class"] == c for x in cls) for c in ("bistable", "field_dominated", "disordered")}
            out["sector_A"][model][str(e)] = {"counts": counts, "rows": cls}
            print(f"(b) {model} e={e}: bistable {counts['bistable']}, field-dominated {counts['field_dominated']}, disordered {counts['disordered']}")
    # (c) consensus time against the voter reference in rounds
    lat = json.load(open(R / "lattice_control.json", encoding="utf-8"))["models"]
    out["consensus_time"] = {}
    for model, m in lat.items():
        out["consensus_time"][model] = {}
        for topo in ("all", "lattice"):
            blk = m["size_scan"][topo]
            per_N = {}
            for key, v in blk.items():
                if not isinstance(v, dict):
                    continue
                N = v["N"]; ref = N * math.log(2)
                times = [r["t_cons_mean"] for r in v["rows"] if r.get("t_cons_mean") is not None]
                per_N[str(N)] = {"voter_rounds_ref": ref, "t_cons_mean": v.get("t_cons_mean"),
                                 "t_cons_median_over_formulations": float(np.median(times)) if times else None,
                                 "n_formulations_below_voter": int(sum(t < ref for t in times)), "n_formulations": len(times),
                                 "frac_consensus": v["frac_consensus"]}
            out["consensus_time"][model][topo] = per_N
            print(f"(c) {model} {topo}: " + "; ".join(f"N={N}: t={d['t_cons_mean']:.1f} (median {d['t_cons_median_over_formulations']:.1f}) vs voter {d['voter_rounds_ref']:.1f}, below in {d['n_formulations_below_voter']}/{d['n_formulations']}"
                                                   for N, d in sorted(per_N.items(), key=lambda q: int(q[0])) if d["t_cons_mean"] is not None))
    # (d) combined gain with memory
    mc = json.load(open(R / "memory_curve.json", encoding="utf-8"))
    e_mc = mc["exposure"]
    out["memory_gain"] = {}
    for model, m in mc["models"].items():
        g = {}
        for k in (0, 1, 2, 3):
            J = m["J_gap_mean"].get(str(k)); Jm = m["J_mem_gap_mean"].get(str(k), 0.0) if k else 0.0
            if J is not None:
                g[str(k)] = {"eJ": e_mc * J, "kJmem": k * Jm, "gain": e_mc * J + k * Jm}
        out["memory_gain"][model] = {"exposure": e_mc, "by_k": g}
        print(f"(d) {model} e={e_mc}: gain eJ + kJ_mem by k: " + ", ".join(f"k{k}: {v['gain']:.2f} (eJ {v['eJ']:.2f} + {v['kJmem']:.2f})" for k, v in g.items()))
    agg = json.load(open(R / "aggregate.json", encoding="utf-8"))
    out["memory_gain_grid"] = {}
    for model in ("qwen0.6b", "qwen1.7b"):
        out["memory_gain_grid"][model] = {}
        for e in (2, 4, 7):
            sub = [r for r in agg["rows"] if r["model"] == model and r["T"] == 1.0 and r["topology"] == "all" and r["channel"] == "list" and r["exposure"] == e]
            k0 = [r["J_gap"] for r in sub if r["k_mem"] == 0]; k3 = [r for r in sub if r["k_mem"] == 3]
            if k0 and k3:
                gain0 = e * float(np.mean(k0)); gain3 = e * float(np.mean([r["J_gap"] for r in k3])) + 3 * float(np.mean([r["J_mem_gap"] for r in k3]))
                out["memory_gain_grid"][model][str(e)] = {"gain_k0": gain0, "gain_k3": gain3}
                print(f"(d) grid {model} e={e}: gain k0 {gain0:.2f} -> k3 {gain3:.2f}")
    # (e) paired differences of z_bar J between 7 and 15, 7 and 30 peers
    rng = np.random.default_rng(31)
    out["zJ_paired"] = {}
    for model in ("qwen0.6b", "qwen1.7b"):
        G = load_mean_gaps(R, model)
        forms = sorted({(w, f) for (w, f, e) in G if f in ("comma", "lines")})
        res = {}
        for e1 in (15, 30):
            d = np.array([e1 * j_gap(G[(w, f, e1)]) - 7 * j_gap(G[(w, f, 7)]) for (w, f) in forms])
            res[f"e{e1}-e7"] = boot_mean(d, np.array([w for (w, _) in forms]), rng)
        out["zJ_paired"][model] = res
        print(f"(e) {model}: " + "; ".join(f"{k}: {v['mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" for k, v in res.items()))
    # (g) tipping fraction from the hysteresis branches at T = 1, 4 peers
    out["tipping"] = {}
    for model in ("qwen0.6b", "qwen1.7b"):
        p = R / f"A_{model}_list_k0_e4_Tan_f0_all_N8" / "measures.json"
        meas = json.load(open(p, encoding="utf-8"))
        rows = []
        for f in meas["formulations"]:
            h = f["by_T"]["1.0"]["hysteresis"]; z = h["z"]; up = h["up"]; down = h["down"]
            zc_up = next((zz for zz, mm in zip(z, up) if mm > 0), None)          # first z on the way up with m > 0
            zc_down = next((zz for zz, mm in zip(z[::-1], down[::-1]) if mm < 0), None)  # first z on the way down with m < 0
            rows.append({"wording": f["wording"], "format": f["format"], "zc_up": zc_up, "zc_down": zc_down,
                         "max_gap": h["max_gap"], "gap_at_z0": float(up[z.index(0.0)] - down[z.index(0.0)]) if 0.0 in z else None})
        out["tipping"][model] = {"z_step": float(z[1] - z[0]), "rows": rows}
        print(f"(g) {model}: z step {z[1]-z[0]:.3f}; zc_up " + str([r["zc_up"] for r in rows]) + "; zc_down " + str([r["zc_down"] for r in rows]))
    write_json(R / "review_r1.json", {"manifest": build_manifest({"script": "review_r1_checks"}, []), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
