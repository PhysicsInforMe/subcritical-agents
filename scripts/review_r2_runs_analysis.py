#!/usr/bin/env python
"""Analysis of the three runs proposed by revision round, R2 (declared 2026-09-23,
the project log (not included)), from the tabulated cells and the generation check.
Output: results/phase_a/review_r2_runs.json.

(1) R2-06  Order of mention. Cells *_allba_N8 (labels named B then A) against
           the original *_all_N8 (A then B), 4 peers, k = 0, both models:
           h_gap and J_gap per formulation (gap regression, intercept and
           slope over two), paired differences ba - ab with a cluster
           bootstrap over the five wordings, sign flips of h_gap.
           Declared rule: sign flip or a fall of |h| by more than half is
           order of mention; otherwise token identity.
(2) R2-05  Decisional instruction. Cell A_qwen0.6b_list_k1_e4_Tan_f0_alldec_N8
           against A_qwen0.6b_list_k1_e4_Tan_f0_all_N8: P(+ | k = -4, mem = +1)
           and P(+ | k = +4, mem = -1) per formulation at T = 1, counts of
           formulations where the own state prevails, J_gap and J_mem from the
           two-term gap regression gap = 2(h + J k + J_mem s_mem).
           Declared rule: own state prevails (P(+|k=-4,mem=+1) > 0.5) in fewer
           than 5 of 10 means recall; in 8 or more of 10 means persistence.
(3) R2-01  Generation check (results/phase_a/generation_check.json): summary
           per model against the declared rule (unparseable below 10%, mean
           absolute difference at most 0.1, same side of 1/2 in at least 28 of
           32 pairs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_format_check import boot_mean  # noqa: E402
from src.common import build_manifest, write_json  # noqa: E402

R = Path("results/phase_a")


def sig(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def tables(cell: str) -> dict[tuple, dict]:
    return {(t["wording"], t["format"]): t for t in json.load(open(R / cell / "gaps.json", encoding="utf-8"))["tables"]}


def fit_k(entries: list[dict]) -> tuple[float, float]:
    """(J, h) from the regression of the mean gap on k: gap = 2(h + J k)."""
    acc: dict[int, list[float]] = {}
    for x in entries:
        acc.setdefault(x["k"], []).append(x["gap"])
    ks = np.array(sorted(acc), float); g = np.array([np.mean(acc[int(k)]) for k in ks])
    slope, inter = np.polyfit(ks, g, 1)
    return float(slope / 2), float(inter / 2)


def fit_k_mem(entries: list[dict]) -> tuple[float, float, float]:
    """(J, J_mem, h) from gap = 2(h + J k + J_mem sum(mem)) on every entry."""
    X = np.array([[1.0, x["k"], sum(x["memory"])] for x in entries]); y = np.array([x["gap"] for x in entries])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1] / 2), float(beta[2] / 2), float(beta[0] / 2)


def t_mean(values: np.ndarray, clusters: np.ndarray) -> dict:
    """t interval on the per-wording means (the wording is the independent unit,
    §3 of the paper; added on 2026-09-30 for R4-F01 next to the cluster bootstrap
    kept for comparison): mean of the per-wording means with a Student t
    quantile on n_wordings - 1 degrees of freedom."""
    from scipy import stats
    w = sorted(set(int(c) for c in clusters))
    m = np.array([float(np.mean(values[clusters == c])) for c in w]); n = m.size
    tq = float(stats.t.ppf(0.975, n - 1)); se = m.std(ddof=1) / np.sqrt(n)
    return {"mean": float(m.mean()), "ci95": [float(m.mean() - tq * se), float(m.mean() + tq * se)], "n_wordings": int(n)}


def main() -> int:
    out: dict = {}
    rng = np.random.default_rng(51)
    # (1) order of mention
    out["order"] = {}
    for m in ("qwen0.6b", "qwen1.7b"):
        ab = tables(f"A_{m}_list_k0_e4_Tan_f0_all_N8")
        p = R / f"A_{m}_list_k0_e4_Tan_f0_allba_N8" / "gaps.json"
        if not p.exists():
            print(f"(1) {m}: ba cell missing"); continue
        ba = tables(f"A_{m}_list_k0_e4_Tan_f0_allba_N8")
        keys = sorted(set(ab) & set(ba))
        rows = []
        for key in keys:
            Jab, hab = fit_k(ab[key]["entries"]); Jba, hba = fit_k(ba[key]["entries"])
            rows.append({"wording": key[0], "format": key[1], "h_ab": hab, "h_ba": hba, "J_ab": Jab, "J_ba": Jba})
        dh = np.array([r["h_ba"] - r["h_ab"] for r in rows]); dJ = np.array([r["J_ba"] - r["J_ab"] for r in rows])
        cl = np.array([r["wording"] for r in rows])
        flips = int(sum((r["h_ab"] > 0) != (r["h_ba"] > 0) for r in rows))
        halved = int(sum(abs(r["h_ba"]) < 0.5 * abs(r["h_ab"]) for r in rows))
        mean_ab = float(np.mean([r["h_ab"] for r in rows])); mean_ba = float(np.mean([r["h_ba"] for r in rows]))
        verdict = "order" if (flips + halved) >= 5 else "token"
        out["order"][m] = {"rows": rows, "h_mean_ab": mean_ab, "h_mean_ba": mean_ba, "dh": boot_mean(dh, cl, rng), "dJ": boot_mean(dJ, cl, rng), "dh_t": t_mean(dh, cl), "dJ_t": t_mean(dJ, cl),
                           "n_sign_flips": flips, "n_halved": halved, "n": len(rows), "n_toward_A_ba": int(sum(r["h_ba"] > 0 for r in rows)),
                           "verdict_declared_rule": verdict}
        print(f"(1) {m}: h mean {mean_ab:+.2f} (A then B) -> {mean_ba:+.2f} (B then A); dh {out['order'][m]['dh']['mean']:+.2f} {out['order'][m]['dh']['ci95']}; "
              f"flips {flips}/{len(rows)}, halved {halved}/{len(rows)}, toward A after swap {out['order'][m]['n_toward_A_ba']}/{len(rows)}; dJ {out['order'][m]['dJ']['mean']:+.2f} {out['order'][m]['dJ']['ci95']} -> {verdict}")
    # (2) decisional instruction
    p = R / "A_qwen0.6b_list_k1_e4_Tan_f0_alldec_N8" / "gaps.json"
    if p.exists():
        orig = tables("A_qwen0.6b_list_k1_e4_Tan_f0_all_N8"); dec = tables("A_qwen0.6b_list_k1_e4_Tan_f0_alldec_N8")
        rows = []
        for key in sorted(set(orig) & set(dec)):
            r = {"wording": key[0], "format": key[1]}
            for lab, tab in (("orig", orig[key]), ("dec", dec[key])):
                e = tab["entries"]
                r[f"P_k-4_mem+1_{lab}"] = float(np.mean([sig(x["gap"]) for x in e if x["k"] == -4 and x["memory"] == [1]]))
                r[f"P_k+4_mem-1_{lab}"] = float(np.mean([sig(x["gap"]) for x in e if x["k"] == 4 and x["memory"] == [-1]]))
                J, Jm, h = fit_k_mem(e); r[f"J_{lab}"] = J; r[f"Jmem_{lab}"] = Jm; r[f"h_{lab}"] = h
            rows.append(r)
        cl = np.array([r["wording"] for r in rows])
        own_orig = int(sum(r["P_k-4_mem+1_orig"] > 0.5 for r in rows)); own_dec = int(sum(r["P_k-4_mem+1_dec"] > 0.5 for r in rows))
        own2_orig = int(sum(r["P_k+4_mem-1_orig"] < 0.5 for r in rows)); own2_dec = int(sum(r["P_k+4_mem-1_dec"] < 0.5 for r in rows))
        verdict = "recall" if own_dec < 5 else ("persistence" if own_dec >= 8 else "undecided")
        out["decisional"] = {"rows": rows, "n": len(rows), "own_wins_vs_minus_peers": {"orig": own_orig, "dec": own_dec}, "own_wins_vs_plus_peers": {"orig": own2_orig, "dec": own2_dec},
                             "P_k-4_mem+1_dec_min": float(min(r["P_k-4_mem+1_dec"] for r in rows)), "P_k-4_mem+1_dec_max": float(max(r["P_k-4_mem+1_dec"] for r in rows)),
                             "dJ": boot_mean(np.array([r["J_dec"] - r["J_orig"] for r in rows]), cl, rng), "dJmem": boot_mean(np.array([r["Jmem_dec"] - r["Jmem_orig"] for r in rows]), cl, rng),
                             "dJ_t": t_mean(np.array([r["J_dec"] - r["J_orig"] for r in rows]), cl), "dJmem_t": t_mean(np.array([r["Jmem_dec"] - r["Jmem_orig"] for r in rows]), cl),
                             "J_mean": {"orig": float(np.mean([r["J_orig"] for r in rows])), "dec": float(np.mean([r["J_dec"] for r in rows]))},
                             "Jmem_mean": {"orig": float(np.mean([r["Jmem_orig"] for r in rows])), "dec": float(np.mean([r["Jmem_dec"] for r in rows]))},
                             "verdict_declared_rule": verdict}
        d = out["decisional"]
        print(f"(2) 0.6B k=1: own state wins vs four minus peers {own_orig}/{len(rows)} -> {own_dec}/{len(rows)} (P range dec {d['P_k-4_mem+1_dec_min']:.2f}-{d['P_k-4_mem+1_dec_max']:.2f}); "
              f"vs four plus peers {own2_orig} -> {own2_dec}; J {d['J_mean']['orig']:.2f} -> {d['J_mean']['dec']:.2f} (dJ {d['dJ']['mean']:+.2f} {d['dJ']['ci95']}); "
              f"J_mem {d['Jmem_mean']['orig']:.2f} -> {d['Jmem_mean']['dec']:.2f} (dJmem {d['dJmem']['mean']:+.2f} {d['dJmem']['ci95']}) -> {verdict}")
    else:
        print("(2) decisional cell missing")
    # (3) generation check
    p = R / "generation_check.json"
    if p.exists():
        g = json.load(open(p, encoding="utf-8"))["models"]
        out["generation"] = {}
        for m, b in g.items():
            s = b["summary"]
            ok = s["unparseable_rate"] < 0.10 and s["mean_abs_diff"] <= 0.1 and s["n_same_side"] >= 28
            by_k: dict[int, list[float]] = {}
            for r in b["rows"]:
                if r["freq_A"] is not None:
                    by_k.setdefault(r["k"], []).append(r["freq_A"] - r["p_table_T"])
            out["generation"][m] = {**s, "representative_declared_rule": bool(ok), "mean_diff_by_k": {str(k): float(np.mean(v)) for k, v in sorted(by_k.items())}}
            print(f"(3) {m}: unparseable {s['unparseable_rate']:.3f}, mean |diff| {s['mean_abs_diff']:.3f}, max {s['max_abs_diff']:.3f}, same side {s['n_same_side']}/{s['n_pairs']}, within 0.1 {s['n_within_0.1']}/{s['n_pairs']} -> {'representative' if ok else 'NOT representative'}")
    write_json(R / "review_r2_runs.json", {"manifest": build_manifest({"script": "review_r2_runs_analysis"}, [51]), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
