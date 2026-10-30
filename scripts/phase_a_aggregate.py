#!/usr/bin/env python
"""Phase A aggregation: J_tilde and h_tilde against the knobs, paired by
formulation, with a mixed model (design notes §16 decision, §17).

Reads every results/phase_a/<cell>/measures.json and builds a long table
with one row per (cell, formulation, T). Then, per model and temperature:
    * paired differences between knob levels within formulation (the
      formulation effect cancels in the pairing), bootstrap over
      formulations for the interval;
    * mixed model  J ~ C(channel) + C(k_mem) + C(exposure) + C(format)
                     + C(format):C(exposure) + C(format):C(k_mem)
                   with a random intercept per wording (statsmodels MixedLM),
      and the format x knob interaction p-values reported so that curves
      are split by format when the interaction is significant;
    * figures: J and h versus exposure per (channel, k_mem), one line per
      format, band = bootstrap over wordings; chi_|m| scan per cell.
Output: results/phase_a/aggregate.json, figures/phase_a_*.png, a text
summary on stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402


def _shape_diagnostics(gaps_path: Path) -> dict:
    """Per formulation, from the sector-A gaps at T = 1:
    * the gap-based estimator J_gap, h_gap (and J_mem_gap with memory):
      least squares of the mean gap per (k, memory sum) on
      gap = 2 (h + J k + J_mem m). It regresses the logits themselves, so
      it is not affected by the clip at |logit| = 20.7 that label_space_fit
      applies to the ordering-averaged P before taking its logit; in the
      saturated regime of the 1.7B (gaps up to 30) that clip biases J
      low (test T3 of 2026-09-12). The two estimators coincide where no
      gap exceeds the clip. The policy used by the dynamics is still the
      ordering-averaged P, so both are reported.
    * R^2 of the linear fit of the mean gap on k (memory marginalised)
      and the unanimity excess, gap(+-e) minus the linear prediction
      (§17, domanda sulla normalizzazione).
    """
    out = {}
    if not gaps_path.exists():
        return out
    d = json.load(open(gaps_path, encoding="utf-8"))
    for tab in d["tables"]:
        byk: dict[int, list[float]] = {}
        bykm: dict[tuple, list[float]] = {}
        for e in tab["entries"]:
            byk.setdefault(e["k"], []).append(e["gap"])
            bykm.setdefault((e["k"], int(sum(e["memory"]))), []).append(e["gap"])
        ks = np.array(sorted(byk), dtype=float)
        g = np.array([np.mean(byk[int(k)]) for k in ks])
        A = np.vstack([np.ones_like(ks), ks]).T
        coef, *_ = np.linalg.lstsq(A, g, rcond=None)
        pred = A @ coef
        ss = ((g - g.mean()) ** 2).sum()
        r2 = float(1 - ((g - pred) ** 2).sum() / ss) if ss > 0 else float("nan")
        has_mem = any(m != 0 for (_, m) in bykm)
        keys = sorted(bykm)
        X = np.array([[1.0, k, m] if has_mem else [1.0, k] for (k, m) in keys], dtype=float)
        y = np.array([np.mean(bykm[key]) for key in keys])
        cg, *_ = np.linalg.lstsq(X, y, rcond=None)
        # ordering bootstrap of J_gap within the formulation: resample the gaps
        # inside each (k, memory sum) group (the same resampling as label_space_fit)
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(300):
            yb = np.array([np.mean(rng.choice(bykm[key], size=len(bykm[key]), replace=True)) for key in keys])
            cb, *_ = np.linalg.lstsq(X, yb, rcond=None)
            boots.append(cb[1] / 2)
        lo_g, hi_g = np.percentile(boots, [2.5, 97.5])
        out[(tab["wording"], tab["format"])] = {
            "J_gap_ci95": [float(lo_g), float(hi_g)],
            "r2_linear_k": r2,
            "unanimity_excess_plus": float(g[-1] - pred[-1]),
            "unanimity_excess_minus": float(g[0] - pred[0]),
            "J_gap": float(cg[1] / 2), "h_gap": float(cg[0] / 2),
            "J_mem_gap": float(cg[2] / 2) if has_mem else None,
            "max_abs_gap": float(max(abs(e["gap"]) for e in tab["entries"])),
        }
    return out


def load_rows(results: Path) -> list[dict]:
    rows = []
    for mpath in sorted(results.glob("*/measures.json")):
        d = json.load(open(mpath, encoding="utf-8"))
        cfg = d["manifest"]["config"]
        shape = _shape_diagnostics(mpath.parent / "gaps.json")
        for f in d["formulations"]:
            sh = shape.get((f["wording"], f["format"]), {})
            for T, blk in f["by_T"].items():
                fit = blk["fit"]
                rows.append({
                    "r2_linear_k": sh.get("r2_linear_k"), "unanimity_excess_plus": sh.get("unanimity_excess_plus"),
                    "unanimity_excess_minus": sh.get("unanimity_excess_minus"),
                    "cell": d["cell_id"], "model": cfg["model_key"], "channel": cfg["channel"],
                    "k_mem": cfg["k_mem"], "exposure": cfg["exposure"], "topology": cfg["topology"],
                    "wording": f["wording"], "format": f["format"], "T": float(T),
                    "J": fit["J"]["estimate"], "J_lo": fit["J"]["ci95"][0], "J_hi": fit["J"]["ci95"][1],
                    "h": fit["h"]["estimate"], "J_mem": fit.get("J_mem", {}).get("estimate"),
                    "zJ": blk["zJ"], "abs_m_final": blk["abs_m_final"], "omega": blk["o_information"],
                    "chi_rounds": blk["chi_abs_rounds"], "hyst_gap": blk["hysteresis"]["max_gap"],
                    "damage_final": blk["damage_mean"][-1], "order_var": blk["order_variance_max"],
                    "prompt_nll": f["prompt_nll"],
                    "J_gap": sh.get("J_gap"), "h_gap": sh.get("h_gap"), "J_mem_gap": sh.get("J_mem_gap"),
                    "J_gap_ci95": sh.get("J_gap_ci95"),
                    "zJ_gap": (cfg["exposure"] * sh["J_gap"]) if sh.get("J_gap") is not None else None,
                    "max_abs_gap": sh.get("max_abs_gap"),
                })
    return rows


def paired_effects(rows: list[dict], model: str, T: float, knob: str, base_filter: dict, field: str = "J") -> dict:
    """Mean paired difference J(level) - J(reference level) within (wording,
    format), bootstrap over wordings. ``base_filter`` fixes the other knobs."""
    sub = [r for r in rows if r["model"] == model and r["T"] == T and r["topology"] == "all"
           and all(r[k] == v for k, v in base_filter.items())]
    levels = sorted({r[knob] for r in sub})
    if len(levels) < 2:
        return {}
    ref = levels[0]
    out = {}
    rng = np.random.default_rng(0)
    for lev in levels[1:]:
        diffs, keys = [], []
        for r in sub:
            if r[knob] != lev:
                continue
            match = [b for b in sub if b[knob] == ref and b["wording"] == r["wording"] and b["format"] == r["format"]]
            if match:
                diffs.append(r[field] - match[0][field])
                keys.append(r["wording"])
        if not diffs:
            continue
        diffs = np.array(diffs)
        words = np.array(keys)
        uniq = np.unique(words)
        boots = []
        for _ in range(2000):
            pick = rng.choice(uniq, size=uniq.size, replace=True)
            boots.append(np.mean(np.concatenate([diffs[words == w] for w in pick])))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[f"{knob}={lev} vs {ref}"] = {"mean_diff_J": float(diffs.mean()), "ci95": [float(lo), float(hi)],
                                       "n_pairs": int(diffs.size)}
    return out


def mixed_model(rows: list[dict], model: str, T: float, field: str = "J") -> dict | None:
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    sub = [r for r in rows if r["model"] == model and r["T"] == T and r["topology"] == "all"]
    if len({r["exposure"] for r in sub}) < 2 or len({r["wording"] for r in sub}) < 2:
        return None
    df = pd.DataFrame(sub)
    terms = ["C(channel)", "C(k_mem)", "C(exposure)"]
    if df["format"].nunique() > 1:
        terms += ["C(format)", "C(format):C(exposure)", "C(format):C(k_mem)"]
    formula = f"{field} ~ " + " + ".join(terms)
    try:
        res = smf.mixedlm(formula, df, groups=df["wording"]).fit(reml=False)
    except Exception as exc:  # singular fits happen on partial grids
        return {"error": str(exc)}
    out = {"formula": formula, "params": {k: float(v) for k, v in res.params.items()},
           "pvalues": {k: float(v) for k, v in res.pvalues.items()},
           "wording_variance": float(res.cov_re.iloc[0, 0]) if res.cov_re.size else None,
           "residual_variance": float(res.scale)}
    inter = {k: v for k, v in out["pvalues"].items() if ":" in k}
    out["format_interaction_significant"] = bool(inter) and min(inter.values()) < 0.05
    return out


def figures(rows: list[dict], fig_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for model in sorted({r["model"] for r in rows}):
        for T in (1.0,):
            fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
            for ai, (channel, k_mem) in enumerate([("list", 0), ("list", 3), ("count", 0), ("count", 3)]):
                ax = axes.flat[ai]
                for fmt in ("comma", "lines"):
                    xs, ys, lo, hi = [], [], [], []
                    for e in (2, 4, 7):
                        sub = [r["J"] for r in rows if r["model"] == model and r["T"] == T and r["channel"] == channel
                               and r["k_mem"] == k_mem and r["exposure"] == e and r["format"] == fmt and r["topology"] == "all"]
                        if not sub:
                            continue
                        b = [np.mean(np.random.default_rng(i).choice(sub, size=len(sub))) for i in range(500)]
                        xs.append(e); ys.append(np.mean(sub)); lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
                    if xs:
                        ax.errorbar(xs, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], marker="o", capsize=3, label=fmt)
                ax.axhline(0, color="grey", lw=0.5)
                ax.set_title(f"{channel}, k_mem={k_mem}")
                ax.set_xlabel("exposure (peers shown)"); ax.set_ylabel("J_tilde (label space)")
                if ax.get_legend_handles_labels()[0]:
                    ax.legend()
            fig.suptitle(f"{model}, T={T}: J_tilde by exposure, mean over wordings, band = bootstrap over wordings")
            fig.tight_layout()
            path = fig_dir / f"phase_a_J_{model}_T{T}.png"
            fig.savefig(path, dpi=130); plt.close(fig); made.append(str(path))
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args(argv)
    rows = load_rows(Path(a.results))
    if not rows:
        print("no measures yet"); return 1
    models = sorted({r["model"] for r in rows})
    Ts = sorted({r["T"] for r in rows})
    agg = {"n_rows": len(rows), "cells": sorted({r["cell"] for r in rows}), "paired": {}, "mixed": {}}
    for model in models:
        for T in Ts:
            key = f"{model}|T={T}"
            agg["paired"][key] = {}
            for field in ("J", "J_gap", "zJ", "zJ_gap"):
                for ch in ("list", "count"):
                    agg["paired"][key][f"{field}|exposure|{ch},k0"] = paired_effects(rows, model, T, "exposure", {"channel": ch, "k_mem": 0}, field)
                    agg["paired"][key][f"{field}|exposure|{ch},k3"] = paired_effects(rows, model, T, "exposure", {"channel": ch, "k_mem": 3}, field)
                    agg["paired"][key][f"{field}|k_mem|{ch},e4"] = paired_effects(rows, model, T, "k_mem", {"channel": ch, "exposure": 4}, field)
                agg["paired"][key][f"{field}|channel|k0,e4"] = paired_effects(rows, model, T, "channel", {"k_mem": 0, "exposure": 4}, field)
            agg["mixed"][key] = mixed_model(rows, model, T)
            agg["mixed_J_gap"] = agg.get("mixed_J_gap", {})
            agg["mixed_J_gap"][key] = mixed_model(rows, model, T, "J_gap")
    agg["figures"] = figures(rows, Path(a.figures))
    agg["rows"] = rows
    write_json(Path(a.results) / "aggregate.json", {"manifest": build_manifest({"results": a.results}, []), **agg})
    # text summary
    for model in models:
        print(f"\n== {model}, T=1.0 ==")
        for r in sorted([r for r in rows if r["model"] == model and r["T"] == 1.0], key=lambda r: (r["channel"], r["k_mem"], r["exposure"])):
            pass
        for (ch, k, e) in sorted({(r["channel"], r["k_mem"], r["exposure"]) for r in rows if r["model"] == model and r["topology"] == "all"}):
            sub = [r for r in rows if r["model"] == model and r["T"] == 1.0 and r["channel"] == ch and r["k_mem"] == k and r["exposure"] == e and r["topology"] == "all"]
            J = np.array([r["J"] for r in sub]); h = np.array([r["h"] for r in sub]); zJ = np.array([r["zJ"] for r in sub])
            om = [r["omega"] for r in sub if r["omega"] is not None]
            print(f"  {ch:5s} k{k} e{e}: J {J.mean():+.3f} (sd {J.std(ddof=1) if J.size > 1 else 0:.3f}, {J.min():+.2f}..{J.max():+.2f})  "
                  f"h {h.mean():+.3f}  zJ {zJ.mean():+.2f}  |m| {np.mean([r['abs_m_final'] for r in sub]):.2f}  "
                  f"Omega {np.mean(om) if om else float('nan'):+.2f}  hyst {np.mean([r['hyst_gap'] for r in sub]):.2f}  "
                  f"R2(k) {np.mean([r['r2_linear_k'] for r in sub if r['r2_linear_k'] is not None]):.2f}  "
                  f"unan.excess+ {np.mean([r['unanimity_excess_plus'] for r in sub if r['unanimity_excess_plus'] is not None]):+.2f}")
        for key, blk in agg["paired"][f"{model}|T=1.0"].items():
            for k2, v in blk.items():
                print(f"  paired {key} {k2}: diff = {v['mean_diff_J']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] n={v['n_pairs']}")
        mm = agg["mixed"].get(f"{model}|T=1.0")
        if mm and "pvalues" in mm:
            print("  mixed model p-values:", {k: round(v, 3) for k, v in mm["pvalues"].items()},
                  "| format x knob interaction significant:", mm["format_interaction_significant"])
    print("figures:", agg["figures"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
