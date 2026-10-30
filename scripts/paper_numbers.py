#!/usr/bin/env python
"""Every number the paper may quote, generated from the result JSONs
(the claims register (not included) rule: numbers are read by script, never transcribed).

Writes paper/numbers.json with
    values   named quantities {name: {"value", "ci", "text"}} for the
             table and figure scripts;
    strings  the set of every formatted number (2 decimals; 3 for p-values,
             R^2 and coverage; 4 for the Phase 0 constants and the bf16
             correlation) that may appear in the prose. The hook
             the number check of the paper sources (not included) blocks any decimal number in
             paper/ that is not in this set, so a value in the paper always
             traces to a result file with the same rounding (house rule P0
             on metric and rounding consistency).
and paper/numbers.md, the same values as a readable table.
The set is curated on purpose: a generic dump of every float in the result
files would cover most two-decimal values and make the check meaningless.
Constants that are not measurements (exact Ising values, model sizes,
published values used for comparison) are listed explicitly in CONSTANTS
with their source.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402

R = Path("results")
V: dict[str, dict] = {}
S: set[str] = set()

# name: (value, decimals, source)
CONSTANTS = {
    "ising.Tc_exact": (2.2692, 4, "Onsager 1944, 2/ln(1+sqrt 2)"),
    "ising.gamma_over_nu_exact": (1.75, 2, "Ising 2D exact"),
    "ising.beta_over_nu_exact": (0.125, 3, "Ising 2D exact"),
    "model.size_small_B": (0.6, 1, "Qwen3-0.6B"),
    "model.size_large_B": (1.7, 1, "Qwen3-1.7B"),
    "lit.denobili.zJ_llama_max": (1.4, 1, "De Nobili, 4 neighbours x J~ 0.35 (llama): total push comparable to our z_bar J_gap at 4 peers"),
    "lit.denobili.J_max": (0.35, 2, "De Nobili, J~ at 4 neighbours, llama upper end"),
    "lit.denobili.h_llama": (2.8, 1, "De Nobili, h~ for llama"),
    "lit.denobili.ratio_llama": (8.0, 1, "De Nobili, h~/J~ for llama"),
    "lit.demarzo.z": (49.0, 1, "De Marzo arXiv:2605.10721, peers observed (N = 50)"),
    "rule.P4b.m_disordered_max": (0.3, 1, "declared rule of the P4b control (the results notes (not included)): |m| below this stays disordered"),
    "rule.P4b.m_ordered_min": (0.7, 1, "declared rule of the P4b control: |m| above this orders"),
    "rule.P4b.dF0_low": (0.9, 1, "§24 P4b outcome: no formulation with F'(0) below this orders"),
    "rule.P4b.dF0_high": (1.1, 1, "§24 P4b outcome: no formulation with F'(0) above this stays disordered"),
    "rule.bootstrap.percentile_low": (2.5, 1, "bootstrap interval bounds used throughout"),
    "rule.bootstrap.percentile_high": (97.5, 1, "bootstrap interval bounds used throughout"),
    "rule.sectorA.field_dominated_min": (0.9, 1, "revision R1-02: a single stable fixed point with |m*| above this is field-dominated consensus"),
    "ising2d.betaJc": (0.4407, 2, "Onsager: beta J_c = ln(1 + sqrt 2)/2 on the square lattice (revision R1-08)"),
    "mf.threshold_J_e4": (0.25, 2, "mean-field threshold of the linear criterion at 4 peers, J = 1/e (revision R1-08)"),
    "lit.flint.T": (0.5, 1, "Flint et al. read their policies at decoding scale 0.5 (the reading notes (not included); revision R2-12)"),
    "lit.flint.H": (5, 0, "Flint et al.: memory of the last H = 5 interactions (the reading notes (not included))"),
    "lit.demarzo.T": (0.2, 1, "De Marzo et al. sample at decoding scale 0.2 (the reading notes (not included); revision R2-12)"),
    "rule.saturation.P": (0.99, 2, "revision R2-07: a unanimity jump is saturated when P(+|+e) and P(-|-e) both exceed this"),
    "rule.label_mass.low": (0.9, 1, "revision R2-01: fraction of tabulated entries whose mass on the two label tokens is below this"),
    # arXiv identifiers of the cited preprints (the reading notes (not included)), so that the
    # bibliography of the arXiv version passes the number guard
    "ref.okawa": (2608.02827, 5, "Okawa, ICML 2026"),
    "ref.denobili": (2605.10528, 5, "De Nobili 2026"),
    "ref.denobili_ng": (2608.02178, 5, "De Nobili, Iyer, Codello, Burioni 2026"),
    "ref.el": (2608.16578, 5, "El et al. 2026"),
    "ref.liu": (2601.17311, 5, "Liu, Kong, Pei 2026"),
    "ref.ashery": (2410.08948, 5, "Ashery, Aiello, Baronchelli, Sci. Adv. 2025"),
    "ref.flint": (2510.22422, 5, "Flint et al., PNAS 2026"),
    "ref.brockers": (2509.06858, 5, "Brockers, Ehrlich, Priesemann 2026"),
    "ref.demarzo": (2605.10721, 5, "De Marzo et al. 2026"),
    "ref.riedl": (2510.05174, 5, "Riedl, ICLR 2026"),
    "ref.bellina": (2601.05384, 5, "Bellina, De Marzo, Garcia 2026"),
    "ref.tanaka": (2603.24676, 5, "Tanaka 2026"),
    "ref.cho": (2505.21588, 5, "Cho, Guntuku, Ungar 2025"),
    "ref.weng": (2501.13381, 5, "Weng, Chen, Wang, ICLR 2025"),
    "ref.okawa_models.gpt": (4.1, 1, "GPT 4.1 family fitted by Okawa (R2-F01, 2026-09-30)"),
    "ref.okawa_models.llama": (3.1, 1, "Llama 3.1 family fitted by Okawa (R2-F01, 2026-09-30)"),
}


def f2(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def put(name: str, value: float, ci: list | None = None, nd: int = 2, extra_nd: tuple = ()) -> None:
    txt = f2(value, nd) + (f" [{f2(ci[0], nd)}, {f2(ci[1], nd)}]" if ci else "")
    V[name] = {"value": float(value), "ci": [float(c) for c in ci] if ci else None, "text": txt}
    for d in (nd, *extra_nd):
        S.add(f2(value, d))
        if ci:
            S.update({f2(ci[0], d), f2(ci[1], d)})


def constants() -> None:
    for name, (val, nd, src) in CONSTANTS.items():
        put(f"const.{name}", val, nd=nd)
        V[f"const.{name}"]["source"] = src


def phase_a_cells() -> None:
    a = json.load(open(R / "phase_a" / "aggregate.json", encoding="utf-8"))
    rows = [r for r in a["rows"] if r["T"] == 1.0 and r["topology"] == "all"]
    for model in sorted({r["model"] for r in rows}):
        for ch in ("list", "count"):
            for k in (0, 3):
                for e in (2, 4, 7):
                    sub = [r for r in rows if r["model"] == model and r["channel"] == ch and r["k_mem"] == k and r["exposure"] == e]
                    if not sub:
                        continue
                    tag = f"{model}.{ch}.k{k}.e{e}"
                    for field in ("J_gap", "zJ_gap", "h_gap", "J_mem_gap", "abs_m_final", "omega", "hyst_gap", "unanimity_excess_plus"):
                        vals = [r[field] for r in sub if r.get(field) is not None]
                        if vals:
                            put(f"{tag}.{field}.mean", float(np.mean(vals)))
                    put(f"{tag}.J_gap.sd", float(np.std([r["J_gap"] for r in sub], ddof=1)))
                    hw = [0.5 * (r["J_gap_ci95"][1] - r["J_gap_ci95"][0]) for r in sub if r.get("J_gap_ci95")]
                    if hw:
                        put(f"{tag}.J_gap.ci_halfwidth_within.mean", float(np.mean(hw)))
                        put(f"{tag}.J_gap.ci_halfwidth_within.max", float(np.max(hw)))
                    put(f"{tag}.J_gap.min", float(min(r["J_gap"] for r in sub)))
                    put(f"{tag}.J_gap.max", float(max(r["J_gap"] for r in sub)))
                    if k:
                        put(f"{tag}.J_mem_gap.min", float(min(r["J_mem_gap"] for r in sub)))
                        put(f"{tag}.J_mem_gap.max", float(max(r["J_mem_gap"] for r in sub)))
                    put(f"{tag}.r2_linear_k.mean", float(np.mean([r["r2_linear_k"] for r in sub])))
        for key, blk in a["paired"][f"{model}|T=1.0"].items():
            for k2, v in blk.items():
                put(f"{model}.paired.{key}.{k2}", v["mean_diff_J"], v["ci95"])
        for label, block in (("mixed", a.get("mixed", {})), ("mixed_J_gap", a.get("mixed_J_gap", {}))):
            mm = block.get(f"{model}|T=1.0")
            if mm and "pvalues" in mm:
                for k2, pv in mm["pvalues"].items():
                    put(f"{model}.{label}.p.{k2}", pv, nd=3)


def exposure_shape() -> None:
    d = json.load(open(R / "phase_a" / "exposure_shape.json", encoding="utf-8"))
    for model, r in d["models"].items():
        for e, j in r["jump"].items():
            put(f"{model}.jump.e{e}.mean", j["mean"])
            if j["sd"] is not None:
                put(f"{model}.jump.e{e}.sd", j["sd"])
            put(f"{model}.J_gap.e{e}", r["J_gap"][e])
            put(f"{model}.zJ_gap.e{e}", r["zJ_gap"][e])
        for e, q in r["ratio"].items():
            put(f"{model}.ratio.e{e}", q["mean"], q["ci95"])
            put(f"{model}.ratio.e{e}.pred_count", q["pred_count"])
        if "exponent_e_ge_ref" in r:
            x = r["exponent_e_ge_ref"]
            put(f"{model}.exponent_e_ge_ref", x["mean"], x["ci95"])
            put(f"{model}.exponent_e_ge_ref.min", x["min"]); put(f"{model}.exponent_e_ge_ref.max", x["max"])
            V[f"{model}.exponent_e_ge_ref.n"] = {"value": x["n"], "ci": None, "text": str(x["n"])}
        c = r["collapse"]
        for k2, v in c["r2_mean"].items():
            put(f"{model}.collapse.r2.{k2}", v, nd=3)
        put(f"{model}.collapse.coef_fraction", c["coef_fraction_in_both"]["mean"], c["coef_fraction_in_both"]["ci95"])
        put(f"{model}.collapse.coef_count", c["coef_count_in_both"]["mean"], c["coef_count_in_both"]["ci95"], nd=3, extra_nd=(2,))
        V[f"{model}.collapse.frac_beats_count"] = {"value": None, "ci": None, "text": c["frac_beats_count_aic"]}


def lattice() -> None:
    d = json.load(open(R / "phase_a" / "lattice_control.json", encoding="utf-8"))
    for model, m in d["models"].items():
        for q, v in m["compare"]["paired"].items():
            put(f"{model}.lattice.{q}.all", v["mean_all"])
            put(f"{model}.lattice.{q}.lattice", v["mean_lattice"])
            put(f"{model}.lattice.{q}.diff", v["mean_diff"], v["ci95"])
        for topo in ("lattice", "all"):
            if "t_cons_exponent" in m["size_scan"][topo]:
                put(f"{model}.size_scan.{topo}.exponent", m["size_scan"][topo]["t_cons_exponent"])
            for n, blk in m["size_scan"][topo].items():
                if isinstance(blk, dict) and "frac_consensus" in blk:
                    put(f"{model}.size_scan.{topo}.N{blk['N']}.frac_consensus", blk["frac_consensus"])


def stubborn() -> None:
    p = R / "phase_a" / "stubborn_placement.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for model, m in d["models"].items():
        for k2, v in m["means"].items():
            put(f"{model}.stubborn.{k2}", v)
        x = m["lattice_random_minus_contiguous"]
        put(f"{model}.stubborn.lattice_random_minus_contiguous", x["mean"], x["ci95"])


def memory_curve() -> None:
    p = R / "phase_a" / "memory_curve.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for model, m in d["models"].items():
        for k, v in m["J_gap_mean"].items():
            put(f"{model}.memory_curve.J_gap.k{k}", v)
        for k, v in m["J_mem_gap_mean"].items():
            put(f"{model}.memory_curve.J_mem_gap.k{k}", v)
        for name, v in m["paired"].items():
            put(f"{model}.memory_curve.paired.{name}", v["mean"], v["ci95"])
        V[f"{model}.memory_curve.verdict"] = {"value": None, "ci": None, "text": m["verdict"]}
        for k, v in m.get("J_mem_gap", {}).items():
            put(f"{model}.memory_curve.J_mem_gap.k{k}.ci", v["mean"], v["ci95"])
        for name, v in m.get("J_mem_paired", {}).items():
            put(f"{model}.memory_curve.J_mem_paired.{name}", v["mean"], v["ci95"])


def phase0() -> None:
    """Curated Phase 0 gate values (results/phase0/*.json)."""
    ising = json.load(open(R / "phase0" / "ising_calibration.json", encoding="utf-8"))
    for key in ("Tc", "binder_crossing", "gamma_over_nu", "beta_over_nu"):
        put(f"phase0.ising.{key}", ising[key]["estimate"], ising[key]["ci95"], nd=4, extra_nd=(3, 2))
    drift = json.load(open(R / "phase0" / "drift_control.json", encoding="utf-8"))
    put("phase0.drift.t_cons_exponent", drift["t_cons_exponent"]["estimate"], drift["t_cons_exponent"]["ci95"])
    for n, v in drift["t_cons_over_N2"].items():
        put(f"phase0.drift.t_cons_over_N2.N{n}", v)
    reg = json.load(open(R / "phase0" / "regression_recovery.json", encoding="utf-8"))
    for k2, v in reg["recovery_study"]["coverage95"].items():
        put(f"phase0.regression.coverage95.{k2}", v, nd=3)
    for k2, v in reg["recovery_study"]["relative_bias"].items():
        put(f"phase0.regression.relative_bias.{k2}", v, nd=3, extra_nd=(4,))
    for term in ("two_term", "three_term"):
        for k2, v in reg[term]["true"].items():
            put(f"phase0.regression.{term}.true.{k2}", v)
        for k2, v in reg[term]["fit"].items():
            put(f"phase0.regression.{term}.fit.{k2}", v["estimate"], v["ci95"], nd=3, extra_nd=(2,))
    dn = json.load(open(R / "phase0" / "denobili_check.json", encoding="utf-8"))
    for c in dn["cells"]:
        mk = "qwen0.6b" if "0.6B" in c["model"] else "qwen1.7b"
        tag = f"phase0.denobili.{mk}.T{c['temperature']}"
        put(f"{tag}.h", c["fit"]["h"]["estimate"], c["fit"]["h"]["ci95"])
        put(f"{tag}.J", c["fit"]["J"]["estimate"], c["fit"]["J"]["ci95"])
        if abs(c["fit"]["J"]["estimate"]) > 1e-9:
            put(f"{tag}.h_over_J", abs(c["fit"]["h"]["estimate"]) / abs(c["fit"]["J"]["estimate"]), nd=1, extra_nd=(2,))
        put(f"{tag}.cycle_test.max_abs", c["cycle_test"]["max_abs_cycle_sum"])
        put(f"{tag}.order_variance.max", max(float(v) for v in c["order_variance"].values()))


def field_ratio() -> None:
    """|h~|/J~ per formulation on the 4-peer, no-memory list cells (gap estimator),
    and the replication with De Nobili's literal prompt (results/phase0/denobili_literal_*.json)."""
    a = json.load(open(R / "phase_a" / "aggregate.json", encoding="utf-8"))
    for model in ("qwen0.6b", "qwen1.7b"):
        sub = [r for r in a["rows"] if r["model"] == model and r["T"] == 1.0 and r["channel"] == "list"
               and r["k_mem"] == 0 and r["exposure"] == 4 and r["topology"] == "all"]
        if not sub:
            continue
        ratios = [abs(r["h_gap"]) / abs(r["J_gap"]) for r in sub if abs(r["J_gap"]) > 1e-9]
        put(f"{model}.field_ratio.e4.median", float(np.median(ratios)), nd=1, extra_nd=(2,))
        put(f"{model}.field_ratio.e4.min", float(min(ratios)), nd=1, extra_nd=(2,))
        put(f"{model}.field_ratio.e4.max", float(max(ratios)), nd=1, extra_nd=(2,))
        V[f"{model}.field_ratio.e4.n_above_3"] = {"value": sum(x > 3 for x in ratios), "ci": None,
                                                 "text": f"{sum(x > 3 for x in ratios)}/{len(sub)}"}
        V[f"{model}.field_ratio.e4.n_J_below_0.1"] = {"value": sum(abs(r["J_gap"]) <= 0.1 for r in sub), "ci": None,
                                                     "text": f"{sum(abs(r['J_gap']) <= 0.1 for r in sub)}/{len(sub)}"}
        p = R / "phase0" / f"denobili_literal_{model}.json"
        if p.exists():
            d = json.load(open(p, encoding="utf-8"))
            for k2 in ("h", "J"):
                put(f"{model}.denobili_literal.{k2}", d["fit"][k2]["estimate"], d["fit"][k2]["ci95"])
            rr = d["fit"]["ratio_abs_h_over_J"]
            put(f"{model}.denobili_literal.ratio", rr["estimate"], rr["ci95"], nd=1, extra_nd=(2,))
            put(f"{model}.denobili_literal.J_odd", d["J_odd"])
            for k2, v in d["p_yes_by_k_T1"].items():
                put(f"{model}.denobili_literal.p_yes.k{k2}", v, nd=3, extra_nd=(2,))


def bf16_check() -> None:
    """bfloat16 against float32 on the same cell (0.6B, list, k0, e2), recomputed from the gaps."""
    a = R / "phase_a" / "A_qwen0.6b_list_k0_e2_Tan_f0_all_N8" / "gaps.json"
    b = R / "phase_a_bf16check" / "A_qwen0.6b_list_k0_e2_Tan_f0_all_N8" / "gaps.json"
    if not (a.exists() and b.exists()):
        return
    ga, gb = json.load(open(a, encoding="utf-8")), json.load(open(b, encoding="utf-8"))
    key = lambda t, e: (t["wording"], t["format"], tuple(e["states"]), tuple(e["memory"]))
    da = {key(t, e): e["gap"] for t in ga["tables"] for e in t["entries"]}
    db = {key(t, e): e["gap"] for t in gb["tables"] for e in t["entries"]}
    common = sorted(set(da) & set(db))
    if len(common) < 4:
        return
    x = np.array([da[k] for k in common]); y = np.array([db[k] for k in common])
    put("bf16.n_gaps", float(len(common)), nd=0)
    put("bf16.diff.mean", float(np.mean(y - x)))
    put("bf16.diff.sd", float(np.std(y - x, ddof=1)))
    put("bf16.diff.max_abs", float(np.max(np.abs(y - x))))
    put("bf16.corr", float(np.corrcoef(x, y)[0, 1]), nd=4, extra_nd=(3,))
    put("bf16.gap.min", float(x.min())); put("bf16.gap.max", float(x.max()))


def fp32_check_17b() -> None:
    """float32 against bfloat16 for the 1.7B on the same cells (C3, test declared
    on 2026-09-21): the two-peer cell (ten formulations) and one 30-peer unit
    (wording 0, comma), gap by gap on identical orderings."""
    for e, label in ((2, "e2"), (30, "e30")):
        cell = f"A_qwen1.7b_list_k0_e{e}_Tan_f0_all_N{8 if e == 2 else 32}"
        a = R / "phase_a" / cell / "gaps.json"; b = R / "phase_a_fp32check" / cell / "gaps.json"
        if not (a.exists() and b.exists()):
            continue
        ga, gb = json.load(open(a, encoding="utf-8")), json.load(open(b, encoding="utf-8"))
        key = lambda t, x: (t["wording"], t["format"], tuple(x["states"]), tuple(x["memory"]))
        da = {key(t, x): x["gap"] for t in ga["tables"] for x in t["entries"]}
        db = {key(t, x): x["gap"] for t in gb["tables"] for x in t["entries"]}
        common = sorted(set(da) & set(db))
        if len(common) < 4:
            continue
        x = np.array([da[k] for k in common]); y = np.array([db[k] for k in common])   # x bf16, y fp32
        tag = f"fp32check.qwen1.7b.{label}"
        put(f"{tag}.n_gaps", float(len(common)), nd=0)
        put(f"{tag}.diff.mean", float(np.mean(y - x))); put(f"{tag}.diff.sd", float(np.std(y - x, ddof=1)))
        put(f"{tag}.diff.max_abs", float(np.max(np.abs(y - x))))
        put(f"{tag}.corr", float(np.corrcoef(x, y)[0, 1]), nd=4, extra_nd=(3,))
        put(f"{tag}.gap.min", float(x.min())); put(f"{tag}.gap.max", float(x.max()))
        put(f"{tag}.max_abs_over_range_pct", float(100 * np.max(np.abs(y - x)) / (x.max() - x.min())), nd=1, extra_nd=(2,))
        print(f"fp32 check 1.7B {label}: n={len(common)} mean diff {np.mean(y-x):+.3f} sd {np.std(y-x, ddof=1):.3f} "
              f"max |diff| {np.max(np.abs(y-x)):.3f} corr {np.corrcoef(x, y)[0, 1]:.5f} gap range [{x.min():.2f}, {x.max():.2f}] "
              f"max/range {100*np.max(np.abs(y-x))/(x.max()-x.min()):.2f}%")


def fp32_recheck_quantities() -> None:
    """Declared follow-up of the C3 rule (2026-09-21): z_bar J_gap, F'(0) and R^2 of
    the 30-peer unit (wording 0, comma) in bfloat16 against float32, with the
    ordering bootstrap of the bfloat16 table as the sampling error."""
    a = R / "phase_a" / "A_qwen1.7b_list_k0_e30_Tan_f0_all_N32" / "gaps.json"
    b = R / "phase_a_fp32check" / "A_qwen1.7b_list_k0_e30_Tan_f0_all_N32" / "gaps.json"
    if not (a.exists() and b.exists()):
        return
    from scripts.phase_a_orderings_check import stats
    rng = np.random.default_rng(3)
    tab = lambda path: {(t["wording"], t["format"]): t for t in json.load(open(path, encoding="utf-8"))["tables"]}
    bf, fp = tab(a)[(0, "comma")], tab(b)[(0, "comma")]
    s_bf, s_fp = stats(bf["entries"], 30), stats(fp["entries"], 30)
    boots = [stats(bf["entries"], 30, rng) for _ in range(300)]
    for q, nd in (("zJ_gap", 2), ("dF0_sym", 2), ("r2_linear_k", 3), ("jump", 2)):
        lo, hi = np.percentile([x[q] for x in boots], [2.5, 97.5])
        put(f"fp32check.qwen1.7b.e30.w0_comma.{q}.bf16", s_bf[q], [lo, hi], nd=nd)
        put(f"fp32check.qwen1.7b.e30.w0_comma.{q}.fp32", s_fp[q], nd=nd)


def fp32_recheck_full() -> None:
    """Full float32 recheck (results/phase_a/fp32_recheck.json): per exposure,
    F'(0) and z_bar J_gap in both precisions, inside counts, class counts."""
    p = R / "phase_a" / "fp32_recheck.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for e, blk in d["exposures"].items():
        s = blk["summary"]; tag = f"fp32recheck.qwen1.7b.e{e}"
        _text(f"{tag}.dF0_inside", s["dF0_fp32_inside_bf16_ci"])
        put(f"{tag}.dF0.mean_bf16", s["dF0_mean_bf16"]); put(f"{tag}.dF0.mean_fp32", s["dF0_mean_fp32"])
        put(f"{tag}.zJ.mean_bf16", s["zJ_mean_bf16"]); put(f"{tag}.zJ.mean_fp32", s["zJ_mean_fp32"])
        z = s["zJ_paired_fp32_minus_bf16"]; put(f"{tag}.zJ.paired_fp32_minus_bf16", z["mean"], z["ci95"], nd=2, extra_nd=(3,))
        for prec in ("bf16", "fp32"):
            c = s[f"classes_{prec}"]
            _text(f"{tag}.classes_{prec}", f"{c['above']}/{c['critical']}/{c['below']}")
    _text("fp32recheck.qwen1.7b.dF0_inside_total", d["rule"]["dF0_inside_total"])


def dF0_classes() -> None:
    """Interval-based mean-field classes (results/phase_a/dF0_classes.json and
    the float32 twin): counts per class, mean F'(0), per model, exposure and
    number of orderings."""
    for fname, tag0 in (("dF0_classes.json", "classes"), ("dF0_classes_fp32.json", "classes_fp32")):
        p = R / "phase_a" / fname
        if not p.exists():
            continue
        d = json.load(open(p, encoding="utf-8"))
        for model, blocks in d["models"].items():
            for key, b in blocks.items():
                tag = f"{model}.{tag0}.{key}"
                c = b["counts"]; n = len(b["rows"])
                for cls in ("ordered", "disordered", "undetermined"):
                    _text(f"{tag}.{cls}", f"{c[cls]}/{n}", c[cls])
                _text(f"{tag}.triple", f"{c['ordered']}/{c['disordered']}/{c['undetermined']}")
                _text(f"{tag}.exact", str(c.get("exact", 0)), c.get("exact", 0))
                put(f"{tag}.dF0_mean", b["dF0_mean"])
                put(f"{tag}.dF0_min", float(min(r["dF0"] for r in b["rows"]))); put(f"{tag}.dF0_max", float(max(r["dF0"] for r in b["rows"])))


def review_r1() -> None:
    """Numbers computed for the answers to revision round, R1 (results/phase_a/review_r1.json,
    mf_dynamics_check_sectorA.json, mf_dynamics_check_nearcrit.json)."""
    p = R / "phase_a" / "review_r1.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for e, c in d["ceiling"].items():
        put(f"const.mf.ceiling.e{e}", c, nd=2, extra_nd=(3,))
    for model, blocks in d["ratio"].items():
        for key, b in blocks.items():
            tag = f"{model}.ceiling_ratio.{key}"
            put(f"{tag}.mean", b["ratio_mean"]); put(f"{tag}.min", b["ratio_min"]); put(f"{tag}.max", b["ratio_max"])
            _text(f"{tag}.n_marginal", f"{b['n_marginal']}/{len(b['per_formulation'])}", b["n_marginal"])
    for model, by_e in d["sector_A"].items():
        for e, b in by_e.items():
            c = b["counts"]; tag = f"{model}.sectorA.e{e}"
            for k in c:
                _text(f"{tag}.{k}", f"{c[k]}/{len(b['rows'])}", c[k])
            _text(f"{tag}.triple", f"{c['bistable']}/{c['field_dominated']}/{c['disordered']}")
            dis = [abs(r["stable_m_A"][0]) for r in b["rows"] if r["class"] == "disordered"]
            if dis:
                put(f"{tag}.disordered_abs_m.min", min(dis)); put(f"{tag}.disordered_abs_m.max", max(dis))
    for model, by_e in d["sector_A"].items():
        dis = [abs(r["stable_m_A"][0]) for b in by_e.values() for r in b["rows"] if r["class"] == "disordered"]
        if dis:
            put(f"{model}.sectorA.disordered_abs_m.min_all", min(dis)); put(f"{model}.sectorA.disordered_abs_m.max_all", max(dis))
    for model, topos in d["consensus_time"].items():
        for topo, per_N in topos.items():
            for N, v in per_N.items():
                tag = f"{model}.tcons.{topo}.N{N}"
                put(f"{tag}.voter_rounds", v["voter_rounds_ref"], nd=1)
                put(f"{tag}.N_lnN", float(N) * math.log(float(N)), nd=1)
                if v["t_cons_mean"] is not None:
                    put(f"{tag}.mean", v["t_cons_mean"], nd=1); put(f"{tag}.median", v["t_cons_median_over_formulations"], nd=1)
                    _text(f"{tag}.n_below_voter", f"{v['n_formulations_below_voter']}/{v['n_formulations']}", v["n_formulations_below_voter"])
        a16 = topos["lattice"]["16"]["t_cons_mean"]; a64 = topos["lattice"]["64"]["t_cons_mean"]
        if a16 and a64:
            put(f"{model}.tcons.lattice.ratio_64_16", a64 / a16, nd=1)
            put(f"{model}.tcons.lattice.NlnN_ratio_64_16", 64 * math.log(64) / (16 * math.log(16)), nd=1)
        c16 = topos["all"]["16"]["t_cons_mean"]; c64 = topos["all"]["64"]["t_cons_mean"]
        if c16 and c64:
            put(f"{model}.tcons.all.ratio_64_16", c64 / c16, nd=1)
    for model, g in d["memory_gain"].items():
        for k, v in g["by_k"].items():
            put(f"{model}.memgain.e4.k{k}", v["gain"]); put(f"{model}.memgain.e4.k{k}.kJmem", v["kJmem"])
        g0 = g["by_k"]["0"]["gain"]; g3 = g["by_k"]["3"]["gain"]
        put(f"{model}.memgain.e4.reduction_k3", (g0 - g3) / g0, nd=2)
        _text(f"{model}.memgain.e4.reduction_k3.percent", f"{round(100 * (g0 - g3) / g0)}%", round(100 * (g0 - g3) / g0))
    for model, by_e in d["memory_gain_grid"].items():
        for e, v in by_e.items():
            put(f"{model}.memgain.e{e}.k0", v["gain_k0"]); put(f"{model}.memgain.e{e}.k3", v["gain_k3"])
    for model, res in d["zJ_paired"].items():
        for k, v in res.items():
            put(f"{model}.zJ_paired.{k}", v["mean"], v["ci95"])
    for model, b in d["tipping"].items():
        rows = b["rows"]; n = len(rows)
        up = sum(r["zc_up"] in (0.125, 0.25) for r in rows); down = sum(r["zc_down"] in (-0.125, -0.25) for r in rows)
        _text(f"{model}.tipping.up.n_one_or_two_agents", f"{up}/{n}", up)
        _text(f"{model}.tipping.down.n_one_or_two_agents", f"{down}/{n}", down)
        put(f"{model}.tipping.z_step", b["z_step"], nd=3)
    # sector-A dynamics against the sector-A fixed points; near-critical rerun
    st = json.load(open(R / "phase_a" / "meanfield_stability.json", encoding="utf-8"))["models"]
    pA = R / "phase_a" / "mf_dynamics_check_sectorA.json"
    if pA.exists():
        dA = json.load(open(pA, encoding="utf-8"))
        put("sectorA_dyn.N", dA["N"], nd=0); put("sectorA_dyn.rounds", dA["rounds"], nd=0); put("sectorA_dyn.seeds", dA["seeds"], nd=0)
        n_ok = n_tot = n_bist = 0; worst = 0.0
        for model, blk in dA["models"].items():
            for r in blk["rows"]:
                s = [x for x in st[model] if x["exposure"] == dA["exposure"] and x["wording"] == r["wording"] and x["format"] == r["format"]][0]
                n_tot += 1
                if s["n_stable_A"] >= 2:
                    n_bist += 1; ok = r["abs_m_final"] > 0.9
                else:
                    dev = abs(r["m_signed_final_mean"] - s["stable_m_A"][0]); worst = max(worst, dev); ok = dev <= 0.1
                n_ok += ok
        _text("sectorA_dyn.agree", f"{n_ok}/{n_tot}", n_ok); put("sectorA_dyn.n_bistable", n_bist, nd=0)
        put("sectorA_dyn.max_dev_single_fixed_point", worst, nd=2)
    pN = R / "phase_a" / "mf_dynamics_check_nearcrit.json"
    if pN.exists():
        dN = json.load(open(pN, encoding="utf-8"))
        put("nearcrit.rounds", dN["rounds"], nd=0)
        n_ok = n = 0
        for model, blk in dN["models"].items():
            for r in blk["rows"]:
                n += 1; n_ok += r["agree"]
                put(f"{model}.nearcrit.w{r['wording']}_{r['format']}.abs_m", r["abs_m_final"]); put(f"{model}.nearcrit.w{r['wording']}_{r['format']}.dF0", r["dF0_sym"], nd=2, extra_nd=(3,))
        _text("nearcrit.agree", f"{n_ok}/{n}", n_ok); put("nearcrit.n_cases", n, nd=0)
        old = json.load(open(R / "phase_a" / "mf_dynamics_check.json", encoding="utf-8"))["models"]
        n_old = sum(r["agree"] for m in old.values() for r in m["rows"]); n_all = sum(len(m["rows"]) for m in old.values())
        _text("nearcrit.agree_total_after", f"{n_old + n_ok}/{n_all}", n_old + n_ok)


def review_r2() -> None:
    """Numbers computed for the answers to revision round, R2 (results/phase_a/review_r2.json)."""
    p = R / "phase_a" / "review_r2.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for m, b in d["label_mass"].items():
        put(f"{m}.label_mass.mean", b["mean_all"], nd=4, extra_nd=(3, 2)); put(f"{m}.label_mass.min", b["min_all"], nd=4, extra_nd=(3, 2))
        put(f"{m}.label_mass.frac_below", b["frac_below_0.9_all"], nd=3, extra_nd=(2,))
        _text(f"{m}.label_mass.frac_below.percent", f"{100 * b['frac_below_0.9_all']:.1f}%", 100 * b["frac_below_0.9_all"])
        put(f"{m}.label_mass.cell_mean_min", b["cell_mean_min"], nd=3, extra_nd=(2,)); put(f"{m}.label_mass.cell_mean_max", b["cell_mean_max"], nd=3, extra_nd=(2,))
    for m, by_e in d["list_vs_count"].items():
        for e, v in by_e.items():
            put(f"{m}.count_minus_list.e{e}", v["mean"], v["ci95"]); put(f"{m}.count_minus_list.e{e}.J_list", v["J_list_mean"]); put(f"{m}.count_minus_list.e{e}.J_count", v["J_count_mean"])
            _text(f"{m}.count_minus_list.e{e}.n_negative", f"{v['n_negative']}/{v['n']}", v["n_negative"])
    for m, b in d["memory_k1"].items():
        _text(f"{m}.memk1.own_wins_vs_minus_peers", f"{b['n_own_wins_vs_minus_peers']}/{b['n']}", b["n_own_wins_vs_minus_peers"])
        _text(f"{m}.memk1.own_wins_vs_plus_peers", f"{b['n_own_wins_vs_plus_peers']}/{b['n']}", b["n_own_wins_vs_plus_peers"])
        for k in ("P_k-4_mem+1_min", "P_k-4_mem+1_max", "P_k+4_mem-1_min", "P_k+4_mem-1_max"):
            put(f"{m}.memk1.{k}", b[k])
    for m, by_e in d["h_sign"].items():
        for e, v in by_e.items():
            _text(f"{m}.h_sign.e{e}.n_positive", f"{v['n_positive']}/{v['n']}", v["n_positive"]); put(f"{m}.h_sign.e{e}.min", v["min"]); put(f"{m}.h_sign.e{e}.max", v["max"])
    for m, by_e in d["saturation"].items():
        for e, v in by_e.items():
            _text(f"{m}.saturated.e{e}", f"{v['n_saturated']}/{v['n']}", v["n_saturated"])
    for e, v in d["smollm_J"].items():
        if e != "n_formulations":
            put(f"smollm1.7b.J_gap.e{e}", v, nd=3, extra_nd=(2,))
    for m, by_e in d["dF0_T"].items():
        for e, v in by_e.items():
            for T in ("T1.0", "T0.7"):
                put(f"{m}.dF0_{T}.e{e}.mean", v[T]["mean"]); _text(f"{m}.dF0_{T}.e{e}.n_above_1", f"{v[T]['n_above_1']}/{v[T]['n']}", v[T]["n_above_1"])
    for m, b in d["dF0_k1"].items():
        put(f"{m}.dF0_k1.e4.mean", b["mean_k1"]); put(f"{m}.dF0_k1.e4.mean_k0", b["mean_k0"])
        _text(f"{m}.dF0_k1.e4.n_above_1", f"{b['n_above_1_k1']}/{b['n']}", b["n_above_1_k1"]); _text(f"{m}.dF0_k1.e4.n_above_1_k0", f"{b['n_above_1_k0']}/{b['n']}", b["n_above_1_k0"])
        put(f"{m}.dF0_k1.e4.n_class_changes", b["n_class_changes"], nd=0)
    for key, v in d["covariates"].items():
        m, q = key.split("|")
        put(f"{m}.covariates.{q}.var_base", v["wording_var_base"], nd=3); put(f"{m}.covariates.{q}.var_cov", v["wording_var_cov"], nd=3)
        put(f"{m}.covariates.{q}.p_nll", v["p_nll"], nd=3, extra_nd=(2,)); put(f"{m}.covariates.{q}.p_posix", v["p_posix"], nd=3, extra_nd=(2,))
    for m, rows in d["formulation_table"].items():
        for r in rows:
            tag = f"{m}.formulation.w{r['wording']}_{r['format']}"
            put(f"{tag}.J_gap", r["J_gap"]); put(f"{tag}.h_gap", r["h_gap"])
            if r["nll_mean"] is not None:
                put(f"{tag}.nll", r["nll_mean"])


def review_r2_runs() -> None:
    """Numbers of the three runs proposed by R2 (results/phase_a/review_r2_runs.json,
    generation_check.json, generation_check_40.json)."""
    p = R / "phase_a" / "review_r2_runs.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for m, b in d.get("order", {}).items():
        tag = f"{m}.order"
        put(f"{tag}.h_ab", b["h_mean_ab"]); put(f"{tag}.h_ba", b["h_mean_ba"])
        put(f"{tag}.dh", b["dh"]["mean"], b["dh"]["ci95"]); put(f"{tag}.dJ", b["dJ"]["mean"], b["dJ"]["ci95"])
        if "dh_t" in b:
            put(f"{tag}.dh_t", b["dh_t"]["mean"], b["dh_t"]["ci95"]); put(f"{tag}.dJ_t", b["dJ_t"]["mean"], b["dJ_t"]["ci95"])
        put(f"{tag}.share_order", -b["dh"]["mean"] / b["h_mean_ab"], nd=2)
        _text(f"{tag}.share_order.percent", f"{round(-100 * b['dh']['mean'] / b['h_mean_ab'])}%", round(-100 * b["dh"]["mean"] / b["h_mean_ab"]))
        for k in ("n_sign_flips", "n_halved", "n_toward_A_ba"):
            _text(f"{tag}.{k}", f"{b[k]}/{b['n']}", b[k])
    if "decisional" in d:
        b = d["decisional"]; tag = "qwen0.6b.decisional"
        _text(f"{tag}.own_wins_vs_minus.orig", f"{b['own_wins_vs_minus_peers']['orig']}/{b['n']}", b["own_wins_vs_minus_peers"]["orig"])
        _text(f"{tag}.own_wins_vs_minus.dec", f"{b['own_wins_vs_minus_peers']['dec']}/{b['n']}", b["own_wins_vs_minus_peers"]["dec"])
        _text(f"{tag}.own_wins_vs_plus.orig", f"{b['own_wins_vs_plus_peers']['orig']}/{b['n']}", b["own_wins_vs_plus_peers"]["orig"])
        _text(f"{tag}.own_wins_vs_plus.dec", f"{b['own_wins_vs_plus_peers']['dec']}/{b['n']}", b["own_wins_vs_plus_peers"]["dec"])
        put(f"{tag}.P_min", b["P_k-4_mem+1_dec_min"]); put(f"{tag}.P_max", b["P_k-4_mem+1_dec_max"])
        put(f"{tag}.J.orig", b["J_mean"]["orig"]); put(f"{tag}.J.dec", b["J_mean"]["dec"]); put(f"{tag}.Jmem.orig", b["Jmem_mean"]["orig"]); put(f"{tag}.Jmem.dec", b["Jmem_mean"]["dec"])
        put(f"{tag}.dJ", b["dJ"]["mean"], b["dJ"]["ci95"]); put(f"{tag}.dJmem", b["dJmem"]["mean"], b["dJmem"]["ci95"])
        if "dJ_t" in b:
            put(f"{tag}.dJ_t", b["dJ_t"]["mean"], b["dJ_t"]["ci95"]); put(f"{tag}.dJmem_t", b["dJmem_t"]["mean"], b["dJmem_t"]["ci95"])
    for fname, tag0 in (("generation_check.json", "gen12"), ("generation_check_40.json", "gen40")):
        q = R / "phase_a" / fname
        if not q.exists():
            continue
        g = json.load(open(q, encoding="utf-8"))["models"]
        for m, b in g.items():
            s_ = b["summary"]; tag = f"{m}.{tag0}"
            put(f"{tag}.unparseable", s_["unparseable_rate"], nd=3, extra_nd=(2,)); put(f"{tag}.unparseable.percent_value", 100 * s_["unparseable_rate"], nd=1)
            _text(f"{tag}.unparseable.percent", f"{100 * s_['unparseable_rate']:.1f}%", 100 * s_["unparseable_rate"])
            put(f"{tag}.mean_abs_diff", s_["mean_abs_diff"], nd=3, extra_nd=(2,)); put(f"{tag}.max_abs_diff", s_["max_abs_diff"], nd=3, extra_nd=(2,))
            put(f"{tag}.pearson", s_["pearson"], nd=3, extra_nd=(2,))
            _text(f"{tag}.same_side", f"{s_['n_same_side']}/{s_['n_pairs']}", s_["n_same_side"]); _text(f"{tag}.within_0.1", f"{s_['n_within_0.1']}/{s_['n_pairs']}", s_["n_within_0.1"])
            put(f"{tag}.T", s_["T"], nd=1); put(f"{tag}.n_samples", s_["n_samples"], nd=0)
            for w in sorted({r["wording"] for r in b["rows"]}):
                rows = [r for r in b["rows"] if r["wording"] == w]
                u = 1 - sum(r["n_parsed"] for r in rows) / sum(r["n"] for r in rows)
                put(f"{tag}.unparseable.w{w}.percent_value", 100 * u, nd=1); _text(f"{tag}.unparseable.w{w}.percent", f"{100 * u:.1f}%", 100 * u)


def review_r3() -> None:
    """Numbers of the re-analysis for revision round, R3 (results/phase_a/review_r3.json):
    t intervals (4 df) on the per-wording means next to the cluster bootstrap,
    coverage simulation, additive null, bfloat16 noise on F'(0), wording-level
    Spearman, mixed model with wording x format, subset-test tails, sector-A
    classes on 32 orderings."""
    p = R / "phase_a" / "review_r3.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for m, P in d["paired"].items():
        for name, r in P.items():
            tag = f"{m}.t4.{name}"
            put(tag, r["t4"]["mean"], r["t4"]["ci95"]); _text(f"{tag}.n_negative", f"{r['t4']['n_negative']}/{r['t4']['n_wordings']}", r["t4"]["n_negative"])
            put(f"{m}.cboot.{name}", r["cluster_boot"]["mean"], r["cluster_boot"]["ci95"])
            for i, x in enumerate(r["t4"]["per_wording"]):
                put(f"{tag}.w{i}", x)
    for k, v in d["coverage"].items():
        put(f"coverage.{k}.cluster_boot", v["cluster_percentile_boot"], nd=3, extra_nd=(2,)); put(f"coverage.{k}.t4", v["t4"], nd=3, extra_nd=(2,))
        put(f"coverage.{k}.sd_between", v["sd_between"]); put(f"coverage.{k}.sd_within", v["sd_within"])
    for m, v in d["additive_null"].items():
        tag = f"{m}.additive_null"
        put(f"{tag}.spearman_increment", v["spearman_increment_vs_jump7"]); put(f"{tag}.p_increment", v["p"], nd=2, extra_nd=(3,))
        put(f"{tag}.rho_obs", v["spearman_exponent_obs"]); put(f"{tag}.rho_null", v["spearman_exponent_null"])
        put(f"{tag}.split_obs", v["median_split_obs"]); put(f"{tag}.split_null", v["median_split_null"])
        put(f"{tag}.increment_weak", v["increment_weak_half"]); put(f"{tag}.increment_strong", v["increment_strong_half"]); put(f"{tag}.b_common", v["b_common"], nd=3, extra_nd=(2,))
        put(f"{tag}.n", v["n"], nd=0)
    b = d["bf16_dF0_e4"]
    put("qwen1.7b.bf16_classes.e4.sd", b["sd_used"])
    c = b["counts"]; _text("qwen1.7b.bf16_classes.e4.triple", f"{c['ordered']}/{c['disordered']}/{c['undetermined']}")
    copy = [r for r in b["rows"] if r["first_peer_copy_frac"] >= 0.999 and r["class_bf16"] == "ordered"]   # first-peer copy: marginal by construction
    _text("qwen1.7b.bf16_classes.e4.defensible_triple", f"{c['ordered'] - len(copy)}/{c['disordered']}/{c['undetermined'] + len(copy)}")
    for k in c:
        _text(f"qwen1.7b.bf16_classes.e4.{k}", f"{c[k]}/10", c[k])
    for r in b["rows"]:
        tag = f"qwen1.7b.bf16_classes.e4.w{r['wording']}_{r['format']}"
        put(f"{tag}.dF0", r["dF0"], nd=3, extra_nd=(2,)); put(f"{tag}.frac_above_1", r["frac_above_1"]); put(f"{tag}.copy", r["first_peer_copy_frac"])
    for m, v in d["wording_level"].items():
        put(f"{m}.wording_rho", v["rho"]); put(f"{m}.wording_rho.p_exact", v["p_exact"], nd=3, extra_nd=(2,))
        for k, s_ in v["median_split"].items():
            put(f"{m}.median_split.{k}", s_["diff"], s_["ci"]); _text(f"{m}.median_split.{k}.passes", "passes" if s_["passes"] else "fails", s_["passes"])
    for m, v in d["mixed_wf"].items():
        for k in ("format", "interaction"):
            put(f"{m}.mixed_wf.{k}.p", v[k]["p"], nd=3, extra_nd=(2,)); put(f"{m}.mixed_wf.{k}.LR", v[k]["LR"]); put(f"{m}.mixed_wf.{k}.df", v[k]["df"], nd=0)
        if v["vc_wording_format"] is not None:
            put(f"{m}.mixed_wf.vc", v["vc_wording_format"], nd=3); put(f"{m}.mixed_wf.residual", v["residual"], nd=3)
    for e, v in d["shared_orderings"].items():
        _text(f"shared_orderings.e{e}", "identical" if v["identical"] else "different", v["identical"])
    for m, blk in d["subset_tails"].items():
        for e, v in blk.items():
            put(f"{m}.subset_tail.e{e}", v["frac_below_observed"], nd=3, extra_nd=(2,)); put(f"{m}.subset_sd_shared.e{e}", v["sd_shared"], nd=3, extra_nd=(2,))
    for m, blk in d["sectorA_32"].items():
        for e, v in blk.items():
            c = v["counts"]; _text(f"{m}.sectorA32.e{e}.triple", f"{c['bistable']}/{c['field_dominated']}/{c['disordered']}")
            _text(f"{m}.sectorA32.e{e}.n_firm", f"{v['n_firm_95']}/{len(v['rows'])}", v["n_firm_95"])


def review_r3_newwordings() -> None:
    """Five new wordings (results/phase_a/review_r3_newwordings.json): t intervals over
    ten, five new and five original wordings for the exposure and memory differences."""
    p = R / "phase_a" / "review_r3_newwordings.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for m, res in d["models"].items():
        for name in ("J.e4-e2", "J.e7-e2", "J.k3-k0"):
            for grp in ("ten_wordings", "new_five", "original_five"):
                r = res[name][grp]
                if r.get("ci95"):
                    put(f"{m}.nw.{name}.{grp}", r["mean"], r["ci95"]); _text(f"{m}.nw.{name}.{grp}.n_negative", f"{r['n_negative']}/{r['n_wordings']}", r["n_negative"])
        put(f"{m}.nw.J_e4.new", res["J_mean_e4"]["new"]); put(f"{m}.nw.J_e4.orig", res["J_mean_e4"]["orig"])
        for r in res["formulations_new_e4"]:
            tag = f"{m}.nw.formulation.w{r['wording']}_{r['format']}"
            put(f"{tag}.J_gap", r["J_gap"])
            if r["h_gap"] is not None:
                put(f"{tag}.h_gap", r["h_gap"])
        vals = [r["J_gap"] for r in res["formulations_new_e4"]]
        put(f"{m}.nw.J_e4.new_min", min(vals)); put(f"{m}.nw.J_e4.new_max", max(vals))


def toy_paraphrases() -> None:
    p = R / "toy" / "paraphrases.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    summ = d.get("summary")
    if isinstance(summ, dict):
        def walk(o, prefix):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{prefix}.{k}")
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                put(f"toy.paraphrases{prefix}", float(o))
        walk(summ, "")


def review_newwordings_ba() -> None:
    """Order of mention on the five new wordings (results/phase_a/review_newwordings_ba.json):
    h in the A-first and B-first versions, paired difference d with a t interval over wordings,
    share d / h(A-first), on the new, the original and the ten wordings."""
    p = R / "phase_a" / "review_newwordings_ba.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for m, res in d["models"].items():
        for g in ("new", "original", "ten"):
            b = res[g]; tag = f"{m}.order_{g}"
            put(f"{tag}.h_A_first", b["h_mean_A_first"]); put(f"{tag}.h_B_first", b["h_mean_B_first"])
            put(f"{tag}.d", b["d_t"]["mean"], b["d_t"]["ci95"])
            put(f"{tag}.share", b["share"], nd=2)
            _text(f"{tag}.share.percent", f"{round(100 * b['share'])}%", round(100 * b["share"]))
            _text(f"{tag}.n_wordings_positive", f"{b['n_wordings_positive']}/{b['n_wordings']}", b["n_wordings_positive"])
            _text(f"{tag}.n_toward_A_when_B_first", f"{b['n_toward_A_when_B_first']}/{b['n_formulations']}", b["n_toward_A_when_B_first"])
            for r in b["rows"]:
                put(f"{tag}.w{r['wording']}.{r['format']}.h_A_first", r["h_A_first"]); put(f"{tag}.w{r['wording']}.{r['format']}.h_B_first", r["h_B_first"])
                put(f"{tag}.w{r['wording']}.{r['format']}.d", r["d"])
            for w, v in b["d_t"]["per_wording"].items():
                put(f"{tag}.w{w}.d_mean", v)


def review_labels_xy() -> None:
    """Token-identity test of the label bias (results/phase_a/review_labels_xy.json): with the
    labels X/Y in both naming orders, per model and wording set, h toward X in the X-first and
    Y-first versions, d = h(X first) - h(Y first) and m = mean over the two orders with t
    intervals over wordings, and the A/B reference of the same wordings."""
    p = R / "phase_a" / "review_labels_xy.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for m, res in d["models"].items():
        for g in ("original", "new", "ten"):
            b = res[g]; tag = f"{m}.labels_xy_{g}"
            put(f"{tag}.h_X_first", b["h_mean_X_first"]); put(f"{tag}.h_Y_first", b["h_mean_Y_first"]); put(f"{tag}.h_both_orders", b["h_mean_both_orders"])
            put(f"{tag}.d", b["d_t"]["mean"], b["d_t"]["ci95"]); put(f"{tag}.m", b["m_t"]["mean"], b["m_t"]["ci95"])
            put(f"{tag}.J_X_first", b["J_mean_X_first"]); put(f"{tag}.J_Y_first", b["J_mean_Y_first"])
            _text(f"{tag}.n_toward_X_when_X_first", f"{b['n_toward_X_when_X_first']}/{b['n_formulations']}", b["n_toward_X_when_X_first"])
            _text(f"{tag}.n_toward_X_when_Y_first", f"{b['n_toward_X_when_Y_first']}/{b['n_formulations']}", b["n_toward_X_when_Y_first"])
            _text(f"{tag}.verdict", ", ".join(b["verdict"]))
            for w, v in b["d_t"]["per_wording"].items():
                put(f"{tag}.w{w}.d", v)
            for w, v in b["m_t"]["per_wording"].items():
                put(f"{tag}.w{w}.m", v)
        for g, b in res["AB_reference"].items():
            tag = f"{m}.labels_ab_{g}"
            put(f"{tag}.h_first", b["h_mean_first"]); put(f"{tag}.h_second", b["h_mean_second"])
            put(f"{tag}.h_both_orders", 0.5 * (b["h_mean_first"] + b["h_mean_second"]))
            put(f"{tag}.J_first", b["J_mean_first"]); put(f"{tag}.J_second", b["J_mean_second"])
        ab = res["AB_reference"]
        put(f"{m}.labels_ab_ten.h_both_orders", 0.25 * (ab["original"]["h_mean_first"] + ab["original"]["h_mean_second"] + ab["new"]["h_mean_first"] + ab["new"]["h_mean_second"]))


def _text(name: str, text: str, value=None) -> None:
    V[name] = {"value": value, "ci": None, "text": text}


def orderings_check() -> None:
    """Test 1 of §24: 8 against 32 orderings (results/phase_a/orderings_check.json)
    and the shared-ordering subset test (orderings_subset.json). The 32-ordering
    values of z_bar * J_gap and F'(0) are the ones the paper quotes at 10 and 30
    peers (§20 decision of 2026-09-15); the F'(0) classification counts use the
    §24 rule: above 1.05 ordered, within 0.05 of 1 critical, below 0.95 disordered."""
    p = R / "phase_a" / "orderings_check.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for model, blk in d["models"].items():
        for e, r in blk.items():
            tag = f"{model}.ord32.e{e}"
            for q, s in r["summary"].items():
                if not isinstance(s, dict):
                    continue
                nd = 3 if q == "r2_linear_k" else 2
                put(f"{tag}.{q}.mean8", s["mean8"], nd=nd)
                put(f"{tag}.{q}.mean32", s["mean32"], nd=nd)
                put(f"{tag}.{q}.diff32_8", s["paired_diff_32_minus_8"], s["ci95"], nd=nd, extra_nd=(3,))
                _text(f"{tag}.{q}.inside", s["inside_ci8"])
            _text(f"{tag}.dF0.class_flips", r["summary"]["dF0_class_flips"])
            put(f"{tag}.J_gap.mean32", r["summary"]["zJ_gap"]["mean32"] / int(e))
            v32 = [row["dF0_sym"]["v32"] for row in r["rows"]]
            above = sum(v > 1.05 for v in v32); crit = sum(abs(v - 1) <= 0.05 for v in v32); below = len(v32) - above - crit
            _text(f"{tag}.dF0.n_above", f"{above}/{len(v32)}", above)
            _text(f"{tag}.dF0.n_critical", f"{crit}/{len(v32)}", crit)
            _text(f"{tag}.dF0.n_below", f"{below}/{len(v32)}", below)
            put(f"{tag}.dF0.min32", float(min(v32))); put(f"{tag}.dF0.max32", float(max(v32)))
    p = R / "phase_a" / "orderings_subset.json"
    if p.exists():
        d = json.load(open(p, encoding="utf-8"))
        for model, blk in d["models"].items():
            for e, r in blk.items():
                tag = f"{model}.ord_subset.e{e}"
                put(f"{tag}.observed_8_minus_32", r["observed_8_minus_32"], nd=3, extra_nd=(2,))
                for mode in ("shared", "independent"):
                    put(f"{tag}.{mode}.centre", r[mode]["centre"], [r[mode]["q025"], r[mode]["q975"]], nd=3, extra_nd=(2,))
                    put(f"{tag}.{mode}.sd", r[mode]["sd"], nd=3, extra_nd=(2,))
                put(f"{tag}.sd_ratio", r["sd_ratio_shared_over_independent"])


def meanfield_stability() -> None:
    """Test 4 of §24: F'(0) of the mean-field map on the 8-ordering tables
    (results/phase_a/meanfield_stability.json), per model and exposure."""
    p = R / "phase_a" / "meanfield_stability.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for model, rows in d["models"].items():
        for e in sorted({r["exposure"] for r in rows}):
            sub = [r for r in rows if r["exposure"] == e]
            tag = f"{model}.mf8.e{e}"
            put(f"{tag}.dF0.mean", float(np.mean([r["dF0_sym"] for r in sub])))
            put(f"{tag}.dF0.min", float(min(r["dF0_sym"] for r in sub))); put(f"{tag}.dF0.max", float(max(r["dF0_sym"] for r in sub)))
            _text(f"{tag}.dF0.n_above_1", f"{sum(r['dF0_sym'] > 1 for r in sub)}/{len(sub)}")
            _text(f"{tag}.zJ.n_above_1", f"{sum(r['zJ_gap'] > 1 for r in sub)}/{len(sub)}")


def format_check() -> None:
    """Test 2 of §24: twenty formulations, four list formats
    (results/phase_a/format_check.json)."""
    p = R / "phase_a" / "format_check.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for model, r in d["models"].items():
        tag = f"{model}.fmt"
        for f, by_e in r["J_gap_mean_by_exposure_format"].items():
            for e, v in by_e.items():
                put(f"{tag}.J_gap.{f}.e{e}", v)
        for g, blk in r["P2a"].items():
            for k2, v in blk.items():
                put(f"{tag}.paired.{g}.{k2}", v["mean"], v["ci95"])
                V[f"{tag}.paired.{g}.{k2}"]["n"] = v["n"]
        for g, b in r["P2b"].items():
            if b.get("rho") is not None:
                put(f"{tag}.spearman.{g}.rho", b["rho"]); put(f"{tag}.spearman.{g}.p", b["p"], nd=3)
                _text(f"{tag}.spearman.{g}.n", str(b["n"]), b["n"])
        c = r["P2c"]
        if c and "format" in c:
            put(f"{tag}.lrt.format.p", c["format"]["p"], nd=3, extra_nd=(2,))
            put(f"{tag}.lrt.interaction.p", c["interaction"]["p"], nd=3, extra_nd=(2,))
            _text(f"{tag}.lrt.n_rows", str(c["n_rows"]), c["n_rows"])


def second_family() -> None:
    """Test 3 of §24: SmolLM2-1.7B-Instruct (results/phase_a/second_family.json)."""
    p = R / "phase_a" / "second_family.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for model, r in d["models"].items():
        if not model.startswith("smollm"):
            continue
        tag = f"{model}"
        for k2, v in r["P3_paired"].items():
            put(f"{tag}.paired.{k2}", v["mean"], v["ci95"])
        s = r["shape"]
        for e in r["exposures"]:
            put(f"{tag}.J_gap.e{e}", s["J_gap"][str(e)]); put(f"{tag}.zJ_gap.e{e}", s["zJ_gap"][str(e)])
            put(f"{tag}.jump.e{e}.mean", s["jump"][str(e)]["mean"])
        for e, q in s["ratio"].items():
            put(f"{tag}.ratio.e{e}", q["mean"], q["ci95"]); put(f"{tag}.ratio.e{e}.pred_count", q["pred_count"])
        if "exponent_e_ge_ref" in s:
            x = s["exponent_e_ge_ref"]
            put(f"{tag}.exponent_e_ge_ref", x["mean"], x["ci95"])
            put(f"{tag}.exponent_e_ge_ref.min", x["min"]); put(f"{tag}.exponent_e_ge_ref.max", x["max"])
        c = s["collapse"]
        put(f"{tag}.collapse.coef_fraction", c["coef_fraction_in_both"]["mean"], c["coef_fraction_in_both"]["ci95"])
        put(f"{tag}.collapse.coef_count", c["coef_count_in_both"]["mean"], c["coef_count_in_both"]["ci95"], nd=3, extra_nd=(2,))
        _text(f"{tag}.collapse.frac_beats_count", c["frac_beats_count_aic"])


def h10_test() -> None:
    """H10 as a declared test (results/phase_a/h10_test.json): exponent against
    strength per formulation, paired split, ceiling control; three models."""
    p = R / "phase_a" / "h10_test.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    for model, r in d["models"].items():
        tag = f"{model}.h10"
        _text(f"{tag}.n_formulations", str(r["n_formulations"]), r["n_formulations"])
        for k2, v in r["a"].items():
            put(f"{tag}.{k2}.rho", v["rho"], v["ci95_cluster"][:2]); put(f"{tag}.{k2}.p", v["p"], nd=3)
            _text(f"{tag}.{k2}.n", str(v["n"]), v["n"])
        for k2, v in r["b"].items():
            put(f"{tag}.split.{k2}.diff", v["diff_weak_minus_strong"], v["ci95_cluster"][:2])
            put(f"{tag}.split.{k2}.mean_weak", v["mean_weak"]); put(f"{tag}.split.{k2}.mean_strong", v["mean_strong"])
            _text(f"{tag}.split.{k2}.n", f"{v['n_weak']} contro {v['n_strong']}")
        c = r["c"]
        put(f"{tag}.ceiling.sd_jump7", c["sd_jump7"]); put(f"{tag}.ceiling.sd_jump30", c["sd_jump30"])
        put(f"{tag}.ceiling.ratio", c["ratio_sd30_over_sd7"])


def memory_by_formulation() -> None:
    """Per-formulation J_gap with and without memory (k = 3 against k = 0) at
    4 peers, list and count channels: the count channel has five formulations
    and claims.md B8 quotes each difference."""
    a = json.load(open(R / "phase_a" / "aggregate.json", encoding="utf-8"))
    rows = [r for r in a["rows"] if r["T"] == 1.0 and r["topology"] == "all" and r["exposure"] == 4]
    for model in sorted({r["model"] for r in rows}):
        for ch in ("list", "count"):
            k0 = {(r["wording"], r["format"]): r["J_gap"] for r in rows if r["model"] == model and r["channel"] == ch and r["k_mem"] == 0}
            k3 = {(r["wording"], r["format"]): r["J_gap"] for r in rows if r["model"] == model and r["channel"] == ch and r["k_mem"] == 3}
            for key in sorted(k0):
                if key in k3:
                    tag = f"{model}.{ch}.e4.w{key[0]}_{key[1]}"
                    put(f"{tag}.J_gap.k0", k0[key]); put(f"{tag}.J_gap.k3", k3[key]); put(f"{tag}.J_gap.k3_minus_k0", k3[key] - k0[key])


def f0_decomposition() -> None:
    """Decomposition of e J~ - F'(0) at 4 peers (results/phase_a/F0_decomposition_e4.json,
    round 5 of 2026-10-24): per model the means of the seven slopes (linear, additive,
    additive with bias, odd part in logit, even part through the symmetrisation in
    probability, dispersion without bias, paper), the counts above / within 0.02 of /
    below 1, the sequential steps in both orders with the Shapley mean and the
    interaction, the share of the residual due to the bias, the sech^2(h~) range, the
    sensitivity to a Gaussian perturbation of the gaps, and every per-formulation slope."""
    p = R / "phase_a" / "F0_decomposition_e4.json"
    if not p.exists():
        return
    d = json.load(open(p, encoding="utf-8"))
    put("f0dec.margin", d["margin"])
    keys = ("eJ", "F0_add", "F0_add_bias", "F0_odd", "F0_even", "F0_disp", "F0_paper")
    for m, res in d["models"].items():
        s = res["summary"]; tag = f"{m}.f0dec"
        n = len(res["rows"])
        for k in keys:
            put(f"{tag}.{k}.mean", s["means"][k])
            c = s["counts"][k]
            _text(f"{tag}.{k}.n_above_1", f"{c['above_1']}/{n}", c["above_1"])
            _text(f"{tag}.{k}.n_within_margin", f"{c['within_margin']}/{n}", c["within_margin"])
            _text(f"{tag}.{k}.n_below_1", f"{c['below_1']}/{n}", c["below_1"])
            _text(f"{tag}.{k}.n_above_1_point", f"{c['above_1_point']}/{n}", c["above_1_point"])
            _text(f"{tag}.{k}.triple", f"{c['above_1']}/{c['within_margin']}/{c['below_1']}")
        c = s["contributions_mean"]
        put(f"{tag}.step.saturation", c["saturation"]); put(f"{tag}.step.shape_k", c["shape_k"])
        put(f"{tag}.residual", c["residual_after_saturation"])
        put(f"{tag}.residual_over_additive", c["residual_after_saturation"] / s["means"]["F0_add"])
        for order in ("bias_then_disp", "disp_then_bias"):
            put(f"{tag}.step.{order}.bias", c[order]["bias"]); put(f"{tag}.step.{order}.dispersion", c[order]["dispersion"])
        put(f"{tag}.step.shapley.bias", c["shapley"]["bias"]); put(f"{tag}.step.shapley.dispersion", c["shapley"]["dispersion"])
        put(f"{tag}.step.interaction", c["interaction_bias_x_disp"])
        b = s["bias_share_of_residual"]
        put(f"{tag}.bias_share.bias_then_disp", b["bias_then_disp"]); put(f"{tag}.bias_share.disp_then_bias", b["disp_then_bias"])
        _text(f"{tag}.bias_share.order_robust", "yes" if b["order_robust"] else "no", b["order_robust"])
        _text(f"{tag}.n_strong", str(s["n_strong_odd_above_0.9"]), s["n_strong_odd_above_0.9"])
        if s["strong_F0_odd_range"]:
            put(f"{tag}.strong.F0_odd.min", s["strong_F0_odd_range"][0]); put(f"{tag}.strong.F0_odd.max", s["strong_F0_odd_range"][1])
            put(f"{tag}.strong.F0_paper.min", s["strong_F0_paper_range"][0]); put(f"{tag}.strong.F0_paper.max", s["strong_F0_paper_range"][1])
        put(f"{tag}.sech2_h.min", s["sech2_h_range"][0]); put(f"{tag}.sech2_h.max", s["sech2_h_range"][1])
        se = s["sensitivity"]
        put(f"{tag}.noise.s", se["s"]); put(f"{tag}.noise.std_F0_max", se["std_F0_max"], nd=3, extra_nd=(2,))
        put(f"{tag}.noise.bound_s_over_4", se["s"] / 4, nd=3, extra_nd=(2,))
        for q in ("n_firm_above_1_odd", "n_firm_below_1_odd", "n_firm_above_1_paper", "n_firm_below_1_paper"):
            _text(f"{tag}.noise.{q}", f"{se[q]}/{n}", se[q])
        for r in res["rows"]:
            rt = f"{tag}.w{r['wording']}_{r['format']}"
            put(f"{rt}.h", r["h"]); put(f"{rt}.sech2_h", r["sech2_h"])
            for k in keys:
                put(f"{rt}.{k}", r[k], nd=3, extra_nd=(2,))
            put(f"{rt}.noise.std_F0", r["sensitivity"]["std_F0_linear"], nd=3, extra_nd=(2,))
            put(f"{rt}.noise.frac_above_1_odd", r["sensitivity"]["frac_above_1_odd"]); put(f"{rt}.noise.frac_above_1_paper", r["sensitivity"]["frac_above_1_paper"])
            for k, v in r["even_part"].items():
                put(f"{rt}.even_part.k{k}", v)


def main() -> int:
    constants(); phase_a_cells(); exposure_shape(); lattice(); stubborn(); memory_curve(); field_ratio(); phase0(); bf16_check(); toy_paraphrases()
    orderings_check(); meanfield_stability(); format_check(); second_family(); h10_test(); memory_by_formulation(); fp32_check_17b(); fp32_recheck_quantities(); fp32_recheck_full(); dF0_classes(); review_r1(); review_r2(); review_r2_runs(); review_r3(); review_r3_newwordings(); review_newwordings_ba(); review_labels_xy(); f0_decomposition()
    out = Path("paper"); out.mkdir(exist_ok=True)
    write_json(out / "numbers.json", {"manifest": build_manifest({"script": "paper_numbers"}, []),
                                      "values": V, "strings": sorted(S)})
    with open(out / "numbers.md", "w", encoding="utf-8") as fh:
        fh.write("# Numeri ammessi nel paper (generati da scripts/paper_numbers.py)\n\n| nome | valore |\n|---|---|\n")
        for k in sorted(V):
            fh.write(f"| `{k}` | {V[k]['text']} |\n")
    print(f"{len(V)} named values, {len(S)} admissible number strings -> paper/numbers.json, paper/numbers.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
