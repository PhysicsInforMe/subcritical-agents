#!/usr/bin/env python
"""Zero-cost analyses answering the schede of revision round, R2 (the revision round (not included)),
all from existing gap tables and result files. Output: results/phase_a/review_r2.json
and paper/appendix_prompts.md (the five wordings and rendered examples).

(a) R2-01, R2-08  Probability mass on the two label tokens after the JSON prefix,
                  exp(logp_plus) + exp(logp_minus), per model and list cell at
                  k_mem = 0: mean, minimum, fraction below 0.9.
(b) R2-03         Paired difference J_gap(count) - J_gap(list, comma) per wording,
                  at 2, 4, 7 peers, bootstrap over the five wordings.
(c) R2-05         At 4 peers and k_mem = 1: P(+ | k = -4, mem = +1) and
                  P(+ | k = +4, mem = -1) per formulation, at T = 1.
(d) R2-06         Sign of the label bias h_gap (intercept over two of the gap
                  regression in sector A) per formulation at k_mem = 0.
(e) R2-07         Saturation of the unanimity jump: formulations with
                  P(+ | k = +e) > 0.99 and P(- | k = -e) > 0.99 at T = 1.
(f) R2-08         J_gap of SmolLM2 by exposure (second_family.json).
(g) R2-09         F'(0) of the symmetrised map at T = 0.7 against T = 1 on the
                  cells of Table 3, with the count of formulations above 1.
(h) R2-04         F'(0) of the mean-field map with memory k = 1 at 4 peers: the
                  memory is the agent's own previous state, so the round map is
                  m -> E_{s ~ (1+m)/2, k ~ 2Bin(e,(1+m)/2)-e}[2 P_sym(+ | k, s) - 1]
                  on the symmetrised table P_sym(+|k,mem) = [P_A(+|k,mem) + 1 - P_A(+|-k,-mem)]/2.
(i) R2-10         Covariates of the formulation (covariates.json), and per
                  formulation the mean prompt NLL and J_gap at 4 peers.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_format_check import boot_mean  # noqa: E402
from scripts.phase_a_meanfield_stability import mf_map, policy_from_entries  # noqa: E402
from src.common import build_manifest, write_json  # noqa: E402

R = Path("results/phase_a")
MODELS = ("qwen0.6b", "qwen1.7b", "smollm1.7b")


def sig(x: float, T: float = 1.0) -> float:
    return float(1.0 / (1.0 + np.exp(-x / T)))


def mean_gaps(entries: list[dict]) -> dict[int, float]:
    acc: dict[int, list[float]] = {}
    for x in entries:
        acc.setdefault(x["k"], []).append(x["gap"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def fit_Jh(g: dict[int, float]) -> tuple[float, float]:
    ks = np.array(sorted(g), float)
    slope, inter = np.polyfit(ks, np.array([g[int(k)] for k in ks]), 1)
    return float(slope / 2), float(inter / 2)


def cell(model: str, channel: str, k: int, e: int, suffix: str) -> Path | None:
    hits = glob.glob(str(R / f"A_{model}_{channel}_k{k}_e{e}_Tan_f0_{suffix}_N*/gaps.json"))
    return Path(hits[0]) if hits else None


def tables(path: Path) -> list[dict]:
    return json.load(open(path, encoding="utf-8"))["tables"]


def main() -> int:
    out: dict = {}
    # (a) label mass
    out["label_mass"] = {}
    for m in MODELS:
        per_cell = {}
        allm = []
        # the cells of the exposure results that existed at the review (suffixes all, allf2, allo32), not the
        # cells added later for the reviews (allba, allnw); explicit list instead of the glob "all*" (revision R4-01)
        for path in sorted(p for suf in ("all", "allf2", "allo32") for p in glob.glob(str(R / f"A_{m}_list_k0_e*_Tan_f0_{suf}_N*/gaps.json"))):
            name = Path(path).parent.name
            mass = np.array([np.exp(x["logp_plus"]) + np.exp(x["logp_minus"]) for t in tables(Path(path)) for x in t["entries"]])
            allm.append(mass)
            per_cell[name] = {"mean": float(mass.mean()), "min": float(mass.min()), "frac_below_0.9": float((mass < 0.9).mean()), "n": int(mass.size)}
        a = np.concatenate(allm)
        out["label_mass"][m] = {"cells": per_cell, "mean_all": float(a.mean()), "min_all": float(a.min()), "frac_below_0.9_all": float((a < 0.9).mean()),
                                "cell_mean_min": float(min(v["mean"] for v in per_cell.values())), "cell_mean_max": float(max(v["mean"] for v in per_cell.values()))}
        print(f"(a) {m}: mass mean {a.mean():.4f} min {a.min():.4f} frac<0.9 {(a < 0.9).mean():.3f}; cell means {min(v['mean'] for v in per_cell.values()):.3f}-{max(v['mean'] for v in per_cell.values()):.3f}")
    # (b) list vs count
    rng = np.random.default_rng(41)
    out["list_vs_count"] = {}
    for m in ("qwen0.6b", "qwen1.7b"):
        out["list_vs_count"][m] = {}
        for e in (2, 4, 7):
            jl = {t["wording"]: fit_Jh(mean_gaps(t["entries"]))[0] for t in tables(cell(m, "list", 0, e, "all")) if t["format"] == "comma"}
            jc = {t["wording"]: fit_Jh(mean_gaps(t["entries"]))[0] for t in tables(cell(m, "count", 0, e, "all"))}
            ws = sorted(set(jl) & set(jc))
            d = np.array([jc[w] - jl[w] for w in ws])
            res = boot_mean(d, np.array(ws), rng)
            res.update({"J_list_mean": float(np.mean([jl[w] for w in ws])), "J_count_mean": float(np.mean([jc[w] for w in ws])),
                        "n_negative": int((d < 0).sum()), "per_wording": {str(w): float(x) for w, x in zip(ws, d)}})
            out["list_vs_count"][m][str(e)] = res
            print(f"(b) {m} e={e}: count - list {res['mean']:+.2f} [{res['ci95'][0]:+.2f}, {res['ci95'][1]:+.2f}], negative {res['n_negative']}/{len(ws)}; J list {res['J_list_mean']:.2f} count {res['J_count_mean']:.2f}")
    # (c) memory k = 1 at 4 peers
    out["memory_k1"] = {}
    for m in ("qwen0.6b", "qwen1.7b"):
        rows = []
        for t in tables(cell(m, "list", 1, 4, "all")):
            p_own_vs_peers = [sig(x["gap"]) for x in t["entries"] if x["k"] == -4 and x["memory"] == [1]]      # remembers +, four peers -
            p_peers_vs_own = [sig(x["gap"]) for x in t["entries"] if x["k"] == 4 and x["memory"] == [-1]]      # remembers -, four peers +
            rows.append({"wording": t["wording"], "format": t["format"], "P_plus_k-4_mem+1": float(np.mean(p_own_vs_peers)), "P_plus_k+4_mem-1": float(np.mean(p_peers_vs_own))})
        a = np.array([r["P_plus_k-4_mem+1"] for r in rows]); b = np.array([r["P_plus_k+4_mem-1"] for r in rows])
        out["memory_k1"][m] = {"rows": rows, "n_own_wins_vs_minus_peers": int((a > 0.5).sum()), "n_own_wins_vs_plus_peers": int((b < 0.5).sum()),
                               "P_k-4_mem+1_min": float(a.min()), "P_k-4_mem+1_max": float(a.max()), "P_k+4_mem-1_min": float(b.min()), "P_k+4_mem-1_max": float(b.max()), "n": len(rows)}
        print(f"(c) {m}: P(+|k=-4,mem=+) {a.min():.2f}-{a.max():.2f}, own wins in {(a > 0.5).sum()}/{len(rows)}; P(+|k=+4,mem=-) {b.min():.2f}-{b.max():.2f}, own wins in {(b < 0.5).sum()}/{len(rows)}")
    # (d) sign of the label bias
    out["h_sign"] = {}
    for m in ("qwen0.6b", "qwen1.7b"):
        out["h_sign"][m] = {}
        for e in (2, 4, 7):
            hs = {f"w{t['wording']}_{t['format']}": fit_Jh(mean_gaps(t["entries"]))[1] for t in tables(cell(m, "list", 0, e, "all"))}
            v = np.array(list(hs.values()))
            out["h_sign"][m][str(e)] = {"n_positive": int((v > 0).sum()), "n": int(v.size), "min": float(v.min()), "max": float(v.max()), "per_formulation": hs}
            print(f"(d) {m} e={e}: h toward A in {(v > 0).sum()}/{v.size}, range {v.min():+.2f} to {v.max():+.2f}")
    # (e) saturation of the jump
    out["saturation"] = {}
    for m in MODELS:
        out["saturation"][m] = {}
        for e in (2, 4, 7, 10, 15, 20, 30):
            n_sat = n = 0
            for suffix in ("all", "allf2"):
                p = cell(m, "list", 0, e, suffix)
                if p is None:
                    continue
                for t in tables(p):
                    g = mean_gaps(t["entries"])
                    if e in g and -e in g:
                        n += 1; n_sat += (sig(g[e]) > 0.99) and (1 - sig(g[-e]) > 0.99)
            if n:
                out["saturation"][m][str(e)] = {"n_saturated": n_sat, "n": n}
        print(f"(e) {m}: saturated/total by e " + ", ".join(f"{e}: {v['n_saturated']}/{v['n']}" for e, v in out["saturation"][m].items()))
    # (f) SmolLM2 J by exposure
    sf = json.load(open(R / "second_family.json", encoding="utf-8"))["models"]["smollm1.7b"]["J_gap_by_formulation"]
    es = sorted({int(e) for v in sf.values() for e in v})
    out["smollm_J"] = {str(e): float(np.mean([v[str(e)] for v in sf.values()])) for e in es}
    out["smollm_J"]["n_formulations"] = len(sf)
    print("(f) smollm J by e: " + ", ".join(f"{e}: {out['smollm_J'][str(e)]:.3f}" for e in es) + f" ({len(sf)} formulations)")
    # (g) F'(0) at T = 0.7 on the cells of Table 3
    out["dF0_T"] = {}
    for m in ("qwen0.6b", "qwen1.7b"):
        out["dF0_T"][m] = {}
        for e in (2, 4, 7, 10, 30):
            p = cell(m, "list", 0, e, "allo32") if e >= 7 else cell(m, "list", 0, e, "all")
            res = {}
            for T in (1.0, 0.7):
                d0 = []
                for t in tables(p):
                    _, ps = policy_from_entries(t["entries"], T)
                    _, dF = mf_map(ps, e); d0.append(float(dF(0.0)))
                res[f"T{T}"] = {"mean": float(np.mean(d0)), "n_above_1": int(sum(x > 1 for x in d0)), "n": len(d0)}
            out["dF0_T"][m][str(e)] = {"cell": p.parent.name, **res}
            print(f"(g) {m} e={e}: F'(0) T=1 {res['T1.0']['mean']:.2f} ({res['T1.0']['n_above_1']}/{res['T1.0']['n']} above 1); T=0.7 {res['T0.7']['mean']:.2f} ({res['T0.7']['n_above_1']}/{res['T0.7']['n']})")
    # (h) F'(0) with memory k = 1 at 4 peers
    out["dF0_k1"] = {}
    e = 4
    for m in ("qwen0.6b", "qwen1.7b"):
        rows = []
        k0 = {f"w{t['wording']}_{t['format']}": float(mf_map(policy_from_entries(t["entries"], 1.0)[1], e)[1](0.0)) for t in tables(cell(m, "list", 0, e, "all"))}
        for t in tables(cell(m, "list", 1, e, "all")):
            acc: dict[tuple, list[float]] = {}
            for x in t["entries"]:
                acc.setdefault((x["k"], x["memory"][0]), []).append(sig(x["gap"]))
            pA = {key: float(np.mean(v)) for key, v in acc.items()}
            psym = {(k, s): 0.5 * (pA[(k, s)] + 1 - pA[(-k, -s)]) for (k, s) in pA}
            ks = 2 * np.arange(e + 1) - e

            def F(mm: float) -> float:
                w = binom.pmf(np.arange(e + 1), e, (1 + mm) / 2)
                q = (1 + mm) / 2
                return float(2 * sum(w[i] * (q * psym[(int(k), 1)] + (1 - q) * psym[(int(k), -1)]) for i, k in enumerate(ks)) - 1)

            h = 1e-4; d0 = (F(h) - F(-h)) / (2 * h)
            name = f"w{t['wording']}_{t['format']}"
            rows.append({"formulation": name, "dF0_k1": float(d0), "dF0_k0": k0.get(name)})
        n1 = sum(r["dF0_k1"] > 1 for r in rows); n0 = sum(r["dF0_k0"] > 1 for r in rows if r["dF0_k0"] is not None)
        out["dF0_k1"][m] = {"rows": rows, "mean_k1": float(np.mean([r["dF0_k1"] for r in rows])), "mean_k0": float(np.mean([r["dF0_k0"] for r in rows])),
                            "n_above_1_k1": int(n1), "n_above_1_k0": int(n0), "n": len(rows),
                            "n_class_changes": int(sum((r["dF0_k1"] > 1) != (r["dF0_k0"] > 1) for r in rows))}
        print(f"(h) {m} e=4: F'(0) k0 mean {out['dF0_k1'][m]['mean_k0']:.2f} ({n0}/{len(rows)} above 1) -> k1 mean {out['dF0_k1'][m]['mean_k1']:.2f} ({n1}/{len(rows)}); class changes {out['dF0_k1'][m]['n_class_changes']}")
    # (i) covariates and per-formulation NLL / J at 4 peers
    cv = json.load(open(R / "covariates.json", encoding="utf-8"))
    out["covariates"] = {k: v for k, v in cv.items() if k != "manifest"}
    out["formulation_table"] = {}
    for m in ("qwen0.6b", "qwen1.7b"):
        rows = []
        for t in tables(cell(m, "list", 0, 4, "all")):
            J, h = fit_Jh(mean_gaps(t["entries"]))
            nll = [x["nll"] for x in t["entries"] if "nll" in x]
            rows.append({"wording": t["wording"], "format": t["format"], "J_gap": J, "h_gap": h, "nll_mean": float(np.mean(nll)) if nll else None})
        out["formulation_table"][m] = rows
    # appendix: the wordings and rendered examples (prompt builder only, no model)
    import yaml
    from src.llm.phase_a import build_prompt_a
    cfg = yaml.safe_load(open("configs/phase_a_formulations.yaml", encoding="utf-8"))
    fmt2 = yaml.safe_load(open("configs/phase_a_formulations_fmt2.yaml", encoding="utf-8"))
    states = (1, -1, 1, 1)
    lines = ["# Appendix. The five wordings and their rendering", "",
             "Generated by `scripts/review_r2_checks.py` from `configs/phase_a_formulations.yaml` and `src/llm/phase_a.py`; the text below is what the model reads, byte for byte, before the JSON prefix `{\"state\": \"`. Slots: {a} {b} {n_peers} {peers} {memory}.", ""]
    for i, w in enumerate(cfg["wordings"]):
        lines += [f"**Wording {i}.** `{w}`", ""]
    lines += ["**Formats of the peer list** (four peers with states A, B, A, A): " + "; ".join(f"{f}: `{build_prompt_a('{peers}', 'list', states, (), 'A', f)!r}`".replace("'", "") for f in list(cfg["formats"]) + list(fmt2.get("formats", []))), ""]
    lines += ["**Rendered examples, wording 0, four peers (A, B, A, A).**", ""]
    ex = [("list, comma, no memory", "list", (), "comma"), ("list, lines, no memory", "list", (), "lines"), ("count channel, no memory", "count", (), "comma"),
          ("list, comma, memory k = 1 (own last state A)", "list", (1,), "comma"), ("list, comma, memory k = 3 (own last states B, A, A)", "list", (-1, 1, 1), "comma")]
    for lab, ch, mem, f in ex:
        lines += [f"*{lab}*", "", "```", build_prompt_a(cfg["wordings"][0], ch, states, mem, "A", f), "```", ""]
    lines += ["Defects present in the tabulated data and not corrected: the count channel renders \"1 hold B\" and \"1 of them hold A\" (singular subject with plural verb); the memory sentence at k = 1 reads \"In the previous rounds Player 1 held, oldest first: A.\" for a single state; wording 3 opens its instruction with \"State Player 1 should now be in\"; the lines format starts with a line break inside the sentence and ends with the full stop after the last line.", ""]
    Path("paper/appendix_prompts.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(R / "review_r2.json", {"manifest": build_manifest({"script": "review_r2_checks"}, [41]), **out})
    print("written results/phase_a/review_r2.json and paper/appendix_prompts.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
