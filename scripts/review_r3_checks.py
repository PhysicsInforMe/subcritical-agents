#!/usr/bin/env python
"""Zero-cost re-analyses answering the schede of revision round, R3 (the revision round (not included)),
all from existing gap tables and result files. Output: results/phase_a/review_r3.json.

(1) R3-01, R3-07  Every paired difference of the paper recomputed as a t interval
                  with 4 degrees of freedom on the five per-wording means
                  (wording = independent unit), next to the cluster percentile
                  bootstrap in use: exposure 4-2 and 7-2 (10 and 20 formulations),
                  memory k3-k0 and the steps, J_mem per state steps, the count
                  channel, z_bar J 7->15 and 7->30, count minus list, order B/A,
                  separator contrasts against comma.
(2) R3-01         Coverage of the cluster percentile bootstrap with 5 clusters of 2
                  (and of 4) members, normal data with the between- and
                  within-wording variances of the 0.6B exposure difference 4-2;
                  2000 replicates x 2000 resamples; t interval for comparison.
(3) R3-02         Additive null for the exponent of the unanimity jump: absolute
                  increment jump(30) - jump(7) against jump(7) (Spearman), and the
                  exponent, Spearman and median split that a common additive
                  increment reproduces, per model.
(4) R3-03         bfloat16 noise (sd from the float32 check) propagated to F'(0) of
                  the 1.7B at 4 peers, 400 draws per formulation: fraction above 1
                  and interval classes; first-peer-copy check on every table.
(5) R3-04         Spearman at wording level (n = 5, exact permutation p) for
                  exponent against jump at 7, per model; both median-split rules
                  from h10_test.json with their outcomes.
(6) R3-06         Mixed model of the separator with a wording x format variance
                  component (statsmodels, ML), LRT for format and interaction,
                  three models; paired contrasts against comma with t intervals
                  over wordings (minimum detectable effect = half-width).
(7) R3-09         Are the 8 sampled orderings identical between the two models at
                  10 and 30 peers? Tail probabilities of the subset test.
(8) R3-12         Sector-A classes on the 32-ordering tables at 7, 10 and 30 peers
                  with an ordering bootstrap (200 draws): class of the full table
                  and fraction of draws agreeing.
"""
from __future__ import annotations

import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_format_check import boot_mean, j_gap, load_mean_gaps  # noqa: E402
from scripts.phase_a_meanfield_stability import fixed_points, mf_map, policy_from_entries  # noqa: E402
from scripts.phase_a_memory_curve import gap_fit  # noqa: E402
from src.common import build_manifest, write_json  # noqa: E402

R = Path("results/phase_a")
T4 = float(stats.t.ppf(0.975, 4))


def t_int(vals: dict[int, float]) -> dict:
    """t interval with 4 df on the five per-wording means (values keyed by wording)."""
    m = np.array([np.mean(v) for v in vals.values()])
    return {"mean": float(m.mean()), "ci95": [float(m.mean() - T4 * m.std(ddof=1) / np.sqrt(m.size)), float(m.mean() + T4 * m.std(ddof=1) / np.sqrt(m.size))],
            "n_wordings": int(m.size), "n_negative": int((m < 0).sum()), "per_wording": [float(x) for x in m]}


def both(d: np.ndarray, words: np.ndarray, rng) -> dict:
    per = {int(w): [float(x) for x in d[words == w]] for w in np.unique(words)}
    t = t_int({w: v for w, v in per.items()})
    b = boot_mean(d, words, rng)
    t["excludes_zero"] = bool(t["ci95"][0] > 0 or t["ci95"][1] < 0)
    return {"t4": t, "cluster_boot": {"mean": b["mean"], "ci95": b["ci95"], "excludes_zero": bool(b["ci95"][0] > 0 or b["ci95"][1] < 0)}}


def fmt(r: dict) -> str:
    t, b = r["t4"], r["cluster_boot"]
    return f"t4 {t['mean']:+.2f} [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}] ({'excl' if t['excludes_zero'] else 'incl'} 0; neg {t['n_negative']}/{t['n_wordings']}) | boot [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}] ({'excl' if b['excludes_zero'] else 'incl'})"


def cell_tables(model, channel, k, e, suffix="all"):
    hits = glob.glob(str(R / f"A_{model}_{channel}_k{k}_e{e}_Tan_f0_{suffix}_N*/gaps.json"))
    return json.load(open(hits[0], encoding="utf-8"))["tables"] if hits else None


def main() -> int:
    rng = np.random.default_rng(61)
    out: dict = {"paired": {}}
    # ---------- (1) paired differences, t4 vs cluster bootstrap ----------
    for m in ("qwen0.6b", "qwen1.7b", "smollm1.7b"):
        G = load_mean_gaps(R, m)
        P = out["paired"].setdefault(m, {})
        forms10 = sorted({(w, f) for (w, f, e) in G if f in ("comma", "lines")})
        forms20 = sorted({(w, f) for (w, f, e) in G})
        for lab, forms in (("10", forms10), ("20", forms20)):
            words = np.array([w for w, _ in forms])
            for e1, e0 in ((4, 2), (7, 2)):
                if all((w, f, e1) in G and (w, f, e0) in G for w, f in forms):
                    d = np.array([j_gap(G[(w, f, e1)]) - j_gap(G[(w, f, e0)]) for w, f in forms])
                    P[f"J.e{e1}-e{e0}.n{lab}"] = both(d, words, rng); print(f"(1) {m} J e{e1}-e{e0} n{lab}: {fmt(P[f'J.e{e1}-e{e0}.n{lab}'])}")
            for e1 in (15, 30):
                if lab == "20" and all((w, f, e1) in G and (w, f, 7) in G for w, f in forms):
                    d = np.array([e1 * j_gap(G[(w, f, e1)]) - 7 * j_gap(G[(w, f, 7)]) for w, f in forms])
                    P[f"zJ.e{e1}-e7.n20"] = both(d, words, rng); print(f"(1) {m} zJ e{e1}-e7: {fmt(P[f'zJ.e{e1}-e7.n20'])}")
        if m == "smollm1.7b":
            continue
        # memory curve at 4 peers, list, 10 formulations
        Jk, Jm = {}, {}
        for k in (0, 1, 2, 3):
            tabs = cell_tables(m, "list", k, 4)
            for tb in tabs:
                J, jm = gap_fit(tb["entries"], k > 0)
                Jk[(tb["wording"], tb["format"], k)] = J; Jm[(tb["wording"], tb["format"], k)] = jm
        forms = sorted({(w, f) for (w, f, k) in Jk}); words = np.array([w for w, _ in forms])
        for k1, k0 in ((3, 0), (1, 0), (2, 1), (3, 2)):
            d = np.array([Jk[(w, f, k1)] - Jk[(w, f, k0)] for w, f in forms])
            P[f"J.k{k1}-k{k0}"] = both(d, words, rng); print(f"(1) {m} memory J k{k1}-k{k0}: {fmt(P[f'J.k{k1}-k{k0}'])}")
        for k1, k0 in ((2, 1), (3, 2), (3, 1)):
            d = np.array([Jm[(w, f, k1)] - Jm[(w, f, k0)] for w, f in forms])   # J_mem per state: gap_fit returns the coefficient of the sum of the memory states
            P[f"Jmem_per_state.k{k1}-k{k0}"] = both(d, words, rng); print(f"(1) {m} J_mem/state k{k1}-k{k0}: {fmt(P[f'Jmem_per_state.k{k1}-k{k0}'])}")
        # count channel k3 - k0 at 4 peers (five wordings)
        c0 = {tb["wording"]: gap_fit(tb["entries"], False)[0] for tb in cell_tables(m, "count", 0, 4)}
        c3 = {tb["wording"]: gap_fit(tb["entries"], True)[0] for tb in cell_tables(m, "count", 3, 4)}
        ws = sorted(set(c0) & set(c3)); d = np.array([c3[w] - c0[w] for w in ws])
        P["J.count.k3-k0"] = both(d, np.array(ws), rng); print(f"(1) {m} count k3-k0: {fmt(P['J.count.k3-k0'])}")
        # count minus list (R2-03) at 2, 4, 7
        for e in (2, 4, 7):
            jl = {tb["wording"]: gap_fit(tb["entries"], False)[0] for tb in cell_tables(m, "list", 0, e) if tb["format"] == "comma"}
            jc = {tb["wording"]: gap_fit(tb["entries"], False)[0] for tb in cell_tables(m, "count", 0, e)}
            ws = sorted(set(jl) & set(jc)); d = np.array([jc[w] - jl[w] for w in ws])
            P[f"J.count-list.e{e}"] = both(d, np.array(ws), rng); print(f"(1) {m} count-list e{e}: {fmt(P[f'J.count-list.e{e}'])}")
        # order B/A (R2-06)
        ba = cell_tables(m, "list", 0, 4, "allba")
        if ba:
            ab = {(tb["wording"], tb["format"]): tb for tb in cell_tables(m, "list", 0, 4)}
            keys = sorted((tb["wording"], tb["format"]) for tb in ba); bat = {(tb["wording"], tb["format"]): tb for tb in ba}

            def fit_h(entries):
                acc = {}
                for x in entries:
                    acc.setdefault(x["k"], []).append(x["gap"])
                ks = np.array(sorted(acc), float); g = np.array([np.mean(acc[int(k)]) for k in ks])
                return np.polyfit(ks, g, 1)[1] / 2
            d = np.array([fit_h(bat[key]["entries"]) - fit_h(ab[key]["entries"]) for key in keys]); words = np.array([k[0] for k in keys])
            P["h.ba-ab.e4"] = both(d, words, rng); print(f"(1) {m} order dh: {fmt(P['h.ba-ab.e4'])}")
        # separator contrasts against comma, mean over exposures per (wording, format)
        for f in ("lines", "semicolon", "brackets"):
            ws = sorted({w for (w, ff, e) in G if ff == f})
            d = np.array([np.mean([j_gap(G[(w, f, e)]) - j_gap(G[(w, "comma", e)]) for e in (2, 4, 7, 10, 15, 20, 30) if (w, f, e) in G and (w, "comma", e) in G]) for w in ws])
            P[f"J.{f}-comma"] = both(d, np.array(ws), rng); print(f"(1) {m} {f}-comma: {fmt(P[f'J.{f}-comma'])}")
    # ---------- (2) coverage simulation ----------
    G = load_mean_gaps(R, "qwen0.6b")
    forms = sorted({(w, f) for (w, f, e) in G if f in ("comma", "lines")})
    d = np.array([j_gap(G[(w, f, 4)]) - j_gap(G[(w, f, 2)]) for w, f in forms]); words = np.array([w for w, _ in forms])
    cm = np.array([d[words == w].mean() for w in np.unique(words)])
    s_b = float(cm.std(ddof=1)); s_w = float(np.sqrt(np.mean([d[words == w].var(ddof=1) for w in np.unique(words)])))
    out["coverage"] = {}
    for n_per in (2, 4):
        n_rep, n_boot, n_cl = 2000, 2000, 5
        hits_b = hits_t = 0
        for _ in range(n_rep):
            x = rng.normal(0, s_b, n_cl)[:, None] + rng.normal(0, s_w, (n_cl, n_per))
            means = x.mean(axis=1)
            idx = rng.integers(0, n_cl, (n_boot, n_cl)); bm = means[idx].mean(axis=1)
            lo, hi = np.percentile(bm, [2.5, 97.5]); hits_b += (lo <= 0 <= hi)
            se = means.std(ddof=1) / np.sqrt(n_cl); hits_t += (abs(means.mean()) <= T4 * se)
        out["coverage"][f"members{n_per}"] = {"cluster_percentile_boot": hits_b / n_rep, "t4": hits_t / n_rep, "sd_between": s_b, "sd_within": s_w}
        print(f"(2) coverage, 5 clusters x {n_per}: percentile cluster bootstrap {hits_b / n_rep:.3f}, t4 {hits_t / n_rep:.3f}")
    # ---------- (3) additive null for the exponent ----------
    out["additive_null"] = {}
    for m in ("qwen0.6b", "qwen1.7b", "smollm1.7b"):
        G = load_mean_gaps(R, m)
        es = [7, 10, 15, 20, 30]
        forms = sorted({(w, f) for (w, f, e) in G if all((w, f, e) in G for e in es + [2])})
        jump = {(w, f, e): G[(w, f, e)][e] - G[(w, f, e)][-e] for w, f in forms for e in es + [2]}
        j7 = np.array([jump[(w, f, 7)] for w, f in forms]); inc = np.array([jump[(w, f, 30)] - jump[(w, f, 7)] for w, f in forms]); j2 = np.array([jump[(w, f, 2)] for w, f in forms])
        rho_inc, p_inc = stats.spearmanr(inc, j7)
        # observed exponent (formulations with all jumps > 0.5 logit, as in the declared test)
        def expo(vals):
            return float(np.polyfit(np.log(es), np.log(vals), 1)[0]) if min(vals) > 0.5 else None
        obs = [expo([jump[(w, f, e)] for e in es]) for w, f in forms]
        b = float(np.mean([(jump[(w, f, 30)] - jump[(w, f, 7)]) / 23 for w, f in forms]))
        null = [expo([jump[(w, f, 7)] + b * (e - 7) for e in es]) for w, f in forms]
        ok = [i for i in range(len(forms)) if obs[i] is not None and null[i] is not None]
        rho_obs = stats.spearmanr([obs[i] for i in ok], j7[ok])[0]; rho_null = stats.spearmanr([null[i] for i in ok], j7[ok])[0]
        med = np.median(j2); weak = [i for i in ok if j2[i] <= med]; strong = [i for i in ok if j2[i] > med]
        split_obs = float(np.mean([obs[i] for i in weak]) - np.mean([obs[i] for i in strong])); split_null = float(np.mean([null[i] for i in weak]) - np.mean([null[i] for i in strong]))
        inc_weak = float(np.mean(inc[[i for i in range(len(forms)) if j2[i] <= med]])); inc_strong = float(np.mean(inc[[i for i in range(len(forms)) if j2[i] > med]]))
        out["additive_null"][m] = {"n": len(forms), "spearman_increment_vs_jump7": float(rho_inc), "p": float(p_inc), "b_common": b,
                                   "spearman_exponent_obs": float(rho_obs), "spearman_exponent_null": float(rho_null),
                                   "median_split_obs": split_obs, "median_split_null": split_null, "increment_weak_half": inc_weak, "increment_strong_half": inc_strong,
                                   "sd_jump7": float(j7.std(ddof=1)), "sd_jump30": float(np.array([jump[(w, f, 30)] for w, f in forms]).std(ddof=1))}
        print(f"(3) {m} n={len(forms)}: Spearman(increment, jump7) {rho_inc:+.2f} (p {p_inc:.2f}); exponent-jump7 obs {rho_obs:+.2f} null {rho_null:+.2f}; median split obs {split_obs:+.2f} null {split_null:+.2f}; increments weak {inc_weak:+.2f} strong {inc_strong:+.2f}")
    # ---------- (4) bfloat16 noise on F'(0), 1.7B at 4 peers; first-peer copy ----------
    fp = json.load(open(R / "fp32check.json", encoding="utf-8")) if (R / "fp32check.json").exists() else None
    sd = 0.37
    tabs = cell_tables("qwen1.7b", "list", 0, 4)
    rows = []
    for tb in tabs:
        ents = tb["entries"]
        d0 = []
        for _ in range(400):
            noisy = [{**x, "gap": x["gap"] + rng.normal(0, sd)} for x in ents]
            _, ps = policy_from_entries(noisy, 1.0); d0.append(mf_map(ps, 4)[1](0.0))
        d0 = np.array(d0); base = float(mf_map(policy_from_entries(ents, 1.0)[1], 4)[1](0.0))
        frac = float((d0 > 1).mean()); cls = "ordered" if frac >= 0.975 else ("disordered" if frac <= 0.025 else "undetermined")
        copy = float(np.mean([np.sign(x["gap"]) == np.sign(x["states"][0]) for x in ents]))
        rows.append({"wording": tb["wording"], "format": tb["format"], "dF0": base, "frac_above_1": frac, "ci": [float(np.percentile(d0, 2.5)), float(np.percentile(d0, 97.5))], "class_bf16": cls, "first_peer_copy_frac": copy})
    counts = {c: sum(r["class_bf16"] == c for r in rows) for c in ("ordered", "disordered", "undetermined")}
    out["bf16_dF0_e4"] = {"sd_used": sd, "rows": rows, "counts": counts}
    print(f"(4) 1.7B e4 with bf16 noise sd {sd}: classes {counts}; " + "; ".join(f"w{r['wording']}{r['format'][0]} {r['dF0']:.3f} p>1 {r['frac_above_1']:.2f} copy {r['first_peer_copy_frac']:.2f}" for r in rows))
    # ---------- (5) wording-level Spearman, both median splits ----------
    h = json.load(open(R / "h10_test.json", encoding="utf-8"))["models"]
    out["wording_level"] = {}
    perms = list(itertools.permutations(range(5)))
    for m, blk in h.items():
        rows = [r for r in blk["rows"] if r["exp7"] is not None]
        ws = sorted({r["wording"] for r in rows})
        ex = np.array([np.mean([r["exp7"] for r in rows if r["wording"] == w]) for w in ws]); jp = np.array([np.mean([r["jump7"] for r in rows if r["wording"] == w]) for w in ws])
        rho = stats.spearmanr(ex, jp)[0]
        rk = stats.rankdata(jp); null = [stats.spearmanr(ex, rk[list(p)])[0] for p in perms]
        p_exact = float(np.mean([abs(x) >= abs(rho) - 1e-12 for x in null]))
        out["wording_level"][m] = {"rho": float(rho), "p_exact": p_exact, "n": len(ws), "median_split": {k: {"diff": v["diff_weak_minus_strong"], "ci": v["ci95_cluster"][:2], "passes": v["passes"]} for k, v in blk["b"].items()}}
        print(f"(5) {m}: wording-level rho {rho:+.2f} (exact p {p_exact:.3f}); median split exp7 {blk['b']['exp7']['diff_weak_minus_strong']:+.2f} passes {blk['b']['exp7']['passes']}, exp10 {blk['b']['exp10']['diff_weak_minus_strong']:+.2f} passes {blk['b']['exp10']['passes']}")
    # ---------- (6) mixed model with wording x format component ----------
    import pandas as pd
    import statsmodels.formula.api as smf
    out["mixed_wf"] = {}
    for m in ("qwen0.6b", "qwen1.7b", "smollm1.7b"):
        G = load_mean_gaps(R, m)
        df = pd.DataFrame([{"wording": w, "format": f, "exposure": e, "J_gap": j_gap(g), "wf": f"{w}_{f}"} for (w, f, e), g in G.items()])

        def fit(formula):
            return smf.mixedlm(formula, df, groups=df["wording"], vc_formula={"wf": "0 + C(wf)"}).fit(reml=False)
        m0 = fit("J_gap ~ C(exposure)"); m1 = fit("J_gap ~ C(exposure) + C(format)"); m2 = fit("J_gap ~ C(exposure) + C(format) + C(format):C(exposure)")
        nf, ne = df["format"].nunique(), df["exposure"].nunique()
        lr_f, lr_i = 2 * (m1.llf - m0.llf), 2 * (m2.llf - m1.llf)
        res = {"format": {"LR": float(lr_f), "df": nf - 1, "p": float(stats.chi2.sf(lr_f, nf - 1))}, "interaction": {"LR": float(lr_i), "df": (nf - 1) * (ne - 1), "p": float(stats.chi2.sf(lr_i, (nf - 1) * (ne - 1)))},
               "vc_wording_format": float(m1.vcomp[0]) if len(m1.vcomp) else None, "residual": float(m1.scale)}
        out["mixed_wf"][m] = res
        print(f"(6) {m}: format LR {lr_f:.2f} df {nf - 1} p {res['format']['p']:.3f}; interaction LR {lr_i:.2f} df {(nf - 1) * (ne - 1)} p {res['interaction']['p']:.3f}; vc wf {res['vc_wording_format']}")
    # ---------- (7) shared orderings, subset-test tails ----------
    out["shared_orderings"] = {}
    for e in (10, 30):
        a = cell_tables("qwen0.6b", "list", 0, e); b = cell_tables("qwen1.7b", "list", 0, e)
        sa = [tuple(x["states"]) for x in a[0]["entries"]]; sb = [tuple(x["states"]) for x in b[0]["entries"]]
        out["shared_orderings"][str(e)] = {"identical": sa == sb, "n_entries": len(sa)}
    sub = json.load(open(R / "orderings_subset.json", encoding="utf-8"))["models"]
    out["subset_tails"] = {m: {e: {"frac_below_observed": v["shared"]["frac_below_observed"], "inside": v["shared"]["observed_inside"], "sd_shared": v["shared"]["sd"]} for e, v in blk.items()} for m, blk in sub.items()}
    print("(7) shared orderings identical:", {e: v["identical"] for e, v in out["shared_orderings"].items()}, "; tails:", {m: {e: v["frac_below_observed"] for e, v in b.items()} for m, b in out["subset_tails"].items()})
    # ---------- (8) sector-A classes on 32-ordering tables with ordering bootstrap ----------
    out["sectorA_32"] = {}

    def classify(pA):
        F, dF = mf_map(pA, e_)
        st = [f for f in fixed_points(F, dF) if f["stable"]]
        if len(st) >= 2:
            return "bistable"
        return "field_dominated" if st and abs(st[0]["m"]) > 0.9 else "disordered"
    for m in ("qwen0.6b", "qwen1.7b"):
        out["sectorA_32"][m] = {}
        for e_ in (7, 10, 30):
            tabs = cell_tables(m, "list", 0, e_, "allo32")
            if not tabs:
                continue
            rows = []
            for tb in tabs:
                byk = {}
                for x in tb["entries"]:
                    byk.setdefault(x["k"], []).append(x["gap"])
                pA = {k: float(np.mean(1 / (1 + np.exp(-np.array(v))))) for k, v in byk.items()}
                full = classify(pA)
                agree = 0
                for _ in range(200):
                    pb = {k: float(np.mean(1 / (1 + np.exp(-rng.choice(v, size=len(v), replace=True))))) for k, v in byk.items()}
                    agree += classify(pb) == full
                rows.append({"wording": tb["wording"], "format": tb["format"], "class": full, "frac_agree": agree / 200})
            counts = {c: sum(r["class"] == c for r in rows) for c in ("bistable", "field_dominated", "disordered")}
            firm = sum(r["frac_agree"] >= 0.95 for r in rows)
            out["sectorA_32"][m][str(e_)] = {"counts": counts, "n_firm_95": firm, "rows": rows}
            print(f"(8) {m} e{e_} sector-A classes on 32 orderings: {counts}; firm (>=95% of draws) {firm}/{len(rows)}")
    write_json(R / "review_r3.json", {"manifest": build_manifest({"script": "review_r3_checks"}, [61]), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
