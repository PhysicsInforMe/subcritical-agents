#!/usr/bin/env python
"""Test 2 of the results notes (not included) (predictions P2a-c, written before the
data, 2026-09-14): do the exposure findings survive twenty formulations,
i.e. the five wordings in four list formats (comma, lines from
configs/phase_a_formulations.yaml; semicolon, brackets from
configs/phase_a_formulations_fmt2.yaml)?

For the list channel with no memory, exposures 2-30, the cells with 8
orderings per configuration (A_<m>_list_k0_e<e>_Tan_f0_all_N* and
..._allf2_N*; the 32-ordering cells are not used so that all twenty
formulations share the same sampling), per (wording, format, e):
    * J_gap = slope / 2 of the mean gap against k (§21 estimator);
    * unanimity jump gap(+e) - gap(-e) and its exponent over e >= 7
      (log jump against log e, as in exposure_shape; formulations with a
      jump below 0.5 logit at some e are excluded from the exponent).
Then:
    P2a  paired differences J_gap(e4) - J_gap(e2) and J_gap(e7) - J_gap(e2)
         over the twenty formulations, cluster bootstrap over wordings; also
         over the ten new formulations alone (independent replication of
         B1) and the ten original ones (must reproduce claims.md B1);
    P2b  Spearman correlation between the exponent and the strength at 7
         peers (the unanimity jump at e = 7, §23) over all formulations
         with an exponent, and over the ten new ones alone; the ten
         original ones must reproduce §23 (-0.69 on 8, -0.90 on 10);
    P2c  mixed model J_gap ~ C(exposure) + C(format) + C(format):C(exposure)
         with a random intercept per wording (statsmodels MixedLM, ML),
         likelihood-ratio tests of the format main effect and of the
         format x exposure interaction against the nested models.
Decision rule (§24): P2a holds if both paired differences are negative
with intervals excluding zero on twenty formulations; P2b holds if
Spearman is below -0.5 in both models, otherwise the anticorrelation is
removed from H10; P2c holds if the format LRT and the interaction LRT
are not significant at 0.05. Output: results/phase_a/format_check.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402

ORIGINAL = ("comma", "lines")
NEW = ("semicolon", "brackets")


def load_mean_gaps(results: Path, model: str) -> dict[tuple, dict[int, float]]:
    """{(wording, format, e): {k: mean gap}} over the 8-ordering cells of both format files."""
    out: dict[tuple, dict[int, float]] = {}
    for suffix in ("all", "allf2"):
        for path in glob.glob(str(results / f"A_{model}_list_k0_e*_Tan_f0_{suffix}_N*/gaps.json")):
            e = int(re.search(r"_e(\d+)_", Path(path).parent.name).group(1))
            for tab in json.load(open(path, encoding="utf-8"))["tables"]:
                byk: dict[int, list[float]] = {}
                for x in tab["entries"]:
                    byk.setdefault(x["k"], []).append(x["gap"])
                out[(tab["wording"], tab["format"], e)] = {k: float(np.mean(v)) for k, v in byk.items()}
    return out


def j_gap(g: dict[int, float]) -> float:
    ks = np.array(sorted(g), float)
    return float(np.polyfit(ks, np.array([g[int(k)] for k in ks]), 1)[0] / 2)


def boot_mean(d: np.ndarray, clusters: np.ndarray, rng: np.random.Generator, n: int = 4000) -> dict:
    """Mean with a cluster bootstrap over wordings, as in phase_a_aggregate:
    the formats of one wording share its lexicon, so the wording is the
    independent unit, not the (wording, format) pair."""
    uniq = np.unique(clusters)
    b = [np.mean(np.concatenate([d[clusters == c] for c in rng.choice(uniq, size=uniq.size, replace=True)]))
         for _ in range(n)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)], "n": int(d.size),
            "n_clusters": int(uniq.size), "excludes_zero": bool(hi < 0 or lo > 0)}


def paired(J: dict, forms: list[tuple], e1: int, e0: int, rng: np.random.Generator) -> dict:
    keys = [(w, f) for (w, f) in forms if (w, f, e1) in J and (w, f, e0) in J]
    d = np.array([J[(w, f, e1)] - J[(w, f, e0)] for (w, f) in keys])
    return boot_mean(d, np.array([w for (w, _) in keys]), rng)


def mixed_lrt(rows: list[dict]) -> dict:
    """LRT of C(format) and C(format):C(exposure) in J_gap ~ C(exposure) + ...,
    random intercept per wording, ML fits so that likelihoods are comparable."""
    import pandas as pd
    import statsmodels.formula.api as smf
    df = pd.DataFrame(rows)

    def fit(formula: str):
        return smf.mixedlm(formula, df, groups=df["wording"]).fit(reml=False)

    m0 = fit("J_gap ~ C(exposure)")
    m1 = fit("J_gap ~ C(exposure) + C(format)")
    m2 = fit("J_gap ~ C(exposure) + C(format) + C(format):C(exposure)")
    n_fmt, n_exp = df["format"].nunique(), df["exposure"].nunique()
    lr_fmt, df_fmt = 2 * (m1.llf - m0.llf), n_fmt - 1
    lr_int, df_int = 2 * (m2.llf - m1.llf), (n_fmt - 1) * (n_exp - 1)
    return {"llf": {"exposure": float(m0.llf), "+format": float(m1.llf), "+interaction": float(m2.llf)},
            "format": {"LR": float(lr_fmt), "df": int(df_fmt), "p": float(chi2.sf(lr_fmt, df_fmt))},
            "interaction": {"LR": float(lr_int), "df": int(df_int), "p": float(chi2.sf(lr_int, df_int))},
            "format_params_in_m1": {k: float(v) for k, v in m1.params.items() if "format" in k},
            "wording_variance": float(m1.cov_re.iloc[0, 0]), "residual_variance": float(m1.scale),
            "n_rows": int(len(df))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="qwen0.6b,qwen1.7b")
    ap.add_argument("--e-ref", type=int, default=7)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(24)
    out = {"models": {}}
    for model in a.models.split(","):
        G = load_mean_gaps(Path(a.results), model)
        exposures = sorted({e for (_, _, e) in G})
        forms = sorted({(w, f) for (w, f, e) in G})
        complete = [(w, f) for (w, f) in forms if all((w, f, e) in G for e in exposures)]
        J = {key: j_gap(g) for key, g in G.items()}
        groups = {"all": complete, "original": [x for x in complete if x[1] in ORIGINAL],
                  "new": [x for x in complete if x[1] in NEW]}
        res: dict = {"exposures": exposures, "n_formulations": {k: len(v) for k, v in groups.items()},
                     "J_gap_mean_by_exposure_format": {}, "P2a": {}, "P2b": {}, "P2c": None}
        for f in sorted({f for (_, f) in complete}):
            res["J_gap_mean_by_exposure_format"][f] = {str(e): float(np.mean([J[(w, f, e)] for (w, ff) in complete if ff == f]))
                                                       for e in exposures}
        # P2a
        for gname, fl in groups.items():
            res["P2a"][gname] = {"e4-e2": paired(J, fl, 4, 2, rng), f"e{a.e_ref}-e2": paired(J, fl, a.e_ref, 2, rng)}
        # P2b, plus the stress test with the exponent fitted on e > e_ref only:
        # the jump at e_ref enters both the exponent and the strength, so a
        # negative correlation could be induced by that shared point alone
        strength = {(w, f): G[(w, f, a.e_ref)][a.e_ref] - G[(w, f, a.e_ref)][-a.e_ref] for (w, f) in complete}
        for tag, big in (("", [e for e in exposures if e >= a.e_ref]),
                         ("_excl_e_ref", [e for e in exposures if e > a.e_ref])):
            expo = {}
            for (w, f) in complete:
                jumps = [G[(w, f, e)][e] - G[(w, f, e)][-e] for e in big]
                if len(big) >= 3 and min(jumps) > 0.5:
                    expo[(w, f)] = float(np.polyfit(np.log(big), np.log(jumps), 1)[0])
            for gname, fl in groups.items():
                keys = [x for x in fl if x in expo]
                if len(keys) >= 4:
                    rho, p = spearmanr([expo[x] for x in keys], [strength[x] for x in keys])
                    res["P2b"][gname + tag] = {"rho": float(rho), "p": float(p), "n": len(keys), "exposures": big,
                                               "exponents": {f"w{w}_{f}": expo[(w, f)] for (w, f) in keys},
                                               "strength_e_ref": {f"w{w}_{f}": strength[(w, f)] for (w, f) in keys}}
                else:
                    res["P2b"][gname + tag] = {"rho": None, "n": len(keys), "exposures": big}
        # P2c
        rows = [{"wording": w, "format": f, "exposure": e, "J_gap": J[(w, f, e)]} for (w, f) in complete for e in exposures]
        try:
            res["P2c"] = mixed_lrt(rows)
        except Exception as exc:
            res["P2c"] = {"error": str(exc)}
        out["models"][model] = res
        print(f"== {model}: exposures {exposures}; formulations {res['n_formulations']}")
        for f, d in res["J_gap_mean_by_exposure_format"].items():
            print(f"  J_gap mean {f:9s}: " + " ".join(f"e{e}={d[str(e)]:.2f}" for e in exposures))
        for gname in ("all", "original", "new"):
            p2 = res["P2a"][gname]
            print(f"  P2a {gname:8s}: " + "; ".join(f"{k} = {v['mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] n={v['n']}"
                                                    for k, v in p2.items()))
        for gname in ("all", "original", "new", "all_excl_e_ref", "new_excl_e_ref"):
            b = res["P2b"][gname]
            print(f"  P2b {gname:15s}: Spearman(exponent on e in {b['exposures']}, jump@{a.e_ref}) = " +
                  (f"{b['rho']:+.2f} (p = {b['p']:.3f}, n = {b['n']})" if b.get("rho") is not None else f"n/a (n = {b['n']})"))
        c = res["P2c"]
        if c and "format" in c:
            print(f"  P2c mixed model ({c['n_rows']} rows): format LRT p = {c['format']['p']:.3f} (df {c['format']['df']}); "
                  f"format x exposure LRT p = {c['interaction']['p']:.3f} (df {c['interaction']['df']}); "
                  f"wording var {c['wording_variance']:.3f}, residual var {c['residual_variance']:.3f}")
        else:
            print(f"  P2c: {c}")
    write_json(Path(a.results) / "format_check.json", {"manifest": build_manifest(vars(a), [24]), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
