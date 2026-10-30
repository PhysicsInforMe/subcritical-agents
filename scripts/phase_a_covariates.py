#!/usr/bin/env python
"""How much of the formulation effect on (J, h) do the B15 covariates explain?

For every k_mem = 0 cell (where the prompt NLL was computed) and every
formulation: prompt NLL (Gonen), H_0 = entropy of P(+|k=0), POSIX index of
the formulation against the others in the same cell on the two label
log-probabilities at k = 0 (Chatterjee). Then, per model:
    * Spearman correlations within cells, averaged;
    * mixed models J ~ knobs (+ covariates) with a random intercept per
      wording; the drop in the wording variance is the share of the
      formulation effect that the covariates explain (design notes §14.1 B15).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rows_for(results: Path) -> pd.DataFrame:
    rows = []
    for mpath in sorted(results.glob("*k0*/measures.json")):
        cell = mpath.parent
        m = json.load(open(mpath, encoding="utf-8"))
        g = json.load(open(cell / "gaps.json", encoding="utf-8"))
        cfg = m["manifest"]["config"]
        if cfg["topology"] != "all":
            continue
        # label log-probs at k = 0 per formulation (sector A, ordering-averaged)
        lp = {}
        for tab in g["tables"]:
            e0 = [e for e in tab["entries"] if e["k"] == 0]
            if not e0:                       # odd exposure: use the two central bins
                kmin = min(abs(e["k"]) for e in tab["entries"])
                e0 = [e for e in tab["entries"] if abs(e["k"]) == kmin]
            lp[(tab["wording"], tab["format"])] = (np.mean([e["logp_plus"] for e in e0]),
                                                  np.mean([e["logp_minus"] for e in e0]))
        keys = list(lp)
        for f in m["formulations"]:
            key = (f["wording"], f["format"])
            fit = f["by_T"]["1.0"]["fit"]
            p0 = 1 / (1 + math.exp(-(lp[key][0] - lp[key][1])))
            pref = 0 if p0 >= 0.5 else 1
            posix = float(np.mean([abs(lp[o][pref] - lp[key][pref]) for o in keys if o != key]))
            rows.append({"model": cfg["model_key"], "channel": cfg["channel"], "exposure": cfg["exposure"],
                         "wording": f["wording"], "format": f["format"], "cell": cfg["cell_id"],
                         "J": fit["J"]["estimate"], "h": fit["h"]["estimate"],
                         "nll": f["prompt_nll"], "posix": posix,
                         "H0": -(p0 * math.log2(p0) + (1 - p0) * math.log2(1 - p0))})
    return pd.DataFrame(rows)


def main() -> int:
    df = rows_for(Path("results/phase_a"))
    out = {}
    for model, d in df.groupby("model"):
        print(f"\n== {model}: {len(d)} rows, {d['cell'].nunique()} cells (k_mem = 0) ==")
        for target in ("J", "h"):
            cors = {c: np.nanmean([spearmanr(x[target], x[c]).correlation for _, x in d.groupby("cell") if len(x) > 3])
                    for c in ("nll", "posix", "H0")}
            print(f"  within-cell Spearman {target} vs", {k: round(v, 2) for k, v in cors.items()})
        for target in ("J", "h"):
            base = smf.mixedlm(f"{target} ~ C(channel) + C(exposure)", d, groups=d["wording"]).fit(reml=False)
            cov = smf.mixedlm(f"{target} ~ C(channel) + C(exposure) + nll + posix", d, groups=d["wording"]).fit(reml=False)
            vb, vc = float(base.cov_re.iloc[0, 0]), float(cov.cov_re.iloc[0, 0])
            print(f"  {target}: wording variance {vb:.4f} -> {vc:.4f} with nll+posix "
                  f"({100 * (1 - vc / max(vb, 1e-12)):.0f}% explained); residual {base.scale:.4f} -> {cov.scale:.4f}; "
                  f"p(nll)={cov.pvalues['nll']:.3f} p(posix)={cov.pvalues['posix']:.3f}")
            out[f"{model}|{target}"] = {"wording_var_base": vb, "wording_var_cov": vc,
                                        "p_nll": float(cov.pvalues["nll"]), "p_posix": float(cov.pvalues["posix"])}
    from src.common import build_manifest, write_json   # manifest added for revision R4-03
    out["manifest"] = build_manifest({"script": "phase_a_covariates", "results": "results/phase_a"}, [])
    write_json("results/phase_a/covariates.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
