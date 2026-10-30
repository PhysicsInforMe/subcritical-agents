#!/usr/bin/env python
"""Investigation opened by the outcome of test 1 of the results notes (not included)
(2026-09-15): in all four (model, exposure) comparisons the linear summary
z_bar * J_gap at 32 orderings exceeds the 8-ordering value by 1-3 % with a
paired bootstrap interval (across formulations) that excludes zero.

Two hypotheses, declared before the data (the project log (not included), 2026-09-15):
  H_shared  the 8 orderings of a cell are drawn with one seed shared by
            every formulation (src/llm/phase_a.py, seed=0), so a sampling
            fluctuation common to all formulations appears as a
            "systematic" paired shift; the paired bootstrap across
            formulations cannot see it because the formulations are not
            independent draws of the orderings.
  H_bias    the 8-ordering tabulation underestimates z_bar * J_gap.

Test. From each 32-ordering table draw random subsets of 8 orderings per
k (where more than 8 exist), either shared across the formulations of the
cell, as in the tabulation, or independent per formulation. For each
subset compute z_bar * J_gap = e * slope / 2 of the linear fit of the mean
gap against k, then the paired mean difference (subset - 32) over the
formulations. Decision rule: H_shared if the observed 8 - 32 differences
fall inside the 2.5-97.5 % range of the shared-subset distribution and
that distribution is centred within 0.02 of zero; H_bias otherwise. The
independent-subset distribution is the control: its spread should shrink
by about sqrt(n_formulations).

Output: results/phase_a/orderings_subset.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402


def by_k(entries: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for x in entries:
        out.setdefault(x["k"], []).append(x)
    return out


def zj_r2(byk: dict[int, list[float]], e: int) -> tuple[float, float]:
    """z_bar * J_gap = e * slope / 2 of the linear fit of the mean gap against k
    (the two-unanimity slope, §21), and the R^2 of that fit."""
    ks = np.array(sorted(byk), float)
    g = np.array([np.mean(byk[int(k)]) for k in ks])
    coef = np.polyfit(ks, g, 1)
    pred = np.polyval(coef, ks)
    r2 = float(1 - ((g - pred) ** 2).sum() / ((g - g.mean()) ** 2).sum())
    return float(e * coef[0] / 2), r2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="qwen0.6b,qwen1.7b")
    ap.add_argument("--exposures", default="10,30")
    ap.add_argument("--n-subsets", type=int, default=200)
    ap.add_argument("--subset-size", type=int, default=8)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(15)
    check = json.load(open(Path(a.results) / "orderings_check.json", encoding="utf-8"))["models"]
    out = {"models": {}}
    for model in a.models.split(","):
        out["models"][model] = {}
        for e in (int(x) for x in a.exposures.split(",")):
            p32 = Path(a.results) / f"A_{model}_list_k0_e{e}_Tan_f0_allo32_N32" / "gaps.json"
            tables = json.load(open(p32, encoding="utf-8"))["tables"]
            forms = [by_k(t["entries"]) for t in tables]
            # the orderings at the same index are the same across formulations
            # (same seed in the tabulation); verified, not assumed
            ref = forms[0]
            for f in forms[1:]:
                for k in ref:
                    assert [x["states"] for x in f[k]] == [x["states"] for x in ref[k]], "orderings differ"
            full = np.array([zj_r2(({k: [x["gap"] for x in v] for k, v in f.items()}), e)[0] for f in forms])
            diffs = {"shared": [], "independent": []}
            for _ in range(a.n_subsets):
                idx_shared = {k: rng.choice(len(v), size=min(a.subset_size, len(v)), replace=False)
                              for k, v in ref.items()}
                for mode in diffs:
                    vals = []
                    for f in forms:
                        idx = idx_shared if mode == "shared" else {
                            k: rng.choice(len(v), size=min(a.subset_size, len(v)), replace=False)
                            for k, v in f.items()}
                        sub = {k: [f[k][i]["gap"] for i in idx[k]] for k in f}
                        vals.append(zj_r2(sub, e)[0])
                    diffs[mode].append(float(np.mean(np.array(vals) - full)))
            observed = -check[model][str(e)]["summary"]["zJ_gap"]["paired_diff_32_minus_8"]   # 8 - 32
            res = {"observed_8_minus_32": observed, "mean32": float(full.mean())}
            for mode, d in diffs.items():
                d = np.array(d)
                lo, hi = np.percentile(d, [2.5, 97.5])
                res[mode] = {"centre": float(d.mean()), "sd": float(d.std(ddof=1)),
                             "q025": float(lo), "q975": float(hi),
                             "observed_inside": bool(lo <= observed <= hi),
                             "frac_below_observed": float(np.mean(d <= observed))}
            res["sd_ratio_shared_over_independent"] = res["shared"]["sd"] / res["independent"]["sd"]
            out["models"][model][str(e)] = res
            print(f"== {model} e={e}: observed 8-32 = {observed:+.3f} (mean32 {full.mean():.2f})")
            for mode in ("shared", "independent"):
                r = res[mode]
                print(f"  {mode:12s}: centre {r['centre']:+.3f}, sd {r['sd']:.3f}, 95% [{r['q025']:+.3f}, {r['q975']:+.3f}], "
                      f"observed inside: {r['observed_inside']}, frac below observed {r['frac_below_observed']:.2f}")
            print(f"  sd ratio shared/independent: {res['sd_ratio_shared_over_independent']:.2f} (sqrt(n_forms) = {np.sqrt(len(forms)):.2f})")
    write_json(Path(a.results) / "orderings_subset.json", {"manifest": build_manifest(vars(a), [15]), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
