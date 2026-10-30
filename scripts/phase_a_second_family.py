#!/usr/bin/env python
"""Test 3 of the results notes (not included) (prediction P3, written before the data,
2026-09-14): does the coupling per peer decrease with the number of peers
shown outside the Qwen family? Second family: SmolLM2-1.7B-Instruct
(HuggingFaceTB), list channel, no memory, exposures 2-30, the same ten
formulations as the Qwen cells (toy of 2026-09-14: follows the JSON
prefix, label mass above 0.98).

Per model (the second family and, for the comparison, the two Qwen
models from the same cells):
    * paired differences J_gap(e4) - J_gap(e2) and J_gap(e7) - J_gap(e2)
      per formulation, cluster bootstrap over wordings (the two formats
      of a wording share its lexicon), the P3 decision quantity;
    * the shape statistics of exposure_shape (jump ratio to 7 peers,
      exponent for e >= 7, fraction-against-count model comparison,
      J_gap and z_bar * J_gap per exposure), reported and not predicted.
Decision rule (§24): P3 holds if both paired differences are negative
with intervals excluding zero on the ten formulations.
Output: results/phase_a/second_family.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_exposure_shape import analyse, load_mean_gaps  # noqa: E402
from scripts.phase_a_format_check import boot_mean, j_gap  # noqa: E402
from src.common import build_manifest, write_json  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="smollm1.7b,qwen0.6b,qwen1.7b")
    ap.add_argument("--e-ref", type=int, default=7)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(16)
    out = {"models": {}}
    for model in a.models.split(","):
        G = load_mean_gaps(Path(a.results), model)
        if not G:
            print(f"== {model}: no cells"); continue
        exposures = sorted({e for (_, _, e) in G})
        forms = sorted({(w, f) for (w, f, e) in G if all((w, f, x) in G for x in exposures)})
        J = {key: j_gap(g) for key, g in G.items()}
        paired = {}
        for e1 in (4, a.e_ref):
            d = np.array([J[(w, f, e1)] - J[(w, f, 2)] for (w, f) in forms])
            paired[f"e{e1}-e2"] = boot_mean(d, np.array([w for (w, _) in forms]), rng)
        shape = analyse(model, G, a.e_ref, rng)
        res = {"n_formulations": len(forms), "exposures": exposures, "P3_paired": paired,
               "J_gap_by_formulation": {f"w{w}_{f}": {str(e): J[(w, f, e)] for e in exposures} for (w, f) in forms},
               "shape": {k: shape[k] for k in ("jump", "ratio", "J_gap", "zJ_gap", "collapse") if k in shape}}
        if "exponent_e_ge_ref" in shape:
            res["shape"]["exponent_e_ge_ref"] = shape["exponent_e_ge_ref"]
        out["models"][model] = res
        print(f"== {model}: exposures {exposures}, {len(forms)} formulations")
        for k, v in paired.items():
            print(f"  P3 {k}: {v['mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] n={v['n']} excludes zero: {v['excludes_zero']}")
        print("  J_gap by exposure: " + " ".join(f"e{e}={shape['J_gap'][str(e)]:.2f}" for e in exposures))
        print("  zJ_gap by exposure: " + " ".join(f"e{e}={shape['zJ_gap'][str(e)]:.2f}" for e in exposures))
        for e in exposures:
            if str(e) in shape["ratio"]:
                q = shape["ratio"][str(e)]
                print(f"  jump({e})/jump({a.e_ref}) = {q['mean']:.2f} [{q['ci95'][0]:.2f}, {q['ci95'][1]:.2f}] n={q['n']} (fraction 1.00, count {q['pred_count']:.2f})")
        if "exponent_e_ge_ref" in shape:
            x = shape["exponent_e_ge_ref"]
            print(f"  exponent (e >= {a.e_ref}): {x['mean']:.2f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}] n={x['n']} range [{x['min']:.2f}, {x['max']:.2f}]")
        c = shape["collapse"]
        print(f"  collapse: fraction beats count (AIC) {c['frac_beats_count_aic']}; coef(fraction) {c['coef_fraction_in_both']['mean']:+.2f} "
              f"[{c['coef_fraction_in_both']['ci95'][0]:+.2f}, {c['coef_fraction_in_both']['ci95'][1]:+.2f}], coef(count) {c['coef_count_in_both']['mean']:+.3f} "
              f"[{c['coef_count_in_both']['ci95'][0]:+.3f}, {c['coef_count_in_both']['ci95'][1]:+.3f}]")
    write_json(Path(a.results) / "second_family.json", {"manifest": build_manifest(vars(a), [16]), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
