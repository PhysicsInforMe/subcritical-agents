#!/usr/bin/env python
"""H10 as a declared test (the project log (not included), 2026-09-17, written before the data):
does the shape of the response to the number of peers depend on the
coupling strength of the formulation? Weak formulations keep growing with
the peers shown (count-like exponent near 1), strong ones are flat
(fraction-like exponent near 0). Alternative H_ceiling: strong
formulations cannot grow because the logit gap has a ceiling.

Data: list channel, no memory, exposures 2-30, 8-ordering cells; the
twenty formulations of the two Qwen models (both format files) and the
ten of SmolLM2-1.7B, whose per-formulation correlation has never been
looked at (out-of-family test). Per formulation: unanimity jump
gap(+e) - gap(-e) at every e, exponent of the jump over e >= 7 and over
e >= 10 (log-log slope, formulations with a jump below 0.5 logit at some
e in the range excluded).

(a) Spearman(exponent e>=7, jump at 7) and Spearman(exponent e>=10, jump
    at 2) with p-values and a cluster bootstrap (wordings) interval.
(b) Paired form of the N-solid criterion: formulations split at the
    median of the jump at 2 peers (outside the fit range); mean exponent
    of the weak half minus the strong half, cluster bootstrap interval.
(c) Ceiling control: sd across formulations of the jump at 30 over the
    sd at 7 (below 1: jumps converge to a common ceiling).
Decision rule in the log entry. Output: results/phase_a/h10_test.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_format_check import load_mean_gaps  # noqa: E402
from src.common import build_manifest, write_json  # noqa: E402


def exponent(G: dict, w: int, f: str, es: list[int]) -> float | None:
    jumps = [G[(w, f, e)][e] - G[(w, f, e)][-e] for e in es]
    if len(es) < 3 or min(jumps) <= 0.5:
        return None
    return float(np.polyfit(np.log(es), np.log(jumps), 1)[0])


def cboot(vals: np.ndarray, words: np.ndarray, rng: np.random.Generator, stat, n: int = 4000) -> list[float]:
    """Cluster bootstrap over wordings. With five clusters some resamples
    lack one group (or have a constant statistic) and give NaN: they are
    dropped and the valid fraction is returned as the third element."""
    uniq = np.unique(words)
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([np.flatnonzero(words == c) for c in pick])
        out.append(stat(idx))
    out = np.array(out, float)
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5)), float(np.mean(~np.isnan(out)))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="qwen0.6b,qwen1.7b,smollm1.7b")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(10)
    out = {"models": {}}
    for model in a.models.split(","):
        G = load_mean_gaps(Path(a.results), model)
        es = sorted({e for (_, _, e) in G})
        forms = sorted({(w, f) for (w, f, e) in G if all((w, f, x) in G for x in es)})
        big7 = [e for e in es if e >= 7]; big10 = [e for e in es if e >= 10]
        rows = []
        for (w, f) in forms:
            rows.append({"wording": w, "format": f,
                         "jump2": G[(w, f, 2)][2] - G[(w, f, 2)][-2], "jump7": G[(w, f, 7)][7] - G[(w, f, 7)][-7],
                         "jump30": G[(w, f, 30)][30] - G[(w, f, 30)][-30],
                         "exp7": exponent(G, w, f, big7), "exp10": exponent(G, w, f, big10)})
        res: dict = {"n_formulations": len(forms), "rows": rows, "a": {}, "b": {}, "c": {}}
        # (a) correlations
        for name, xk, yk in (("exp7_vs_jump7", "exp7", "jump7"), ("exp10_vs_jump2", "exp10", "jump2")):
            sub = [r for r in rows if r[xk] is not None]
            x = np.array([r[xk] for r in sub]); y = np.array([r[yk] for r in sub]); wd = np.array([r["wording"] for r in sub])
            rho, p = spearmanr(x, y)
            ci = cboot(x, wd, rng, lambda idx: spearmanr(x[idx], y[idx])[0] if len(set(idx)) > 2 else np.nan)
            res["a"][name] = {"rho": float(rho), "p": float(p), "n": int(len(sub)), "ci95_cluster": ci,
                              "passes": bool(rho < -0.5 and p < 0.05)}
        # (b) paired split by the jump at 2 peers
        for xk in ("exp7", "exp10"):
            sub = [r for r in rows if r[xk] is not None]
            med = float(np.median([r["jump2"] for r in sub]))
            weak = np.array([r[xk] for r in sub if r["jump2"] <= med]); strong = np.array([r[xk] for r in sub if r["jump2"] > med])
            allx = np.array([r[xk] for r in sub]); grp = np.array([r["jump2"] <= med for r in sub]); wd = np.array([r["wording"] for r in sub])
            diff = float(weak.mean() - strong.mean())
            ci = cboot(allx, wd, rng, lambda idx: (allx[idx][grp[idx]].mean() - allx[idx][~grp[idx]].mean())
                       if grp[idx].any() and (~grp[idx]).any() else np.nan)
            res["b"][xk] = {"median_jump2": med, "mean_weak": float(weak.mean()), "mean_strong": float(strong.mean()),
                            "n_weak": int(weak.size), "n_strong": int(strong.size), "diff_weak_minus_strong": diff,
                            "ci95_cluster": ci, "passes": bool(ci[0] > 0)}
        # (c) ceiling control
        j7 = np.array([r["jump7"] for r in rows]); j30 = np.array([r["jump30"] for r in rows])
        res["c"] = {"sd_jump7": float(j7.std(ddof=1)), "sd_jump30": float(j30.std(ddof=1)),
                    "ratio_sd30_over_sd7": float(j30.std(ddof=1) / j7.std(ddof=1)),
                    "mean_jump7": float(j7.mean()), "mean_jump30": float(j30.mean())}
        out["models"][model] = res
        print(f"== {model}: {len(forms)} formulations")
        for k, v in res["a"].items():
            print(f"  (a) {k}: rho {v['rho']:+.2f} (p = {v['p']:.3f}, n = {v['n']}), cluster CI [{v["ci95_cluster"][0]:+.2f}, {v["ci95_cluster"][1]:+.2f}] (valid {v["ci95_cluster"][2]:.2f}) -> {'pass' if v['passes'] else 'FAIL'}")
        for k, v in res["b"].items():
            print(f"  (b) {k}: weak {v['mean_weak']:.2f} (n {v['n_weak']}) - strong {v['mean_strong']:.2f} (n {v['n_strong']}) = {v['diff_weak_minus_strong']:+.2f} "
                  f"[{v["ci95_cluster"][0]:+.2f}, {v["ci95_cluster"][1]:+.2f}] (valid {v["ci95_cluster"][2]:.2f}) -> {"pass" if v["passes"] else "FAIL"}")
        c = res["c"]
        print(f"  (c) sd jump 7 -> 30: {c['sd_jump7']:.2f} -> {c['sd_jump30']:.2f} (ratio {c['ratio_sd30_over_sd7']:.2f}); mean jump {c['mean_jump7']:.2f} -> {c['mean_jump30']:.2f}")
    write_json(Path(a.results) / "h10_test.json", {"manifest": build_manifest(vars(a), [10]), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
