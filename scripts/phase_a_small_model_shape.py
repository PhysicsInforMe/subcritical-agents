#!/usr/bin/env python
"""Tests A and B of the results notes (not included) on the existing tabulations:
why does the exponent of the unanimity jump vary across formulations in
the small model?

Test A (noise): for every formulation, the exponent of jump(e) ~ e^x over
e >= e_ref with a bootstrap over the orderings inside each (e, k) cell:
resample the gaps of the two unanimous configurations at each e (they
have a single ordering, so the resampling acts on the neighbouring cells
used for the jump only through the mean gap per k; to give the jump a
proper uncertainty the jump is recomputed as the fitted gap at k = +-e
from a linear fit on all k, bootstrapping the gaps inside every k).
Test B (position): for every formulation and exposure, regression of the
gap of each ordering on [1, k, s_first, s_last]; the k-only jump 2 b e and
its exponent over e >= e_ref; the position coefficients with their
bootstrap interval. Formulation 'lines' or 'comma' reported.
Output: results/phase_a/small_model_shape.json and a text summary.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402


def load(results: Path, model: str) -> dict[tuple, dict[int, list[tuple[tuple[int, ...], float]]]]:
    """{(wording, format, e): {k: [(states, gap), ...]}} for list, k_mem = 0 cells."""
    out: dict = {}
    for path in glob.glob(str(results / f"A_{model}_list_k0_e*_Tan_f0_all_N*/gaps.json")):
        e = int(re.search(r"_e(\d+)_", Path(path).parent.name).group(1))
        for tab in json.load(open(path, encoding="utf-8"))["tables"]:
            byk: dict[int, list] = {}
            for x in tab["entries"]:
                byk.setdefault(x["k"], []).append((tuple(x["states"]), x["gap"]))
            out[(tab["wording"], tab["format"], e)] = byk
    return out


def jump_linear(byk: dict[int, list], rng: np.random.Generator | None) -> float:
    """Jump 2 b e from the linear fit of the mean gap on k (resampled inside k if rng)."""
    ks = np.array(sorted(byk), float)
    g = []
    for k in ks:
        gaps = np.array([v for _, v in byk[int(k)]])
        if rng is not None:
            gaps = rng.choice(gaps, size=gaps.size, replace=True)
        g.append(gaps.mean())
    b = np.polyfit(ks, np.array(g), 1)[0]
    return float(b * (ks.max() - ks.min()))


def exponent(es: list[int], jumps: list[float]) -> float | None:
    if min(jumps) <= 0.5:
        return None
    return float(np.polyfit(np.log(es), np.log(jumps), 1)[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="qwen0.6b,qwen1.7b")
    ap.add_argument("--e-ref", type=int, default=7)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(3)
    out = {"e_ref": a.e_ref, "models": {}}
    for model in a.models.split(","):
        G = load(Path(a.results), model)
        es = sorted({e for (_, _, e) in G if e >= a.e_ref})
        forms = sorted({(w, f) for (w, f, e) in G})
        rows = []
        print(f"== {model}: exposures {es}, {len(forms)} formulations")
        for (w, f) in forms:
            if not all((w, f, e) in G for e in es):
                continue
            # test A: exponent of the linear-fit jump with ordering bootstrap
            jumps = [jump_linear(G[(w, f, e)], None) for e in es]
            x0 = exponent(es, jumps)
            boots = []
            for _ in range(a.n_boot):
                jb = [jump_linear(G[(w, f, e)], rng) for e in es]
                xb = exponent(es, jb)
                if xb is not None:
                    boots.append(xb)
            ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if len(boots) > 50 else None
            # test B: position regression per exposure
            pos = {}
            k_only_jumps = []
            for e in es:
                X, y = [], []
                for k, lst in G[(w, f, e)].items():
                    for states, gap in lst:
                        X.append([1.0, float(k), float(states[0]), float(states[-1])]); y.append(gap)
                X, y = np.array(X), np.array(y)
                c, *_ = np.linalg.lstsq(X, y, rcond=None)
                cb = []
                for _ in range(300):
                    idx = rng.integers(0, len(y), len(y))
                    cb.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0])
                cb = np.array(cb)
                pos[str(e)] = {"b_k": float(c[1]), "c_first": float(c[2]), "c_first_ci": np.percentile(cb[:, 2], [2.5, 97.5]).tolist(),
                               "c_last": float(c[3]), "c_last_ci": np.percentile(cb[:, 3], [2.5, 97.5]).tolist(), "n": int(len(y))}
                k_only_jumps.append(float(c[1] * 2 * e))
            x_konly = exponent(es, k_only_jumps)
            row = {"wording": w, "format": f, "jumps_linear": jumps, "exponent": x0, "exponent_ci_orderings": ci,
                   "position": pos, "k_only_jumps": k_only_jumps, "exponent_k_only": x_konly}
            rows.append(row)
            sig = [e for e in es if not (pos[str(e)]["c_first_ci"][0] <= 0 <= pos[str(e)]["c_first_ci"][1]) or
                   not (pos[str(e)]["c_last_ci"][0] <= 0 <= pos[str(e)]["c_last_ci"][1])]
            print(f"  w{w} {f:6s}: jumps {np.round(jumps, 2).tolist()} exponent {x0 if x0 is None else round(x0, 2)} "
                  f"CI(orderings) {None if ci is None else [round(c_, 2) for c_ in ci]} | k-only exponent "
                  f"{x_konly if x_konly is None else round(x_konly, 2)}; position coef. significant at e = {sig}; "
                  f"c_first at e=30 {pos[str(es[-1])]['c_first']:+.2f}, c_last {pos[str(es[-1])]['c_last']:+.2f}")
        out["models"][model] = {"exposures": es, "rows": rows}
    write_json(Path(a.results) / "small_model_shape.json", {"manifest": build_manifest(vars(a), [3]), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
