#!/usr/bin/env python
"""Mean-field criterion without the linear assumption (the results notes (not included), test 4; method of Flint et al. PNAS 2026, §10.7: fixed points and
linear stability of the mean-field map built on the tabulated policy).

For a tabulated policy P(+ | k) with e peers drawn at random from a large
all-to-all group at magnetisation m, the number of + peers is
Bin(e, (1+m)/2) and k = 2 n_+ - e. The mean-field map of one asynchronous
sweep is
    F(m) = 2 * sum_n Bin(n; e, (1+m)/2) * P(+ | k = 2n - e) - 1,
and m* is a fixed point if F(m*) = m*, stable if |F'(m*)| < 1. The
disordered state m = 0 is unstable (spontaneous ordering) if F'(0) > 1;
with a label bias the symmetric point is not m = 0 and the relevant
question is whether there is one fixed point (field-dominated or
subcritical) or three (bistable). The global-flip symmetrised policy,
p_sym(k) = (p_A(k) + 1 - p_A(-k)) / 2 with p_A the mean over orderings of the
probabilities, puts the fixed point back at m = 0 and gives the textbook
criterion F'(0) versus 1; it does NOT remove the bias from the slope: since
F_B(m) = -F_A(-m), F_sym'(0) = F_A'(0), the slope of a population half in
each sector (a bimodal random field +-h), which in the linear regime is
e J sech^2(h) (round 5, 2026-10-24; the bias-free slope is the odd part of
the table in logit space, scripts/phase_a_F0_decomposition.py). Both maps
are computed. z_bar * J_gap is reported next to F'(0) for comparison:
for a linear response P = sigma(2(h + J k)) at h = 0 one has
F'(0) = 2 e J * sigma'(0) * 2 = e J, so the two criteria coincide only in
the linear, unbiased case.

Reads every list, k_mem = 0 gaps.json (all exposures, all formulations),
T = 1. Output: results/phase_a/meanfield_stability.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402


def policy_from_entries(entries: list[dict], T: float = 1.0) -> tuple[dict[int, float], dict[int, float]]:
    """Ordering-averaged P(+ | k) in sector A and the flip-symmetrised table."""
    acc: dict[int, list[float]] = {}
    for x in entries:
        acc.setdefault(x["k"], []).append(1.0 / (1.0 + np.exp(-x["gap"] / T)))
    pA = {k: float(np.mean(v)) for k, v in acc.items()}
    p_sym = {k: 0.5 * (pA[k] + (1.0 - pA[-k])) for k in pA if -k in pA}
    return pA, p_sym


def mf_map(p: dict[int, float], e: int):
    ns = np.arange(e + 1)
    ks = 2 * ns - e
    pk = np.array([p[int(k)] for k in ks])

    def F(m: float) -> float:
        w = binom.pmf(ns, e, (1.0 + m) / 2.0)
        return float(2.0 * np.dot(w, pk) - 1.0)

    def dF(m: float, h: float = 1e-4) -> float:
        return (F(min(m + h, 1.0)) - F(max(m - h, -1.0))) / (min(m + h, 1.0) - max(m - h, -1.0))

    return F, dF


def fixed_points(F, dF, n_grid: int = 2001) -> list[dict]:
    ms = np.linspace(-1.0, 1.0, n_grid)
    g = np.array([F(m) - m for m in ms])
    out = []
    for i in range(len(ms) - 1):
        if g[i] == 0.0 or g[i] * g[i + 1] < 0:
            lo, hi = ms[i], ms[i + 1]
            for _ in range(40):  # bisection
                mid = 0.5 * (lo + hi)
                if (F(lo) - lo) * (F(mid) - mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            m_star = 0.5 * (lo + hi)
            out.append({"m": float(m_star), "slope": float(dF(m_star)), "stable": bool(abs(dF(m_star)) < 1.0)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="qwen0.6b,qwen1.7b")
    ap.add_argument("--T", type=float, default=1.0)
    a = ap.parse_args(argv)
    out = {"T": a.T, "models": {}}
    for model in a.models.split(","):
        rows = []
        # suffix "all" only: the cells of the ten original formulations (revision R4-01)
        for path in sorted(glob.glob(str(Path(a.results) / f"A_{model}_list_k0_e*_Tan_f0_all_N*/gaps.json"))):
            cell = Path(path).parent.name
            e = int(re.search(r"_e(\d+)_", cell).group(1))
            for tab in json.load(open(path, encoding="utf-8"))["tables"]:
                pA, ps = policy_from_entries(tab["entries"], a.T)
                FA, dFA = mf_map(pA, e); FS, dFS = mf_map(ps, e)
                fpA, fpS = fixed_points(FA, dFA), fixed_points(FS, dFS)
                # linear summary for comparison: gap-based J and z_bar J
                ks = np.array(sorted({x["k"] for x in tab["entries"]}), float)
                gmean = np.array([np.mean([x["gap"] for x in tab["entries"] if x["k"] == k]) for k in ks])
                J = float(np.polyfit(ks, gmean, 1)[0] / 2)
                rows.append({"cell": cell, "exposure": e, "wording": tab["wording"], "format": tab["format"],
                             "n_orderings_max": max(len([x for x in tab["entries"] if x["k"] == k]) for k in ks),
                             "zJ_gap": e * J, "dF0_sym": float(dFS(0.0)), "ordered_sym": bool(dFS(0.0) > 1.0),
                             "fixed_points_sym": fpS, "n_stable_sym": sum(f["stable"] for f in fpS),
                             "fixed_points_A": fpA, "n_stable_A": sum(f["stable"] for f in fpA),
                             "stable_m_A": [round(f["m"], 3) for f in fpA if f["stable"]]})
        out["models"][model] = rows
        print(f"== {model}: {len(rows)} (cell, formulation) tables")
        for e in sorted({r["exposure"] for r in rows}):
            sub = [r for r in rows if r["exposure"] == e]
            print(f"  e={e:2d}: F'(0) sym mean {np.mean([r['dF0_sym'] for r in sub]):5.2f} "
                  f"(min {min(r['dF0_sym'] for r in sub):5.2f}, max {max(r['dF0_sym'] for r in sub):5.2f}); "
                  f"ordered (F'(0)>1) in {sum(r['ordered_sym'] for r in sub)}/{len(sub)}; "
                  f"zJ_gap>1 in {sum(r['zJ_gap'] > 1 for r in sub)}/{len(sub)}; "
                  f"sector A with 2 stable fixed points (bistable) in {sum(r['n_stable_A'] >= 2 for r in sub)}/{len(sub)}")
    write_json(Path(a.results) / "meanfield_stability.json", {"manifest": build_manifest(vars(a), []), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
