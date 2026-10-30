#!/usr/bin/env python
"""Verification against Flint et al. (§10.7, §18 decision 3): does the
tabulated policy depend on the memory only through the sum of the past
states, or on the full sequence?

Flint et al. keep the memory as a full state (all tuples). Our label-space
fit uses one coefficient on the sum. On every k_mem = 3 cell we compare, on
the tabulated gaps (T = 1, sector A, ordering-averaged per (k, memory)):
    M_sum   gap ~ h + J k + J_mem * sum(memory)             (3 parameters)
    M_pos   gap ~ h + J k + a1 m1 + a2 m2 + a3 m3            (5 parameters,
            one weight per position, oldest first)
    M_full  gap ~ h + J k + one dummy per memory tuple      (2 + 7 parameters)
and report R^2, AIC and the position weights. If M_pos beats M_sum with the
most recent state weighing most, the sum is a coarse summary and the
paper reports the recency profile; if M_full beats M_pos by a lot, the
memory acts as a state, as in Flint, and the sum is not adequate.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np


def fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, p = X.shape
    rss = float((resid ** 2).sum())
    r2 = 1 - rss / float(((y - y.mean()) ** 2).sum())
    aic = n * np.log(rss / n) + 2 * p
    return coef, r2, float(aic)


def main() -> int:
    tuples = list(itertools.product((-1, 1), repeat=3))
    for gp in sorted(Path("results/phase_a").glob("*k3*/gaps.json")):
        d = json.load(open(gp, encoding="utf-8"))
        r2s = {"sum": [], "pos": [], "full": []}
        aics = {"sum": [], "pos": [], "full": []}
        weights = []
        for tab in d["tables"]:
            acc: dict[tuple, list[float]] = {}
            for e in tab["entries"]:
                acc.setdefault((e["k"], tuple(e["memory"])), []).append(e["gap"])
            keys = sorted(acc)
            y = np.array([np.mean(acc[k]) for k in keys])
            k = np.array([kk for kk, _ in keys], dtype=float)
            M = np.array([m for _, m in keys], dtype=float)
            one = np.ones_like(k)
            X_sum = np.column_stack([one, k, M.sum(axis=1)])
            X_pos = np.column_stack([one, k, M])
            dummies = np.array([[1.0 if tuple(int(v) for v in m) == t else 0.0 for t in tuples[1:]] for m in M])
            X_full = np.column_stack([one, k, dummies])
            for name, X in (("sum", X_sum), ("pos", X_pos), ("full", X_full)):
                c, r2, aic = fit(X, y)
                r2s[name].append(r2); aics[name].append(aic)
                if name == "pos":
                    weights.append(c[2:5] / 2.0)
        w = np.array(weights)
        print(f"{d['cell_id']}: R2 sum {np.mean(r2s['sum']):.3f} | pos {np.mean(r2s['pos']):.3f} | full {np.mean(r2s['full']):.3f}; "
              f"AIC sum-pos {np.mean(aics['sum']) - np.mean(aics['pos']):+.1f}, pos-full {np.mean(aics['pos']) - np.mean(aics['full']):+.1f}; "
              f"position weights (oldest..newest) {np.round(w.mean(axis=0), 2).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
