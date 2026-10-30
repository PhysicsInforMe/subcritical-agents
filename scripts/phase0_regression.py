#!/usr/bin/env python
"""Phase 0, check B: synthetic recovery of (J, h) and (J, h_bias, h_spec)
-> results/phase0/regression_recovery.json (design notes §5, §14.1 B1).

Point fits use the maximum-likelihood estimator with a percentile bootstrap
over updates; the recovery study repeats the draw many times and reports
coverage of the 95 % interval and relative bias (standard of Brockers et
al., §10.12). The De Nobili count-based estimator is run alongside and its
agreement with the MLE is reported.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402
from src.measures import (fit_glauber_counts, fit_glauber_mle, recovery_study,  # noqa: E402
                          synthetic_updates)


def _pack(fit, names):
    return {n: {"estimate": fit[n]["estimate"], "ci95": fit[n]["ci95"]} for n in names}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0_regression.yaml")
    ap.add_argument("--out", default="results/phase0/regression_recovery.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.quick:
        cfg.update(n_updates=5000, n_boot=50, recovery_draws=20, recovery_boot=30)
    rng = np.random.default_rng(cfg["seed"])

    two = cfg["two_term"]
    d2 = synthetic_updates(cfg["n_updates"], two["J"], two["h"], with_truth=False, rng=rng)
    fit2 = fit_glauber_mle(d2["k"], d2["s_new"], None, n_boot=cfg["n_boot"], rng=rng)
    cnt2 = fit_glauber_counts(d2["k"], d2["s_new"], None, n_boot=cfg["n_boot"], rng=rng)

    three = cfg["three_term"]
    d3 = synthetic_updates(cfg["n_updates"], three["J"], three["h_bias"], three["h_spec"], rng=rng)
    fit3 = fit_glauber_mle(d3["k"], d3["s_new"], d3["y"], n_boot=cfg["n_boot"], rng=rng)
    cnt3 = fit_glauber_counts(d3["k"], d3["s_new"], d3["y"], n_boot=cfg["n_boot"], rng=rng)

    rec = recovery_study(three, cfg["recovery_updates"], cfg["recovery_draws"],
                         n_boot=cfg["recovery_boot"], rng=rng)
    result = {
        "manifest": build_manifest(cfg, [cfg["seed"]]),
        "two_term": {"true": two, "fit": _pack(fit2, ("J", "h")),
                     "fit_counts": _pack(cnt2, ("J", "h"))},
        "three_term": {"true": three, "fit": _pack(fit3, ("J", "h_bias", "h_spec")),
                       "fit_counts": _pack(cnt3, ("J", "h_bias", "h_spec"))},
        "recovery_study": rec,
    }
    for name, blk in (("two", result["two_term"]), ("three", result["three_term"])):
        for p, v in blk["fit"].items():
            print(f"{name:5s} {p:7s} {v['estimate']:.4f} {v['ci95']} true {blk['true'][p]}")
    print("recovery", rec)
    write_json(a.out, result)
    print("written", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
