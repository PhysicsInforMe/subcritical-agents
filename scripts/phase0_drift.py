#!/usr/bin/env python
"""Phase 0, check C: voter-model drift control -> results/phase0/drift_control.json.

Consensus time in single updates for the neutral voter model on the complete
graph must grow as N^2 (design notes §10.21). The exponent is fitted on the
log-log means and bootstrapped over runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402
from src.dynamics import voter_consensus_time  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0_drift.yaml")
    ap.add_argument("--out", default="results/phase0/drift_control.json")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    Ns = cfg["N_values"]
    seeds = list(range(cfg["runs"]))
    times = {N: np.array([voter_consensus_time(N, cfg["seed"] * 1000 + s) for s in seeds]) for N in Ns}
    assert all((t > 0).all() for t in times.values()), "a run hit max_steps"
    logN = np.log(np.array(Ns, dtype=float))

    def slope(sample_idx):
        means = np.array([times[N][sample_idx].mean() for N in Ns])
        return np.polyfit(logN, np.log(means), 1)[0]

    rng = np.random.default_rng(0)
    point = slope(np.arange(len(seeds)))
    boots = [slope(rng.integers(0, len(seeds), size=len(seeds))) for _ in range(cfg["n_boot"])]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    result = {"manifest": build_manifest(cfg, seeds), "N_values": Ns,
              "t_cons_exponent": {"estimate": float(point), "ci95": [float(lo), float(hi)]},
              "t_cons_mean": {str(N): float(times[N].mean()) for N in Ns},
              "t_cons_over_N2": {str(N): float(times[N].mean() / N ** 2) for N in Ns}}
    print(f"exponent {point:.3f} [{lo:.3f}, {hi:.3f}]; t/N^2:", result["t_cons_over_N2"])
    write_json(a.out, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
