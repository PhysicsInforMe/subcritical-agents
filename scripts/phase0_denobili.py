#!/usr/bin/env python
"""Phase 0, check D: De Nobili check with logits -> results/phase0/denobili_check.json.

For each model: tabulate the logit gap for every ordering of the 4 neighbour
states in both flip sectors (one forward pass each, 32 passes per model),
then for each temperature of the config
    * fit h_tilde, J_tilde in label space on sector A (h_tilde is the label
      bias toward A, the quantity De Nobili report as h_tilde >> J_tilde),
    * fit J_tilde on the sector-symmetrised gaps (the table used by the
      dynamics under global flip; its h is zero by construction because the
      flipped sector is the same set of prompts with the labels swapped, so
      merging sectors makes the gap an odd function of k),
    * report the variance across orderings per k (B3),
    * run the Kolmogorov cycle test on the symmetrised table (§10.28),
    * run the numpy Glauber dynamics on an L x L lattice with global flip and
      compute chi_|m| over a log-spaced temperature scan (B4: same logits,
      rescaled), wide enough to see the crossover when |gap| >> 1.

``--reprocess`` recomputes everything from the raw gaps stored in an
existing JSON without touching the model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402
from src.dynamics import TabulatedPolicy, run_tabulated  # noqa: E402
from src.llm import LogitReader, cycle_test, fit_table, policy_at_temperature, tabulate_policy  # noqa: E402
from src.llm.tabulate import DEFAULT_TEMPLATE  # noqa: E402
from src.measures import chi_abs  # noqa: E402
from src.topology import adjacency_lists, lattice_2d  # noqa: E402


def _policy_from_gaps(sector_gaps: dict, T: float) -> tuple[dict[int, float], dict[int, float]]:
    """Ordering-averaged P(+|k) and its variance across orderings."""
    p, v = {}, {}
    for k, blk in sector_gaps.items():
        probs = np.array([policy_at_temperature(g, 0.0, T) for g in blk["gap"]])
        p[int(k)] = float(probs.mean())
        v[int(k)] = float(probs.var())
    return p, v


def _chi_curve(gaps_A: dict, gaps_B: dict, L: int, T_scan: np.ndarray, seeds: list[int],
               n_sweeps: int, burn: float):
    """chi_|m| versus temperature on the lattice, averaging both flip sectors."""
    adj = adjacency_lists(lattice_2d(L))
    y = np.zeros(L * L, dtype=int)
    chi, err = [], []
    for T in T_scan:
        per_seed = []
        for s in seeds:
            ms = []
            for gaps in (gaps_A, gaps_B):
                p, _ = _policy_from_gaps(gaps, T)
                pol = TabulatedPolicy(4, {(k, 0): min(max(pk, 1e-12), 1 - 1e-12) for k, pk in p.items()})
                rng = np.random.default_rng(s)
                s0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=L * L)
                out = run_tabulated(pol, adj, s0, y, n_sweeps=n_sweeps, seed=s)
                ms.append(out["m"][int(burn * n_sweeps):])
            per_seed.append(chi_abs(np.concatenate(ms), L * L, T))
        chi.append(float(np.mean(per_seed)))
        err.append(float(np.std(per_seed, ddof=1) / np.sqrt(len(per_seed))))
    return chi, err


def cells_from_gaps(model_name: str, revision: str, gaps: dict, cfg: dict, T_scan: np.ndarray) -> list[dict]:
    gA, gB = gaps["plus_is_A"], gaps["plus_is_B"]
    gaps_A_only = {k: gA[k]["gap"] for k in gA}
    merged = {k: gA[k]["gap"] + gB[k]["gap"] for k in gA}
    chi, err = _chi_curve(gA, gB, cfg["chi_scan"]["L"], T_scan, cfg["chi_scan"]["seeds"],
                          cfg["chi_scan"]["n_sweeps"], cfg["chi_scan"]["burn"])
    cells = []
    for T in cfg["temperatures"]:
        pA, vA = _policy_from_gaps(gA, T)
        pB, vB = _policy_from_gaps(gB, T)
        p_sym = {k: 0.5 * (pA[k] + pB[k]) for k in pA}
        fit_label = fit_table(gaps_A_only, T, n_boot=cfg["n_boot"])
        fit_sym = fit_table(merged, T, n_boot=cfg["n_boot"])
        cell = {
            "model": model_name, "revision": revision, "temperature": T,
            "policy": {str(k): pA[k] for k in sorted(pA)},
            "policy_flipped": {str(k): pB[k] for k in sorted(pB)},
            "policy_symmetric": {str(k): p_sym[k] for k in sorted(p_sym)},
            "fit": fit_label,                 # label space, sector A: h = label bias
            "fit_symmetric": fit_sym,         # h = 0 by construction, J from odd part
            "order_variance": {str(k): 0.5 * (vA[k] + vB[k]) for k in sorted(vA)},
            "cycle_test": cycle_test(p_sym, seed=0),
            "n_tokens_prompt": gA["0"]["n_tokens"],
            "chi_abs_curve": {"T": T_scan.tolist(), "chi_abs": chi, "chi_abs_err": err},
        }
        ratio = abs(fit_label["h"]["estimate"]) / max(abs(fit_label["J"]["estimate"]), 1e-12)
        print(f"  T={T}: label-space h={fit_label['h']['estimate']:+.3f} J={fit_label['J']['estimate']:+.3f} "
              f"|h|/|J|={ratio:.2f}; symmetric J={fit_sym['J']['estimate']:+.3f}; "
              f"cycle={cell['cycle_test']['mean_abs_cycle_sum']:.3f}", flush=True)
        cells.append(cell)
    return cells


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0_denobili.yaml")
    ap.add_argument("--out", default="results/phase0/denobili_check.json")
    ap.add_argument("--reprocess", action="store_true", help="recompute from raw gaps in --out")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    template = cfg.get("template") or DEFAULT_TEMPLATE
    sc = cfg["chi_scan"]
    T_scan = np.geomspace(sc["T_min"], sc["T_max"], sc["n_T"])
    cells, raw = [], {}
    if a.reprocess:
        raw = json.load(open(a.out, encoding="utf-8"))["raw"]
        for name, r in raw.items():
            print(f"{name}: reprocessing from raw gaps", flush=True)
            cells += cells_from_gaps(name, r["revision"], r["gaps"], cfg, T_scan)
    else:
        for model in cfg["models"]:
            t0 = time.time()
            reader = LogitReader(model["name"], model.get("revision"))
            gaps = tabulate_policy(reader, z=4, template=template)
            raw[model["name"]] = {"gaps": gaps, "revision": reader.resolved_revision,
                                  "variant_ids": reader.variant_ids, "seconds": time.time() - t0}
            print(f"{model['name']}: tabulated in {time.time() - t0:.1f} s, "
                  f"revision {reader.resolved_revision}", flush=True)
            cells += cells_from_gaps(model["name"], reader.resolved_revision, gaps, cfg, T_scan)
    result = {"manifest": build_manifest(cfg, sc["seeds"], {"template": template}),
              "cells": cells, "raw": raw}
    write_json(a.out, result)
    print("written", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
