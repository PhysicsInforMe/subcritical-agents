#!/usr/bin/env python
"""Phase 0, check A: 2D Ising calibration -> results/phase0/ising_calibration.json.

Samples the equilibrium magnetisation with Swendsen-Wang for every (L, T,
seed) of the config, then estimates from the seed ensemble:
    T_c        from chi_|m| peak positions extrapolated with T_c(L) = T_c + a/L
    gamma/nu   from chi_|m| peak heights, chi_max ~ L^(gamma/nu)
    beta/nu    from <|m|>(T_B) ~ L^(-beta/nu) at the Binder-crossing temperature
    T_B        Binder crossing of consecutive L pairs, averaged
All intervals are percentile bootstraps over seeds (§11.5).
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402
from src.ising import T_C_ONSAGER, simulate  # noqa: E402
from src.measures import (binder_crossing, binder_cumulant, chi_abs, peak_parabolic,  # noqa: E402
                          powerlaw_fit, seed_bootstrap, tc_from_peaks)
from src.measures.order import moments_from_m  # noqa: E402


def _job(args):
    L, T, seed, cfg = args
    out = simulate(L, T, seed, cfg["n_equil"], cfg["n_meas"], cfg["algorithm"], cfg["thin"])
    mo = moments_from_m(out["m"])
    return {"L": L, "T": T, "seed": seed, "chi_abs": chi_abs(out["m"], L * L, T),
            "binder": binder_cumulant(out["m"]), "abs_m": mo["abs_m"], "e": float(out["e"].mean())}


def _curves(rows, L_values, T_grid):
    """Average per (L, T) over the given rows -> dict of arrays indexed by L."""
    cur = {L: {"chi": np.zeros(T_grid.size), "U": np.zeros(T_grid.size),
               "abs_m": np.zeros(T_grid.size), "n": np.zeros(T_grid.size)} for L in L_values}
    Tidx = {round(float(T), 6): i for i, T in enumerate(T_grid)}
    for r in rows:
        i = Tidx[round(float(r["T"]), 6)]
        c = cur[r["L"]]
        c["chi"][i] += r["chi_abs"]; c["U"][i] += r["binder"]
        c["abs_m"][i] += r["abs_m"]; c["n"][i] += 1
    for c in cur.values():
        for key in ("chi", "U", "abs_m"):
            c[key] = c[key] / np.maximum(c["n"], 1)
    return cur


def _estimates(rows, L_values, T_grid, fine_mask):
    cur = _curves(rows, L_values, T_grid)
    Ls = np.array(L_values, dtype=float)
    Tf = T_grid[fine_mask]
    peaks = [peak_parabolic(Tf, cur[L]["chi"][fine_mask]) for L in L_values]
    T_peak = np.array([p[0] for p in peaks])
    chi_max = np.array([p[1] for p in peaks])
    Tc, _ = tc_from_peaks(Ls, T_peak, nu=1.0)
    g_nu, _ = powerlaw_fit(Ls, chi_max)
    crossings = [binder_crossing(Tf, cur[L_values[i]]["U"][fine_mask], cur[L_values[i + 1]]["U"][fine_mask])
                 for i in range(len(L_values) - 1)]
    crossings = [c for c in crossings if np.isfinite(c)]
    if not crossings:
        raise ValueError("no Binder crossing in the fine window")
    T_B = float(np.mean(crossings))
    abs_m_at = np.array([np.interp(T_B, Tf, cur[L]["abs_m"][fine_mask]) for L in L_values])
    minus_b_nu, _ = powerlaw_fit(Ls, abs_m_at)
    return {"Tc": Tc, "gamma_over_nu": g_nu, "beta_over_nu": -minus_b_nu, "binder_crossing": T_B,
            "T_peak": T_peak.tolist(), "chi_max": chi_max.tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0_ising.yaml")
    ap.add_argument("--out", default="results/phase0/ising_calibration.json")
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke tests")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.quick:
        cfg.update(L_values=[8, 12, 16], seeds=[1, 2, 3], n_equil=200, n_meas=800, n_boot=50)
    f = cfg["T_fine"]
    T_fine = np.linspace(f["start"], f["stop"], f["num"])
    T_grid = np.unique(np.concatenate([T_fine, np.array(cfg["T_coarse"], dtype=float)]))
    fine_mask = (T_grid >= f["start"] - 1e-9) & (T_grid <= f["stop"] + 1e-9)
    jobs = [(L, float(T), s, cfg) for L in cfg["L_values"] for T in T_grid for s in cfg["seeds"]]
    print(f"{len(jobs)} runs on {cfg.get('workers', 4)} workers", flush=True)
    with ProcessPoolExecutor(max_workers=cfg.get("workers", 4)) as ex:
        rows = list(ex.map(_job, jobs, chunksize=4))
    by_seed = {s: [r for r in rows if r["seed"] == s] for s in cfg["seeds"]}
    per_seed = list(by_seed.values())

    def stat(key):
        return lambda seeds_rows: _estimates([r for rr in seeds_rows for r in rr],
                                             cfg["L_values"], T_grid, fine_mask)[key]

    rng = np.random.default_rng(0)
    result = {"manifest": build_manifest(cfg, cfg["seeds"], {"T_grid": T_grid.tolist(),
                                                              "T_c_onsager": T_C_ONSAGER}),
              "L_values": cfg["L_values"]}
    for key in ("Tc", "gamma_over_nu", "beta_over_nu", "binder_crossing"):
        est, lo, hi = seed_bootstrap(per_seed, stat(key), n_boot=cfg["n_boot"], rng=rng)
        result[key] = {"estimate": est, "ci95": [lo, hi]}
        print(f"{key:16s} {est:.4f} [{lo:.4f}, {hi:.4f}]", flush=True)
    full = _estimates(rows, cfg["L_values"], T_grid, fine_mask)
    result["peaks"] = {"T_peak": full["T_peak"], "chi_max": full["chi_max"]}
    cur = _curves(rows, cfg["L_values"], T_grid)
    result["curves"] = {str(L): {"T": T_grid.tolist(), "chi_abs": cur[L]["chi"].tolist(),
                                 "binder": cur[L]["U"].tolist(), "abs_m": cur[L]["abs_m"].tolist()}
                        for L in cfg["L_values"]}
    result["raw"] = rows
    write_json(a.out, result)
    print("written", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
