#!/usr/bin/env python
"""Phase A control on the L = 6 lattice (design notes §5 "reticolo L = 6 come
controllo", §17 "misurato e non ancora analizzato a parte").

The lattice cells reuse the policy of the all-to-all cell with exposure 4
(same P(+ | k) table, k = sum of the 4 lattice neighbours instead of 4
random peers), so the comparison isolates the topology at fixed policy:
mean field with random peers against quenched nearest neighbours.

Part 1, from the existing measures.json of the two cells per model: paired
differences lattice - all-to-all within formulation at T = 1 for |m|,
chi_|m|, hysteresis gap, damage, chi-scan peak position, per-round fit.

Part 2, consensus and freezing against size, same policy at T = 1, both
flip sectors: lattice L in {4, 6, 8} and all-to-all N in {8, 16, 36, 64}
with exposure 4. Records the time to |m| = 1 (absorbing consensus), the
fraction of runs frozen in a domain state (no change over the last rounds
with |m| < 1, the stripe states of zero-temperature Glauber dynamics on
the square lattice, Spirin, Krapivsky, Redner 2001) and the fraction still
fluctuating. The scaling of the consensus time with N is the drift check of
§10.21: t_cons proportional to N^2 is noise-driven consensus, a time that
does not grow with N is interaction-driven order.

Output: results/phase_a/lattice_control.json and a text summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_dynamics import _clip  # noqa: E402
from src.common import build_manifest, write_json  # noqa: E402
from src.dynamics.memory import run_memory_dynamics  # noqa: E402
from src.llm.phase_a import policy_tables  # noqa: E402
from src.topology import adjacency_lists, complete_graph, lattice_2d  # noqa: E402

KEYS = ("abs_m_final", "chi_abs_rounds", "hyst_gap", "damage_final", "chi_peak_T", "J_round1", "J_round_last")


def _flat(f: dict, T: str) -> dict:
    b = f["by_T"][T]
    cs = f["chi_scan"]
    imax = int(np.argmax(cs["chi_abs"]))
    return {"abs_m_final": b["abs_m_final"], "chi_abs_rounds": b["chi_abs_rounds"],
            "hyst_gap": b["hysteresis"]["max_gap"], "damage_final": b["damage_mean"][-1],
            "chi_peak_T": cs["T"][imax], "chi_peak": cs["chi_abs"][imax],
            "J_round1": b["per_round_fit"][0]["J"], "J_round_last": b["per_round_fit"][-1]["J"],
            "zJ": b["zJ"]}


def compare_measures(results: Path, model: str, T: str, rng: np.random.Generator) -> dict:
    """Part 1: paired lattice - all differences over the ten formulations."""
    all_ = json.load(open(results / f"A_{model}_list_k0_e4_Tan_f0_all_N8" / "measures.json", encoding="utf-8"))
    lat = json.load(open(results / f"A_{model}_list_k0_e4_Tan_f0_lat6_N36" / "measures.json", encoding="utf-8"))
    fa = {(f["wording"], f["format"]): _flat(f, T) for f in all_["formulations"]}
    fl = {(f["wording"], f["format"]): _flat(f, T) for f in lat["formulations"]}
    keys = sorted(set(fa) & set(fl))
    out = {"n_formulations": len(keys), "per_formulation": [], "paired": {}}
    for k in keys:
        out["per_formulation"].append({"wording": k[0], "format": k[1], "zJ": fa[k]["zJ"],
                                       "all": fa[k], "lattice": fl[k]})
    for q in KEYS:
        diffs = np.array([fl[k][q] - fa[k][q] for k in keys if fl[k][q] is not None and fa[k][q] is not None])
        boots = [np.mean(rng.choice(diffs, size=diffs.size, replace=True)) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["paired"][q] = {"mean_all": float(np.mean([fa[k][q] for k in keys])),
                            "mean_lattice": float(np.mean([fl[k][q] for k in keys])),
                            "mean_diff": float(diffs.mean()), "ci95": [float(lo), float(hi)],
                            "by_formulation": diffs.tolist()}
    return out


def _classify(states: np.ndarray, tail: int) -> tuple[str, int | None]:
    """Consensus time, or frozen / fluctuating at the end of the run."""
    m = states.mean(axis=1)
    hit = np.where(np.abs(m) >= 1.0)[0]
    if hit.size:
        return "consensus", int(hit[0])
    if (states[-tail:] == states[-1]).all():
        return "frozen", None
    return "fluctuating", None


def size_scan(results: Path, model: str, T: float, lattice_L: list[int], all_N: list[int],
              n_seeds: int, max_rounds: int, rng: np.random.Generator) -> dict:
    """Part 2: consensus, freezing and consensus-time scaling at fixed policy."""
    gaps = json.load(open(results / f"A_{model}_list_k0_e4_Tan_f0_all_N8" / "gaps.json", encoding="utf-8"))
    tail = max(10, max_rounds // 10)
    out = {"lattice": {}, "all": {}, "policy_source": gaps["cell_id"]}
    systems = [("lattice", L, adjacency_lists(lattice_2d(L)), None) for L in lattice_L]
    systems += [("all", N, adjacency_lists(complete_graph(N)), 4) for N in all_N]
    for topo, size, adj, exposure in systems:
        N = len(adj)
        rows = []
        for tab in gaps["tables"]:
            pA, pB, _ = policy_tables(tab["entries"], T)
            counts = {"consensus": 0, "frozen": 0, "fluctuating": 0}
            times, m_abs = [], []
            for _ in range(n_seeds):
                for pol in (_clip(pA), _clip(pB)):
                    s0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
                    run = run_memory_dynamics(pol, adj, s0, 0, exposure, max_rounds, int(rng.integers(2**31)))
                    kind, t = _classify(run["states"], tail)
                    counts[kind] += 1
                    if t is not None:
                        times.append(t)
                    m_abs.append(float(abs(run["m"][-1])))
            n_runs = 2 * n_seeds
            rows.append({"wording": tab["wording"], "format": tab["format"],
                         "frac": {k: v / n_runs for k, v in counts.items()},
                         "t_cons_mean": float(np.mean(times)) if times else None,
                         "t_cons_median": float(np.median(times)) if times else None,
                         "abs_m_final_mean": float(np.mean(m_abs))})
        out[topo][str(size)] = {"N": N, "rows": rows,
                                "frac_consensus": float(np.mean([r["frac"]["consensus"] for r in rows])),
                                "frac_frozen": float(np.mean([r["frac"]["frozen"] for r in rows])),
                                "frac_fluctuating": float(np.mean([r["frac"]["fluctuating"] for r in rows])),
                                "t_cons_mean": float(np.mean([r["t_cons_mean"] for r in rows if r["t_cons_mean"] is not None]))
                                if any(r["t_cons_mean"] is not None for r in rows) else None,
                                "abs_m_final_mean": float(np.mean([r["abs_m_final_mean"] for r in rows]))}
        print(f"  {model} {topo:7s} N={N:3d}: consensus {out[topo][str(size)]['frac_consensus']:.2f} "
              f"frozen {out[topo][str(size)]['frac_frozen']:.2f} fluct {out[topo][str(size)]['frac_fluctuating']:.2f} "
              f"t_cons {out[topo][str(size)]['t_cons_mean']} |m| {out[topo][str(size)]['abs_m_final_mean']:.2f}", flush=True)
    # scaling exponent of the consensus time with N, per topology, where defined
    for topo in ("lattice", "all"):
        pts = [(v["N"], v["t_cons_mean"]) for v in out[topo].values() if isinstance(v, dict) and v.get("t_cons_mean")]
        if len(pts) >= 2:
            x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts])
            out[topo]["t_cons_exponent"] = float(np.polyfit(x, y, 1)[0])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="qwen0.6b,qwen1.7b")
    ap.add_argument("--T", default="1.0")
    ap.add_argument("--lattice-L", default="4,6,8")
    ap.add_argument("--all-N", default="8,16,36,64")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--max-rounds", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    results = Path(a.results)
    rng = np.random.default_rng(7)
    t0 = time.time()
    payload = {"T": float(a.T), "models": {}}
    for model in a.models.split(","):
        cmp_ = compare_measures(results, model, a.T, rng)
        print(f"== {model}: lattice L=6 (N=36) minus all-to-all N=8, exposure 4, T={a.T}, "
              f"{cmp_['n_formulations']} formulations ==")
        for q, v in cmp_["paired"].items():
            print(f"  {q:14s} all {v['mean_all']:+.2f}  lattice {v['mean_lattice']:+.2f}  "
                  f"diff {v['mean_diff']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
        print(f"== {model}: size scan, {a.seeds} seeds x 2 sectors, up to {a.max_rounds} rounds ==")
        scan = size_scan(results, model, float(a.T), [int(x) for x in a.lattice_L.split(",")],
                         [int(x) for x in a.all_N.split(",")], a.seeds, a.max_rounds, rng)
        for topo in ("lattice", "all"):
            if "t_cons_exponent" in scan[topo]:
                print(f"  {topo}: t_cons ~ N^{scan[topo]['t_cons_exponent']:.2f}")
        payload["models"][model] = {"compare": cmp_, "size_scan": scan}
        print(f"  ({time.time() - t0:.0f} s)", flush=True)
    out = a.out or str(results / "lattice_control.json")
    write_json(out, {"manifest": build_manifest(vars(a), [7]), **payload})
    print("written", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
