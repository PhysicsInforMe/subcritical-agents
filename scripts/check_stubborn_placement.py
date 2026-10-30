#!/usr/bin/env python
"""Hysteresis with stubborn agents: contiguous versus random placement
(the design notes test 4, De Marzo et al. §10.11; committed minorities on
networks, Xie et al. 2011).

scripts/phase_a_dynamics.py pins the first n_stub sites, which on the
complete graph is irrelevant but on the L x L lattice pins whole rows: a
stripe whose interface is the only place where free agents feel the
pinned ones. The committed-minority literature places the committed
agents uniformly at random, and De Marzo's protocol is on the complete
graph, where the two placements coincide. This check reruns the sweep on
the lattice cell (same policy table as the all-to-all cell, exposure 4)
with random placement, a fresh uniform permutation at every z, and
reports the branch gap next to the contiguous one, per formulation and
both models. Output: results/phase_a/stubborn_placement.json.
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

Z = [-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5]


def sweep(pol: dict, adj, N: int, n_rounds: int, rng: np.random.Generator, placement: str, n_rep: int) -> float:
    """Mean over repetitions of the max |up - down| branch gap on free agents."""
    gaps = []
    for _ in range(n_rep):
        s_cur = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
        branches = {}
        for name, zs in (("up", Z), ("down", Z[::-1])):
            vals = []
            for z in zs:
                stub = np.zeros(N, dtype=np.int8)
                n_stub = int(round(abs(z) * N))
                if n_stub:
                    idx = np.arange(n_stub) if placement == "contiguous" else rng.permutation(N)[:n_stub]
                    stub[idx] = 1 if z > 0 else -1
                out = run_memory_dynamics(pol, adj, s_cur, 0, None if len(adj[0]) <= 4 else 4, n_rounds,
                                          int(rng.integers(2**31)), stubborn=stub)
                s_cur = out["final"]
                free = stub == 0
                vals.append(float(out["states"][n_rounds // 2:, free].mean()) if free.any() else 0.0)
            branches[name] = vals if name == "up" else vals[::-1]
        gaps.append(float(np.abs(np.array(branches["up"]) - np.array(branches["down"])).max()))
    return float(np.mean(gaps))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="qwen0.6b,qwen1.7b")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(3)
    t0 = time.time()
    out = {"T": a.T, "z": Z, "models": {}}
    systems = {"lattice6": adjacency_lists(lattice_2d(6)), "all8": adjacency_lists(complete_graph(8))}
    for model in a.models.split(","):
        gaps = json.load(open(Path(a.results) / f"A_{model}_list_k0_e4_Tan_f0_all_N8" / "gaps.json", encoding="utf-8"))
        rows = []
        for tab in gaps["tables"]:
            pA, _, _ = policy_tables(tab["entries"], a.T)
            pol = _clip(pA)
            row = {"wording": tab["wording"], "format": tab["format"]}
            for sys_name, adj in systems.items():
                for placement in ("contiguous", "random"):
                    row[f"{sys_name}|{placement}"] = sweep(pol, adj, len(adj), a.rounds, rng, placement, a.reps)
            rows.append(row)
            print(f"  {model} w{row['wording']} {row['format']:6s} " +
                  " ".join(f"{k}={row[k]:.2f}" for k in row if "|" in k) + f" ({time.time() - t0:.0f} s)", flush=True)
        keys = [k for k in rows[0] if "|" in k]
        means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        diff = np.array([r["lattice6|random"] - r["lattice6|contiguous"] for r in rows])
        boots = [np.mean(rng.choice(diff, size=diff.size, replace=True)) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["models"][model] = {"rows": rows, "means": means,
                                "lattice_random_minus_contiguous": {"mean": float(diff.mean()), "ci95": [float(lo), float(hi)],
                                                                    "by_formulation": diff.tolist()}}
        print(f"== {model}: means " + " ".join(f"{k}={v:.2f}" for k, v in means.items()))
        print(f"   lattice random - contiguous: {diff.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]")
    path = Path(a.results) / "stubborn_placement.json"
    write_json(path, {"manifest": build_manifest(vars(a), [3]), **out})
    print("written", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
