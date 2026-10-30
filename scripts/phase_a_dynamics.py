#!/usr/bin/env python
"""Phase A, numpy part: measures of one cell from its tabulated policy
-> results/phase_a/<cell_id>/measures.json (design notes §4, §5, §14.2).

For every formulation and every temperature of the config (analytic
rescaling of the same logits, §14.1 B4):
    test 0   label-space fit (h, J, J_mem) with ordering bootstrap, and
             z_bar * J as the mean-field criterion
    dynamics all-to-all N = 8 (or lattice) with global flip: 30 seeds,
             20 rounds, states recorded per round
    per round De Nobili count regression on the simulated update logs (B14)
    test 2   chi_|m| on the temperature scan (long runs, both sectors)
    test 1   O-information of the 8 agents over seeds x late rounds
    test 3   damage spreading, one agent flipped, shared random stream
    test 4   stubborn sweep up and down in z, branch gap
    test 5   response entropy per agent per round (from the policy)
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
from src.dynamics.memory import run_memory_dynamics  # noqa: E402
from src.llm.phase_a import label_space_fit, policy_tables  # noqa: E402
from src.measures import chi_abs, fit_glauber_counts, o_information_binary, shannon_entropy  # noqa: E402
from src.topology import adjacency_lists, complete_graph, lattice_2d  # noqa: E402


def _adjacency(cfg):
    if cfg["topology"] == "all":
        return adjacency_lists(complete_graph(cfg["N"]))
    L = int(round(cfg["N"] ** 0.5))
    return adjacency_lists(lattice_2d(L))


def _clip(p):
    return {k: (min(max(v, 1e-12), 1 - 1e-12) if not isinstance(v, dict) else v) for k, v in p.items()}


def _per_round_fit(logs: list[np.ndarray], n_rounds: int) -> list[dict]:
    """De Nobili count regression per round on the pooled update logs."""
    allrows = np.concatenate(logs)
    out = []
    for r in range(1, n_rounds + 1):
        rows = allrows[allrows[:, 0] == r]
        try:
            f = fit_glauber_counts(rows[:, 2], rows[:, 4], None, min_count=5)
            out.append({"round": r, "h": f["h"]["estimate"], "J": f["J"]["estimate"], "n": int(rows.shape[0])})
        except (ValueError, np.linalg.LinAlgError):
            out.append({"round": r, "h": None, "J": None, "n": int(rows.shape[0])})
    return out


def analyse_formulation(tab: dict, cfg: dict, adj, rng: np.random.Generator) -> dict:
    entries = tab["entries"]
    N, k_mem, exposure = cfg["N"], cfg["k_mem"], cfg["exposure"]
    z_bar = min(exposure, len(adj[0]))
    res = {"wording": tab["wording"], "format": tab["format"], "n_passes": tab["n_passes"],
           "prompt_nll": float(np.mean([e["nll"] for e in entries])) if "nll" in entries[0] else None,
           "by_T": {}}
    for T in cfg["temperatures"]:
        fit = label_space_fit(entries, T, k_mem, n_boot=cfg["n_boot"], rng=rng)
        pA, pB, p_sym = policy_tables(entries, T)
        var = p_sym.pop("_var")
        polA, polB = _clip(pA), _clip(pB)
        # dynamics in both sectors, seeds
        runs = []
        for s in cfg["seeds"]:
            for pol in (polA, polB):
                s0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
                runs.append(run_memory_dynamics(pol, adj, s0, k_mem, exposure if cfg["topology"] == "all" else None,
                                                cfg["n_rounds"], seed=int(rng.integers(2**31))))
        m = np.array([r["m"] for r in runs])                      # (runs, rounds+1)
        late = np.concatenate([r["states"][cfg["n_rounds"] // 2:] for r in runs])
        omega = o_information_binary(late) if N <= 12 else None
        per_round = _per_round_fit([r["log"] for r in runs], cfg["n_rounds"])
        # damage spreading: same seed, one agent flipped, sector A
        dmg = []
        for s in cfg["seeds"][:10]:
            s0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
            seed = int(rng.integers(2**31))
            a = run_memory_dynamics(polA, adj, s0, k_mem, exposure, cfg["n_rounds"], seed)
            s1 = s0.copy(); s1[0] *= -1
            b = run_memory_dynamics(polA, adj, s1, k_mem, exposure, cfg["n_rounds"], seed)
            dmg.append((a["states"] != b["states"]).mean(axis=1))
        dmg = np.array(dmg)
        # stubborn sweep (test 4), sector A, state carried across z
        z_vals = cfg["stubborn_z"]
        s_cur = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
        branches = {}
        for name, zs in (("up", z_vals), ("down", z_vals[::-1])):
            vals = []
            for z in zs:
                stub = np.zeros(N, dtype=np.int8)
                n_stub = int(round(abs(z) * N))
                if n_stub:
                    # random placement, a fresh permutation at every z, as in the
                    # committed-minority literature (Xie et al. 2011); on the
                    # complete graph placement is irrelevant, on the lattice a
                    # contiguous block pins whole rows and hides the effect
                    # (scripts/check_stubborn_placement.py, 2026-09-11)
                    stub[rng.permutation(N)[:n_stub]] = 1 if z > 0 else -1
                out = run_memory_dynamics(polA, adj, s_cur, k_mem, exposure, cfg["n_rounds"], int(rng.integers(2**31)),
                                          stubborn=stub)
                s_cur = out["final"]
                free = stub == 0
                vals.append(float(out["states"][cfg["n_rounds"] // 2:, free].mean()) if free.any() else 0.0)
            branches[name] = vals if name == "up" else vals[::-1]
        gap = np.abs(np.array(branches["up"]) - np.array(branches["down"]))
        # response entropy (test 5): mean entropy of the policy row actually
        # visited by the updates, per round (Brockers, §10.12)
        ent_by_round = []
        for r in range(1, cfg["n_rounds"] + 1):
            hs = []
            for run in runs:
                rows = run["log"][run["log"][:, 0] == r]
                for _, _, k, msum, _ in rows:
                    p = np.mean([v for (kk, mm), v in polA.items() if kk == k and sum(mm) == msum]) if k_mem else polA[(int(k), ())]
                    hs.append(shannon_entropy(np.array([p, 1 - p])))
            ent_by_round.append(float(np.mean(hs)) if hs else None)
        res["by_T"][str(T)] = {
            "fit": fit, "zJ": z_bar * fit["J"]["estimate"], "z_bar": z_bar,
            "order_variance_max": float(max(var.values())),
            "abs_m_final": float(np.abs(m[:, -1]).mean()),
            "chi_abs_rounds": float(chi_abs(m[:, cfg["n_rounds"] // 2:].ravel(), N, T)),
            "o_information": omega, "per_round_fit": per_round,
            "damage_mean": dmg.mean(axis=0).tolist(),
            "hysteresis": {"z": list(z_vals), "up": branches["up"], "down": branches["down"],
                           "max_gap": float(gap.max())},
            "response_entropy_by_round": ent_by_round,
        }
    return res


def chi_scan(tab: dict, cfg: dict, adj, rng: np.random.Generator) -> dict:
    """chi_|m| over a log-spaced temperature scan, fewer seeds, both sectors."""
    sc = cfg["chi_scan"]
    Ts = np.geomspace(sc["T_min"], sc["T_max"], sc["n_T"])
    N, k_mem, exposure = cfg["N"], cfg["k_mem"], cfg["exposure"]
    chi, err = [], []
    for T in Ts:
        pA, pB, _ = policy_tables(tab["entries"], T)
        per_seed = []
        for s in cfg["seeds"][:6]:
            ms = []
            for pol in (_clip(pA), _clip(pB)):
                s0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
                out = run_memory_dynamics(pol, adj, s0, k_mem, exposure if cfg["topology"] == "all" else None,
                                          3 * cfg["n_rounds"], int(rng.integers(2**31)))
                ms.append(out["m"][cfg["n_rounds"]:])
            per_seed.append(chi_abs(np.concatenate(ms), N, T))
        chi.append(float(np.mean(per_seed)))
        err.append(float(np.std(per_seed, ddof=1) / np.sqrt(len(per_seed))))
    return {"T": Ts.tolist(), "chi_abs": chi, "chi_abs_err": err}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--all", default=None)
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    paths = [Path(p) for p in a.configs]
    if a.all:
        paths += sorted(Path(a.all).glob("*.yaml"))
    for p in paths:
        cfg = yaml.safe_load(open(p, encoding="utf-8"))
        cell_dir = Path(a.results) / cfg["cell_id"]
        gaps_path, out_path = cell_dir / "gaps.json", cell_dir / "measures.json"
        if not gaps_path.exists():
            print(f"{cfg['cell_id']}: no gaps.json yet, skipped")
            continue
        if out_path.exists() and not a.force:
            print(f"{cfg['cell_id']}: measures exist, skipped")
            continue
        t0 = time.time()
        gaps = json.load(open(gaps_path, encoding="utf-8"))
        adj = _adjacency(cfg)
        rng = np.random.default_rng(cfg["seeds"][0])
        formulations = []
        for tab in gaps["tables"]:
            r = analyse_formulation(tab, cfg, adj, rng)
            r["chi_scan"] = chi_scan(tab, cfg, adj, rng)
            formulations.append(r)
            f1 = r["by_T"]["1.0"]["fit"]
            print(f"  {cfg['cell_id']} w{tab['wording']} {tab['format']}: h={f1['h']['estimate']:+.3f} "
                  f"J={f1['J']['estimate']:+.3f} zJ={r['by_T']['1.0']['zJ']:.2f} "
                  f"Omega={r['by_T']['1.0']['o_information']} ({time.time() - t0:.0f} s)", flush=True)
        write_json(out_path, {"manifest": build_manifest(cfg, cfg["seeds"], {"gaps_commit": gaps["manifest"]["commit"]}),
                              "cell_id": cfg["cell_id"], "formulations": formulations})
    return 0


if __name__ == "__main__":
    sys.exit(main())
