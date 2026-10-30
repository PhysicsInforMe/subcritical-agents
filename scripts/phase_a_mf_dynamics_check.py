#!/usr/bin/env python
"""Control P4b of the results notes (not included) (written before the data): does
the mean-field map built on the tabulated policy predict the tabulated
dynamics at large N?

For every list, k_mem = 0 table at the chosen exposure (default 4), both
models: the flip-symmetrised policy P(+ | k) drives an asynchronous
all-to-all dynamics with N agents, e random peers per update, from a
random initial state, for n_rounds rounds and n_seeds seeds. Reported
per formulation: final |m| (mean over seeds and the last third of the
rounds), F'(0) of the mean-field map, and the classification agreement:
    F'(0) < 1 predicts |m| < 0.3 (disordered), F'(0) > 1 predicts
    |m| > 0.7 (ordered); the rest is a disagreement to investigate.
The symmetrised policy is used so that the comparison is with the
unbiased criterion F'(0) versus 1. Output:
results/phase_a/mf_dynamics_check.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_meanfield_stability import mf_map, policy_from_entries  # noqa: E402
from src.common import build_manifest, write_json  # noqa: E402
from src.dynamics.memory import run_memory_dynamics  # noqa: E402
from src.topology import adjacency_lists, complete_graph  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="qwen0.6b,qwen1.7b")
    ap.add_argument("--exposure", type=int, default=4)
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--sector", default="sym", help="sym: flip-symmetrised table (the P4b control); A: the labelled sector "
                    "with its bias, whose mean-field map has one or two stable fixed points (revision R1-02)")
    ap.add_argument("--near-critical", type=float, default=None,
                    help="run only the formulations with |F'(0) - 1| below this value (revision R1-07, longer runs)")
    ap.add_argument("--out", default=None, help="output file (default results/phase_a/mf_dynamics_check.json)")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(21)
    adj = adjacency_lists(complete_graph(a.N))
    out = {"exposure": a.exposure, "N": a.N, "seeds": a.seeds, "rounds": a.rounds, "sector": a.sector, "models": {}}
    t0 = time.time()
    for model in a.models.split(","):
        p = Path(a.results) / f"A_{model}_list_k0_e{a.exposure}_Tan_f0_all_N{8 if a.exposure <= 7 else 32}" / "gaps.json"
        rows = []
        for tab in json.load(open(p, encoding="utf-8"))["tables"]:
            pA, ps = policy_from_entries(tab["entries"], 1.0)
            F, dF = mf_map(ps, a.exposure)
            d0 = float(dF(0.0))
            if a.near_critical is not None and abs(d0 - 1) > a.near_critical:
                continue
            table = pA if a.sector == "A" else ps
            pol = {(k, ()): min(max(v, 1e-12), 1 - 1e-12) for k, v in table.items()}
            finals, finals_signed = [], []
            for _ in range(a.seeds):
                s0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=a.N)
                run = run_memory_dynamics(pol, adj, s0, 0, a.exposure, a.rounds, int(rng.integers(2**31)))
                finals.append(float(np.abs(run["m"][-a.rounds // 3:]).mean()))
                finals_signed.append(float(np.mean(run["m"][-a.rounds // 3:])))
            m_abs = float(np.mean(finals))
            pred = "ordered" if d0 > 1 else "disordered"
            obs = "ordered" if m_abs > 0.7 else ("disordered" if m_abs < 0.3 else "intermediate")
            rows.append({"wording": tab["wording"], "format": tab["format"], "dF0_sym": d0, "abs_m_final": m_abs,
                         "abs_m_sd_seeds": float(np.std(finals, ddof=1)), "m_signed_final_mean": float(np.mean(finals_signed)),
                         "frac_seeds_positive": float(np.mean(np.array(finals_signed) > 0)),
                         "predicted": pred, "observed": obs, "agree": (pred == obs)})
            print(f"  {model} w{tab['wording']} {tab['format']:6s}: F'(0) {d0:5.2f} -> {pred:10s} | |m| at N={a.N}: {m_abs:.2f} ({obs}) {'OK' if pred == obs else 'DISAGREE'} ({time.time() - t0:.0f} s)", flush=True)
        n_ok = sum(r["agree"] for r in rows)
        print(f"== {model}: agreement {n_ok}/{len(rows)}; intermediate {sum(r['observed'] == 'intermediate' for r in rows)}")
        out["models"][model] = {"rows": rows, "agreement": f"{n_ok}/{len(rows)}"}
    write_json(Path(a.out) if a.out else Path(a.results) / "mf_dynamics_check.json", {"manifest": build_manifest(vars(a), [21]), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
