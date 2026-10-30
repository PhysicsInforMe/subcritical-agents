#!/usr/bin/env python
"""Memory length curve for S2 (stress test declared on 2026-09-12 before
the data, the project log (not included)): does the peer coupling decrease monotonically with
the number k of own past states shown (H_k), or does it drop as soon as
the memory block is present, independently of its length (H_block)?

For the list channel at exposure 4, all formulations, k in {0, 1, 2, 3}:
    * J_gap (and J_mem_gap where k > 0) from the mean gap per (k_peers,
      memory sum), the estimator of §21;
    * paired differences within formulation k1-k0, k2-k1, k3-k2, k3-k1,
      bootstrap over formulations;
    * decision rule (from the declaration): H_k if k1-k0 < 0 and k3-k1 < 0
      with intervals excluding zero; H_block if k1-k0 < 0 and the k3-k1
      interval contains zero; if k1-k0 contains zero and k3-k1 < 0 the
      effect grows only with length.
Output: results/phase_a/memory_curve.json and a text summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402

KS = (0, 1, 2, 3)


def _boot(d: np.ndarray, rng: np.random.Generator, n: int = 4000) -> dict:
    b = [np.mean(rng.choice(d, size=d.size, replace=True)) for _ in range(n)]
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "n": int(d.size), "by_formulation": d.tolist()}


def gap_fit(entries: list[dict], has_mem: bool) -> tuple[float, float | None]:
    groups: dict[tuple, list[float]] = {}
    for e in entries:
        groups.setdefault((e["k"], int(sum(e["memory"]))), []).append(e["gap"])
    keys = sorted(groups)
    X = np.array([[1.0, k, m] if has_mem else [1.0, k] for (k, m) in keys], dtype=float)
    y = np.array([np.mean(groups[key]) for key in keys])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(c[1] / 2), (float(c[2] / 2) if has_mem else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="qwen0.6b,qwen1.7b")
    ap.add_argument("--exposure", type=int, default=4)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(11)
    out = {"exposure": a.exposure, "models": {}}
    for model in a.models.split(","):
        J: dict[int, dict[tuple, float]] = {}
        Jm: dict[int, dict[tuple, float]] = {}
        for k in KS:
            p = Path(a.results) / f"A_{model}_list_k{k}_e{a.exposure}_Tan_f0_all_N8" / "gaps.json"
            if not p.exists():
                print(f"  {model} k={k}: missing {p}")
                continue
            J[k], Jm[k] = {}, {}
            for tab in json.load(open(p, encoding="utf-8"))["tables"]:
                j, jm = gap_fit(tab["entries"], has_mem=k > 0)
                J[k][(tab["wording"], tab["format"])] = j
                if jm is not None:
                    Jm[k][(tab["wording"], tab["format"])] = jm
        res = {"J_gap_mean": {str(k): float(np.mean(list(J[k].values()))) for k in J},
               "J_mem_gap_mean": {str(k): float(np.mean(list(Jm[k].values()))) for k in Jm if Jm[k]},
               "paired": {}}
        for (ka, kb) in ((1, 0), (2, 1), (3, 2), (3, 1), (3, 0)):
            if ka in J and kb in J:
                keys = sorted(set(J[ka]) & set(J[kb]))
                res["paired"][f"k{ka}-k{kb}"] = _boot(np.array([J[ka][q] - J[kb][q] for q in keys]), rng)
        # decision rule of the declaration
        d10, d31 = res["paired"].get("k1-k0"), res["paired"].get("k3-k1")
        verdict = "indeterminato (celle mancanti)"
        if d10 and d31:
            neg10 = d10["ci95"][1] < 0; zero31 = d31["ci95"][0] <= 0 <= d31["ci95"][1]; neg31 = d31["ci95"][1] < 0
            zero10 = d10["ci95"][0] <= 0 <= d10["ci95"][1]
            if neg10 and neg31:
                verdict = "H_k: decrescita monotona con la lunghezza della memoria"
            elif neg10 and zero31:
                verdict = "H_block: l'effetto compare con la presenza del blocco di memoria, non cresce con la lunghezza"
            elif zero10 and neg31:
                verdict = "effetto che cresce solo con la lunghezza"
            else:
                verdict = "nessuna delle tre previsioni: da riportare come indeterminato"
        res["verdict"] = verdict
        # J_mem per memory state (coefficient of the memory sum): mean over
        # formulations with its interval, and paired differences between
        # consecutive k (claims.md B12, computation declared on 2026-09-21).
        # Separate generator so that the J_gap intervals above are unchanged.
        rng_m = np.random.default_rng(12)
        res["J_mem_gap"] = {str(k): _boot(np.array(list(Jm[k].values())), rng_m) for k in Jm if Jm[k]}
        res["J_mem_paired"] = {}
        for (ka, kb) in ((2, 1), (3, 2), (3, 1)):
            if ka in Jm and kb in Jm and Jm[ka] and Jm[kb]:
                keys = sorted(set(Jm[ka]) & set(Jm[kb]))
                res["J_mem_paired"][f"k{ka}-k{kb}"] = _boot(np.array([Jm[ka][q] - Jm[kb][q] for q in keys]), rng_m)
        for name, v in res["J_mem_paired"].items():
            print(f"  J_mem paired {name}: {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] n={v['n']}")
        out["models"][model] = res
        print(f"== {model}, list, e = {a.exposure}")
        print("  J_gap mean by k:", {k: round(v, 3) for k, v in res["J_gap_mean"].items()},
              "| J_mem_gap mean by k:", {k: round(v, 3) for k, v in res["J_mem_gap_mean"].items()})
        for name, v in res["paired"].items():
            print(f"  paired {name}: {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] n={v['n']}")
        print("  verdict:", verdict)
    write_json(Path(a.results) / "memory_curve.json", {"manifest": build_manifest(vars(a), [11]), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
