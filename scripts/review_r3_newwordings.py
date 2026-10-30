#!/usr/bin/env python
"""Road 2 of the answer to R3-01 / R3-05 (declared 2026-09-23, the project log (not included)):
five new wordings (configs/phase_a_formulations_new.yaml, cells *_allnw_N8)
next to the five original ones. For each Qwen model:

  H1  J_gap(4) - J_gap(2) and J_gap(7) - J_gap(2), list channel, no memory;
  H3  J_gap(k = 3) - J_gap(k = 0) at 4 peers, list channel;

each as the mean over the (wording, format) pairs, with the t interval on the
per-wording means over the TEN wordings (9 df, the decision rule), over the
five new wordings alone (4 df, out-of-family replication, no decision) and
over the five original ones (4 df, as in review_r3.json). A finding is
restored for a model when the ten-wording interval excludes zero; "in both
models" only if it holds in both. Also reported: J_gap and h_gap per new
formulation at 4 peers, and the coupling ratio new / original.
Output: results/phase_a/review_r3_newwordings.json.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_memory_curve import gap_fit  # noqa: E402
from src.common import build_manifest, write_json  # noqa: E402

R = Path("results/phase_a")


def tables(model, k, e, suffix):
    hits = glob.glob(str(R / f"A_{model}_list_k{k}_e{e}_Tan_f0_{suffix}_N*/gaps.json"))
    return json.load(open(hits[0], encoding="utf-8"))["tables"] if hits else None


def t_int(per_wording: dict) -> dict:
    m = np.array(list(per_wording.values()), float); n = m.size
    if n < 2:
        return {"mean": float(m.mean()), "ci95": None, "n_wordings": int(n)}
    tq = float(stats.t.ppf(0.975, n - 1)); se = m.std(ddof=1) / np.sqrt(n)
    ci = [float(m.mean() - tq * se), float(m.mean() + tq * se)]
    return {"mean": float(m.mean()), "ci95": ci, "n_wordings": int(n), "n_negative": int((m < 0).sum()), "excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            "per_wording": {str(k): float(v) for k, v in per_wording.items()}}


def main() -> int:
    out: dict = {"models": {}}
    for m in ("qwen0.6b", "qwen1.7b"):
        J = {}   # (group, wording, format, k, e) -> J
        H = {}
        for group, suffix in (("orig", "all"), ("new", "allnw")):
            for k, e in ((0, 2), (0, 4), (0, 7), (3, 4)):
                tabs = tables(m, k, e, suffix)
                if tabs is None:
                    print(f"missing {m} {group} k{k} e{e}"); continue
                for tb in tabs:
                    Jv, _ = gap_fit(tb["entries"], k > 0)
                    J[(group, tb["wording"], tb["format"], k, e)] = Jv
                    if k == 0:
                        acc = {}
                        for x in tb["entries"]:
                            acc.setdefault(x["k"], []).append(x["gap"])
                        ks = np.array(sorted(acc), float); g = np.array([np.mean(acc[int(q)]) for q in ks])
                        H[(group, tb["wording"], tb["format"], e)] = float(np.polyfit(ks, g, 1)[1] / 2)
        res = {}
        for name, (k1, e1, k0, e0) in {"J.e4-e2": (0, 4, 0, 2), "J.e7-e2": (0, 7, 0, 2), "J.k3-k0": (3, 4, 0, 4)}.items():
            per = {}
            for (group, w, f, k, e), v in J.items():
                if k == k1 and e == e1 and (group, w, f, k0, e0) in J:
                    per.setdefault((group, w), []).append(v - J[(group, w, f, k0, e0)])
            per = {key: float(np.mean(v)) for key, v in per.items()}
            ten = t_int({f"{g}{w}": v for (g, w), v in per.items()})
            new = t_int({f"{g}{w}": v for (g, w), v in per.items() if g == "new"})
            orig = t_int({f"{g}{w}": v for (g, w), v in per.items() if g == "orig"})
            res[name] = {"ten_wordings": ten, "new_five": new, "original_five": orig}
            fmt = lambda r: f"{r['mean']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] neg {r.get('n_negative')}/{r['n_wordings']} {'EXCL' if r.get('excludes_zero') else 'incl'}" if r.get("ci95") else "n/a"
            print(f"{m} {name}: ten {fmt(ten)} | new {fmt(new)} | orig {fmt(orig)}")
        # per new formulation at 4 peers
        rows = [{"wording": w, "format": f, "J_gap": v, "h_gap": H.get(("new", w, f, 4))} for (g, w, f, k, e), v in sorted(J.items()) if g == "new" and k == 0 and e == 4]
        Jn = np.mean([r["J_gap"] for r in rows]); Jo = np.mean([v for (g, w, f, k, e), v in J.items() if g == "orig" and k == 0 and e == 4])
        res["formulations_new_e4"] = rows; res["J_mean_e4"] = {"new": float(Jn), "orig": float(Jo)}
        print(f"{m} J at 4 peers: new {Jn:.2f} (range {min(r['J_gap'] for r in rows):.2f} to {max(r['J_gap'] for r in rows):.2f}), original {Jo:.2f}; h new " + ", ".join(f"{r['h_gap']:+.2f}" for r in rows))
        out["models"][m] = res
    write_json(R / "review_r3_newwordings.json", {"manifest": build_manifest({"script": "review_r3_newwordings"}, []), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
