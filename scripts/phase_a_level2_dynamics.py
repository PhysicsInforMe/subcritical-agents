#!/usr/bin/env python
"""Phase A, channel level 2: the dynamics battery on the tabulated policies
with motivations (design notes §17, "misure di dinamica non ancora calcolate
sulle tabelle con motivazioni").

The level-2 grid (scripts/toy_motivation.py) stores, per (exposure, wording,
condition), the label-space gaps of every peer configuration and draw. Those
entries are a tabulated policy in the same format as a Phase A cell, so the
whole numpy battery of scripts/phase_a_dynamics.py applies unchanged:
test 0 (z_bar J), dynamics with global flip, per-round De Nobili fit,
chi_|m| scan, O-information, damage spreading, stubborn hysteresis,
response entropy. The Phase A cell config of the same exposure (N = 8,
all-to-all, k_mem = 0) supplies seeds, rounds, temperatures and the
stubborn sweep, so the three conditions are compared under the same
protocol as the poor-channel grid.

Output: results/phase_a_level2/measures_<model>.json with one block per
(exposure, wording, condition) and a summary of paired differences
(motiv - list, shuffled - list) by wording at T = 1.
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
from scripts.phase_a_dynamics import _adjacency, analyse_formulation, chi_scan  # noqa: E402
from src.common import build_manifest, write_json  # noqa: E402

SUMMARY_KEYS = ("zJ", "abs_m_final", "chi_abs_rounds", "o_information", "hyst_gap", "damage_final", "chi_peak_T")


def _cell_config(model_key: str, exposure: int, configs_dir: Path) -> dict:
    path = configs_dir / f"A_{model_key}_list_k0_e{exposure}_Tan_f0_all_N8.yaml"
    return yaml.safe_load(open(path, encoding="utf-8"))


def _flat(block: dict, chi: dict) -> dict:
    """Scalars at one temperature used in the summary tables."""
    imax = int(np.argmax(chi["chi_abs"]))
    return {"zJ": block["zJ"], "J": block["fit"]["J"]["estimate"], "h": block["fit"]["h"]["estimate"],
            "abs_m_final": block["abs_m_final"], "chi_abs_rounds": block["chi_abs_rounds"],
            "o_information": block["o_information"], "hyst_gap": block["hysteresis"]["max_gap"],
            "damage_final": block["damage_mean"][-1], "chi_peak_T": chi["T"][imax], "chi_peak": chi["chi_abs"][imax],
            "J_round1": block["per_round_fit"][0]["J"], "J_round_last": block["per_round_fit"][-1]["J"]}


def summarise(blocks: list[dict]) -> dict:
    """Mean over wordings per (exposure, condition) and paired differences
    against the 'list' condition within wording, bootstrap over wordings."""
    out: dict = {"means": {}, "paired": {}}
    rng = np.random.default_rng(0)
    groups: dict[tuple, list[dict]] = {}
    for b in blocks:
        groups.setdefault((b["exposure"], b["condition"]), []).append(b)
    for (e, cond), bl in sorted(groups.items()):
        out["means"][f"e{e}|{cond}"] = {
            k: (float(np.mean([b["flat"][k] for b in bl if b["flat"][k] is not None]))
                if any(b["flat"][k] is not None for b in bl) else None)
            for k in SUMMARY_KEYS}
    exposures = sorted({b["exposure"] for b in blocks})
    for e in exposures:
        base = {b["wording"]: b for b in blocks if b["exposure"] == e and b["condition"] == "list"}
        for cond in ("motiv", "shuffled"):
            here = {b["wording"]: b for b in blocks if b["exposure"] == e and b["condition"] == cond}
            words = sorted(set(base) & set(here))
            if not words:
                continue
            entry = {}
            for k in SUMMARY_KEYS:
                diffs = np.array([here[w]["flat"][k] - base[w]["flat"][k] for w in words
                                  if here[w]["flat"][k] is not None and base[w]["flat"][k] is not None])
                if diffs.size == 0:
                    continue
                boots = [np.mean(rng.choice(diffs, size=diffs.size, replace=True)) for _ in range(2000)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                entry[k] = {"by_wording": diffs.tolist(), "mean": float(diffs.mean()),
                            "ci95": [float(lo), float(hi)]}
            out["paired"][f"e{e}|{cond}-list"] = entry
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="level-2 json files of one model")
    ap.add_argument("--model", required=True, help="model key, e.g. qwen1.7b")
    ap.add_argument("--configs", default="configs/phase_a")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary-T", default="1.0")
    a = ap.parse_args(argv)
    configs_dir = Path(a.configs)
    blocks, manifests = [], []
    t0 = time.time()
    for path in a.inputs:
        d = json.load(open(path, encoding="utf-8"))
        manifests.append({"input": path, "commit": d["manifest"]["commit"], "model": d["manifest"]["model"],
                          "revision": d["manifest"].get("revision"), "dtype": d["manifest"].get("dtype")})
        for r in d["results"]:
            cfg = _cell_config(a.model, r["exposure"], configs_dir)
            adj = _adjacency(cfg)
            rng = np.random.default_rng(cfg["seeds"][0])
            tab = {"entries": r["entries"], "wording": r["wording"], "format": "lines", "n_passes": r["n_passes"]}
            res = analyse_formulation(tab, cfg, adj, rng)
            res["chi_scan"] = chi_scan(tab, cfg, adj, rng)
            res.update({"exposure": r["exposure"], "condition": r["condition"], "cell_config": cfg["cell_id"]})
            res["flat"] = _flat(res["by_T"][a.summary_T], res["chi_scan"])
            blocks.append(res)
            f = res["flat"]
            print(f"e{r['exposure']} w{r['wording']} {r['condition']:8s}: zJ={f['zJ']:.2f} |m|={f['abs_m_final']:.2f} "
                  f"Omega={f['o_information']:+.2f} hyst={f['hyst_gap']:.2f} dmg={f['damage_final']:.2f} "
                  f"chi_peak T={f['chi_peak_T']:.1f} ({time.time() - t0:.0f} s)", flush=True)
    summary = summarise(blocks)
    for key, v in summary["means"].items():
        print(f"{key:14s} " + " ".join(f"{k}={v[k]:.2f}" if v[k] is not None else f"{k}=NA" for k in SUMMARY_KEYS))
    for key, v in summary["paired"].items():
        for k in ("o_information", "hyst_gap", "zJ"):
            if k in v:
                print(f"paired {key} {k}: {np.round(v[k]['by_wording'], 2).tolist()} mean {v[k]['mean']:+.2f} "
                      f"[{v[k]['ci95'][0]:+.2f}, {v[k]['ci95'][1]:+.2f}]")
    out = a.out or f"results/phase_a_level2/measures_{a.model}.json"
    write_json(out, {"manifest": build_manifest({"inputs": a.inputs, "model": a.model, "summary_T": a.summary_T}, [],
                                                {"sources": manifests}),
                     "blocks": blocks, "summary": summary})
    print("written", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
