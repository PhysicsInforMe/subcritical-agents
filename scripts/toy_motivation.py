#!/usr/bin/env python
"""Toy test on the channel with motivations (design notes §5 Fase A, §7 toy rule).

One cell (exposure 4, no memory, all-to-all reading), both flip sectors by
symmetry, three wordings in the numbered-lines format, three conditions
on the same 16 peer configurations:
    list      states only                          (channel level 1)
    motiv     states + generated motivations        (channel level 2)
    shuffled  states + the same motivations attached to random labels
Each motivated configuration is drawn ``draws`` times from the pool, so the
label-space fit of (h, J) per condition has an ordering-and-draw bootstrap.
Decision rule: the channel is a knob if J(motiv) - J(list), paired by
wording, is outside the bootstrap band and larger than J(shuffled) - J(list).
Caveat written in advance (Cho et al., §10.23): in an abstract game the
motivations carry no evidence, and Cho found the Reason format to matter on
factual tasks and not on opinion tasks; a null here does not close the
channel question, which then moves to the task with truth (Phase B).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402
from src.llm import LogitReader  # noqa: E402
from src.llm.motivation import build_prompt_motivation, draw_motivations, motivation_pool  # noqa: E402
from src.llm.phase_a import label_space_fit, peer_configurations  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_motivation.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default=None)
    ap.add_argument("--exposures", default=None, help="comma list, overrides config exposure")
    ap.add_argument("--wordings", default=None, help="comma list of wording ids, overrides config")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    forms = yaml.safe_load(open(cfg["formulations"], encoding="utf-8"))
    if a.wordings:
        cfg["wording_ids"] = [int(x) for x in a.wordings.split(",")]
    exposures = [int(x) for x in a.exposures.split(",")] if a.exposures else [cfg["exposure"]]
    wordings = [forms["wordings"][i] for i in cfg["wording_ids"]]
    import torch
    torch.set_num_threads(cfg.get("threads", 8))
    reader = LogitReader(a.model, dtype=a.dtype)
    t0 = time.time()
    pool = motivation_pool(reader, cfg["pool_per_label"], seed=cfg["seed"], temperature=cfg["gen_temperature"])
    print(f"pool generated in {time.time() - t0:.0f} s; examples: A: {pool['A'][0]!r} | B: {pool['B'][0]!r}", flush=True)
    rng = np.random.default_rng(cfg["seed"])
    results = []
    for exposure in exposures:
      configs = peer_configurations(exposure, "list", cfg["max_orderings"], rng)
      for wi, wording in zip(cfg["wording_ids"], wordings):
        for cond in ("list", "motiv", "shuffled"):
              entries = []
              for k, orderings in configs.items():
                  for states in orderings:
                      draws = 1 if cond == "list" else cfg["draws"]
                      for dr in range(draws):
                          mot = None if cond == "list" else draw_motivations(states, "A", pool, rng, cond == "shuffled")
                          text = build_prompt_motivation(wording, states, "A", mot)
                          lA, lB, ntok = reader.label_logits(text)
                          entries.append({"memory": [], "k": int(k), "states": list(states), "gap": lA - lB,
                                          "logp_plus": lA, "logp_minus": lB, "draw": dr, "n_tokens": ntok,
                                          "motivations": mot})
              fit = label_space_fit(entries, cfg["temperature"], 0, n_boot=cfg["n_boot"], rng=rng)
              results.append({"wording": wi, "exposure": exposure, "condition": cond, "fit": fit, "n_passes": len(entries),
                              "n_tokens_mean": float(np.mean([e["n_tokens"] for e in entries])), "entries": entries})
              print(f"e{exposure} w{wi} {cond:8s}: h={fit['h']['estimate']:+.3f} [{fit['h']['ci95'][0]:+.2f},{fit['h']['ci95'][1]:+.2f}] "
                    f"J={fit['J']['estimate']:+.3f} [{fit['J']['ci95'][0]:+.2f},{fit['J']['ci95'][1]:+.2f}] "
                    f"tokens {results[-1]['n_tokens_mean']:.0f} ({time.time() - t0:.0f} s)", flush=True)
    # paired differences by wording
    summary = {}
    for exposure in exposures:
      for cond in ("motiv", "shuffled"):
        pick = lambda w, c: next(r for r in results if r["wording"] == w and r["condition"] == c and r["exposure"] == exposure)
        dJ = [pick(w, cond)["fit"]["J"]["estimate"] - pick(w, "list")["fit"]["J"]["estimate"] for w in cfg["wording_ids"]]
        dh = [pick(w, cond)["fit"]["h"]["estimate"] - pick(w, "list")["fit"]["h"]["estimate"] for w in cfg["wording_ids"]]
        summary[f"e{exposure}|{cond}"] = {"dJ_by_wording": dJ, "dJ_mean": float(np.mean(dJ)), "dh_by_wording": dh, "dh_mean": float(np.mean(dh))}
        print(f"e{exposure} {cond} - list: dJ per wording {np.round(dJ, 3).tolist()} mean {np.mean(dJ):+.3f}; dh mean {np.mean(dh):+.3f}")
    model_key = a.model.split("/")[-1].lower().replace("qwen3-", "qwen")
    out = a.out or f"results/toy/motivation_{model_key}.json"
    write_json(out, {"manifest": build_manifest(cfg, [cfg["seed"]], {"model": a.model, "dtype": reader.dtype,
                                                                     "revision": reader.resolved_revision}),
                     "pool": pool, "results": results, "summary": summary})
    print("written", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
