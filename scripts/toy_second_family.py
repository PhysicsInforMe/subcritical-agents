#!/usr/bin/env python
"""Toy for a second model family (the results notes (not included), test 3): does the
model follow the JSON prefix and answer with one of the two labels?

For one wording (the first of configs/phase_a_formulations.yaml, comma
format), the five peer configurations k = -4, -2, 0, 2, 4 (one ordering
each) and both flip sectors read directly (no symmetry shortcut), report
    * the total probability mass on the two labels after the prefix
      (must be well above 0.5, otherwise the model does not follow the
      protocol and the family is changed before the grid),
    * the label gaps and P(+) per k in both sectors,
    * the chat template used and the label token variants.
Usage: python scripts/toy_second_family.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --dtype bfloat16
Output: results/toy/second_family_<model>.json
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
from src.llm.phase_a import build_prompt_a  # noqa: E402
from src.llm.tabulate import LABELS, LogitReader  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    t0 = time.time()
    reader = LogitReader(a.model, dtype=a.dtype)
    forms = yaml.safe_load(open("configs/phase_a_formulations.yaml", encoding="utf-8"))
    wording = forms["wordings"][0]
    rows = []
    for plus in LABELS:
        for k in (-4, -2, 0, 2, 4):
            n_plus = (4 + k) // 2
            states = tuple([1] * n_plus + [-1] * (4 - n_plus))
            text = build_prompt_a(wording, "list", states, (), plus, "comma")
            lA, lB, ntok = reader.label_logits(text)
            # total mass on the two labels: exp of the log-sum-exp of both, which are log-probabilities
            mass = math.exp(lA) + math.exp(lB)
            gap = (lA - lB) if plus == LABELS[0] else (lB - lA)
            rows.append({"sector_plus": plus, "k": k, "gap_plus_minus_minus": gap, "p_plus": 1 / (1 + math.exp(-gap)),
                         "label_mass": mass, "n_tokens": ntok})
            print(f"  sector +={plus} k={k:+d}: gap {gap:+.2f}  P(+) {1 / (1 + math.exp(-gap)):.3f}  mass on labels {mass:.3f}", flush=True)
    masses = [r["label_mass"] for r in rows]
    verdict = "segue il protocollo" if min(masses) > 0.5 else "NON segue il protocollo: cambiare famiglia o prefisso"
    gapsA = {r["k"]: r["gap_plus_minus_minus"] for r in rows if r["sector_plus"] == LABELS[0]}
    slope = float(np.polyfit(list(gapsA), list(gapsA.values()), 1)[0] / 2)
    model_key = a.model.split("/")[-1].lower()
    out = {"manifest": build_manifest({"model": a.model, "dtype": a.dtype, "wording": wording}, [0],
                                      {"revision": reader.resolved_revision, "variant_ids": reader.variant_ids,
                                       "seconds": time.time() - t0}),
           "rows": rows, "min_label_mass": min(masses), "J_gap_sector_A_one_ordering": slope, "verdict": verdict}
    path = Path("results/toy") / f"second_family_{model_key}.json"
    write_json(path, out)
    print(f"{model_key}: min label mass {min(masses):.3f}; J (one ordering, sector A) {slope:+.3f}; {verdict} ({time.time() - t0:.0f} s) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
