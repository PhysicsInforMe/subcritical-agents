#!/usr/bin/env python
"""Sampled control of the first-token policy against free generation (revision R2-01).

For two formulations (wording 0 and wording 4, comma format), 4 peers, every
one of the 16 peer configurations, no memory: the model generates freely
after the chat template (thinking disabled, no system prompt, NO imposed
JSON prefix), n samples per configuration at decoding scale T with pure
sampling (top_p = 1, top_k = 0). The answer is parsed with the regex
'"state"\\s*:\\s*"?\\s*([AaBb])' on the generated text; anything else is
counted as unparseable. The frequency of A among parsed answers is compared
with the tabulated first-token policy rescaled to the same T,
P = sigma(gap / T), read from the existing gaps.json of the cell.

Reported per model: unparseable rate, mean and maximum absolute difference
between frequency and table over the 32 (formulation, configuration) pairs,
Pearson correlation, and the number of pairs where frequency and table sit on
the same side of 1/2. Output: results/phase_a/generation_check.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402

from src.common import build_manifest, write_json  # noqa: E402
from src.llm.phase_a import build_prompt_a  # noqa: E402
from src.llm.tabulate import LABELS, LogitReader  # noqa: E402

MODELS = {"qwen0.6b": ("Qwen/Qwen3-0.6B", "float32"), "qwen1.7b": ("Qwen/Qwen3-1.7B", "bfloat16")}
PAT = re.compile(r'"state"\s*:\s*"?\s*([AaBb])')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wordings", default="0,4")
    ap.add_argument("--exposure", type=int, default=4)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--T", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/phase_a/generation_check.json")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    name, dtype = MODELS[a.model]
    reader = LogitReader(name, dtype=dtype)
    forms = yaml.safe_load(open("configs/phase_a_formulations.yaml", encoding="utf-8"))
    cell = Path("results/phase_a") / f"A_{a.model}_list_k0_e{a.exposure}_Tan_f0_all_N8" / "gaps.json"
    tabs = {(t["wording"], t["format"]): t for t in json.load(open(cell, encoding="utf-8"))["tables"]}
    out_path = Path(a.out)
    prev = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {"models": {}}
    rows = []
    t0 = time.time()
    for wi in [int(x) for x in a.wordings.split(",")]:
        tab = tabs[(wi, "comma")]
        for x in tab["entries"]:
            states = tuple(x["states"])
            text = build_prompt_a(forms["wordings"][wi], "list", states, (), LABELS[0], "comma")
            messages = [{"role": "user", "content": text}]
            try:
                chat = reader.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                chat = reader.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = reader.tok(chat, return_tensors="pt", add_special_tokens=False)
            torch.manual_seed(100 * wi + sum(1 for s in states if s == 1) * 7 + hash(states) % 97)
            with torch.no_grad():
                gen = reader.model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=a.T, top_p=1.0, top_k=0,
                                            num_return_sequences=a.n, pad_token_id=reader.tok.eos_token_id)
            texts = [reader.tok.decode(g[ids["input_ids"].shape[1]:], skip_special_tokens=True) for g in gen]
            parsed = [PAT.search(tx) for tx in texts]
            labels = [m.group(1).upper() for m in parsed if m]
            n_ok = len(labels); freq = (sum(l == "A" for l in labels) / n_ok) if n_ok else None
            p_tab = float(1 / (1 + np.exp(-x["gap"] / a.T)))
            rows.append({"wording": wi, "states": list(states), "k": x["k"], "n": a.n, "n_parsed": n_ok, "freq_A": freq, "p_table_T": p_tab,
                         "gap": x["gap"], "examples_unparsed": [tx for tx, m in zip(texts, parsed) if not m][:3]})
            print(f"  w{wi} k={x['k']:+d} {states}: parsed {n_ok}/{a.n}, freq A {freq if freq is None else round(freq, 2)} vs table {p_tab:.2f} ({time.time() - t0:.0f} s)", flush=True)
    ok = [r for r in rows if r["freq_A"] is not None]
    f = np.array([r["freq_A"] for r in ok]); p = np.array([r["p_table_T"] for r in ok]); d = np.abs(f - p)
    summ = {"unparseable_rate": float(1 - sum(r["n_parsed"] for r in rows) / sum(r["n"] for r in rows)),
            "mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max()), "pearson": float(np.corrcoef(f, p)[0, 1]) if f.std() > 0 and p.std() > 0 else None,
            "n_same_side": int(((f > 0.5) == (p > 0.5)).sum()), "n_pairs": int(f.size),
            "n_within_0.1": int((d <= 0.1).sum()), "T": a.T, "n_samples": a.n, "wordings": a.wordings}
    print(f"== {a.model}: unparseable {summ['unparseable_rate']:.3f}, mean |diff| {summ['mean_abs_diff']:.3f}, max {summ['max_abs_diff']:.3f}, "
          f"corr {summ['pearson']}, same side {summ['n_same_side']}/{summ['n_pairs']}, within 0.1 {summ['n_within_0.1']}/{summ['n_pairs']}")
    prev["models"][a.model] = {"summary": summ, "rows": rows, "manifest": build_manifest(vars(a), [0])}
    write_json(out_path, prev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
