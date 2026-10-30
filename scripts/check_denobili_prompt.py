#!/usr/bin/env python
"""Replication of De Nobili et al. (arXiv:2605.10528, the reading notes (not included))
with their prompt taken literally, on our models (declared before the data
on 2026-09-14, the project log (not included); review of claims row A4).

Their protocol: L x L lattice, binary yes/no state, prompt
    "One rule: reply only yes/no. Your neighbours have states: [...].
     What state would you like to have?"
sampled answers, logistic fit on the update logs
    logit P(yes | k) = 2 (h~ + J~ k),  k = sum of the 4 neighbours,
global-flip ensemble with yes and no swapped in every prompt.

Here: same prompt text, 4 neighbours, all 16 orderings, the two label
logits read directly after the assistant prefix (no JSON), every single-
token variant of yes/no summed. The flipped sector is derived by symmetry
(the prompt with yes and no swapped for configuration s is the sector-A
prompt for -s). h~ and J~ from the regression of the mean gap on k (the
gap estimator of §21), J~ also from the odd part of the table. The ratio
|h~|/J~ does not depend on the decoding temperature, because both scale as
1/T. Output: results/phase0/denobili_literal_<model>.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402
from src.llm.tabulate import LogitReader, enumerate_orderings  # noqa: E402

PROMPT = "One rule: reply only yes/no. Your neighbours have states: {states}. What state would you like to have?"
LABELS_YN = ("yes", "no")


class YesNoReader(LogitReader):
    """LogitReader with yes/no labels and no JSON prefix: the answer is the
    first token of the reply, as in a sampled one-word answer."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.variant_ids = {}
        for lab in LABELS_YN:
            ids = set()
            for v in (lab, lab.capitalize(), lab.upper(), " " + lab, " " + lab.capitalize(), " " + lab.upper()):
                enc = self.tok.encode(v, add_special_tokens=False)
                if len(enc) == 1:
                    ids.add(enc[0])
            self.variant_ids[lab] = sorted(ids)

    def label_logits(self, user_text: str):
        import torch
        messages = [{"role": "user", "content": user_text}]
        try:
            chat = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            chat = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(chat, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1].double()
        logp = torch.log_softmax(logits, dim=-1)
        out = [float(torch.logsumexp(logp[self.variant_ids[lab]], dim=0)) for lab in LABELS_YN]
        return out[0], out[1], int(ids["input_ids"].shape[1])


def fit_gap(byk: dict[int, list[float]], rng: np.random.Generator, n_boot: int = 2000) -> dict:
    """h~, J~ from gap = 2(h + J k) on the ordering-averaged gaps; bootstrap over orderings within k."""
    def one(sample):
        ks = np.array(sorted(sample), float)
        g = np.array([np.mean(sample[int(k)]) for k in ks])
        c, *_ = np.linalg.lstsq(np.c_[np.ones_like(ks), ks], g, rcond=None)
        return c[0] / 2, c[1] / 2
    h, J = one(byk)
    boots = np.array([one({k: list(rng.choice(v, size=len(v), replace=True)) for k, v in byk.items()}) for _ in range(n_boot)])
    ratios = np.abs(boots[:, 0]) / np.maximum(np.abs(boots[:, 1]), 1e-9)
    return {"h": {"estimate": float(h), "ci95": np.percentile(boots[:, 0], [2.5, 97.5]).tolist()},
            "J": {"estimate": float(J), "ci95": np.percentile(boots[:, 1], [2.5, 97.5]).tolist()},
            "ratio_abs_h_over_J": {"estimate": float(abs(h) / max(abs(J), 1e-9)),
                                   "ci95": np.percentile(ratios, [2.5, 97.5]).tolist()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    t0 = time.time()
    reader = YesNoReader(a.model, dtype=a.dtype)
    print("variant ids:", reader.variant_ids, flush=True)
    entries, byk = [], {}
    for k in (-4, -2, 0, 2, 4):
        for states in enumerate_orderings(4, k):
            shown = ", ".join("yes" if s == 1 else "no" for s in states)
            text = PROMPT.format(states=shown)
            lyes, lno, ntok = reader.label_logits(text)
            entries.append({"k": k, "states": list(states), "gap": lyes - lno, "logp_yes": lyes, "logp_no": lno, "n_tokens": ntok})
            byk.setdefault(k, []).append(lyes - lno)
    rng = np.random.default_rng(0)
    fit = fit_gap(byk, rng)
    # J from the odd part of the table (symmetrised sectors), h-free
    odd = {k: 0.5 * (np.mean(byk[k]) - np.mean(byk[-k])) for k in (2, 4)}
    J_odd = float(np.polyfit([2.0, 4.0, -2.0, -4.0], [odd[2], odd[4], -odd[2], -odd[4]], 1)[0] / 2)
    mean_gap = {str(k): float(np.mean(v)) for k, v in byk.items()}
    p_yes = {str(k): float(np.mean(1 / (1 + np.exp(-np.array(v))))) for k, v in byk.items()}
    p_yes_r = {k: round(v, 3) for k, v in p_yes.items()}
    model_key = a.model.split("/")[-1].lower().replace("qwen3-", "qwen")
    out = {"manifest": build_manifest({"prompt": PROMPT, "labels": LABELS_YN, "model": a.model, "dtype": a.dtype}, [0],
                                      {"revision": reader.resolved_revision, "variant_ids": reader.variant_ids,
                                       "seconds": time.time() - t0}),
           "entries": entries, "mean_gap_by_k": mean_gap, "p_yes_by_k_T1": p_yes, "fit": fit, "J_odd": J_odd,
           "n_passes": len(entries)}
    path = Path("results/phase0") / f"denobili_literal_{model_key}.json"
    write_json(path, out)
    print(f"{model_key}: h={fit['h']['estimate']:+.3f} {np.round(fit['h']['ci95'], 3).tolist()} "
          f"J={fit['J']['estimate']:+.3f} {np.round(fit['J']['ci95'], 3).tolist()} "
          f"|h|/J={fit['ratio_abs_h_over_J']['estimate']:.2f} {np.round(fit['ratio_abs_h_over_J']['ci95'], 2).tolist()} "
          f"J_odd={J_odd:+.3f}; mean gap by k {mean_gap}; P(yes) by k {p_yes_r} "
          f"({time.time() - t0:.0f} s) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
