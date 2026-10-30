#!/usr/bin/env python
"""Toy test: does the prompt formulation move (J_tilde, h_tilde), and do
objective covariates explain the movement? (design notes §14.1 B15, §7.)

For each of the formulations in the config (wordings x list formats):
    * tabulate the logit gaps for all neighbour configurations in both flip
      sectors (32 forward passes), plus the prompt NLL of each instantiated
      prompt (Gonen et al.);
    * fit h_tilde, J_tilde with the ordering bootstrap;
    * covariates: mean prompt NLL, H_0 = entropy of P(+ | k = 0), and the
      POSIX index of the formulation against all the others computed on the
      two label log-probabilities at k = 0 (Chatterjee et al.).
Then compare the spread of J_tilde and h_tilde across formulations with the
bootstrap half-width within a formulation, and report Spearman correlations
of the parameters with the covariates. The decision rule for Phase A is
printed at the end: the curve J_eff(channel) is only declared if the
between-formulation spread is smaller than the effect between channel
levels, and the covariates tell whether that spread is explained.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402
from src.llm import LogitReader, fit_table, policy_at_temperature, tabulate_policy  # noqa: E402
from src.measures import shannon_entropy  # noqa: E402


def _sector_avg_logp(gaps: dict, k: str) -> tuple[float, float]:
    """Sector- and ordering-averaged log P(+) and log P(-) at bin k."""
    lp, lm = [], []
    for sector in ("plus_is_A", "plus_is_B"):
        lp += gaps[sector][k]["logp_plus"]
        lm += gaps[sector][k]["logp_minus"]
    return float(np.mean(lp)), float(np.mean(lm))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_paraphrases.yaml")
    ap.add_argument("--out", default="results/toy/paraphrases.json")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    T = float(cfg["temperature"])
    reader = LogitReader(cfg["model"]["name"], cfg["model"].get("revision"))
    rows = []
    t0 = time.time()
    for wi, wording in enumerate(cfg["wordings"]):
        for fmt in cfg["formats"]:
            gaps = tabulate_policy(reader, z=4, template=wording, state_format=fmt, with_nll=True)
            # label-space fit on sector A: h is the label bias, J the coupling.
            # (merging the two sectors would zero h by construction, see
            # scripts/phase0_denobili.py)
            fit = fit_table({k: gaps["plus_is_A"][k]["gap"] for k in gaps["plus_is_A"]}, T, n_boot=cfg["n_boot"])
            # P(+ | k = 0) in label space (sector A), consistent with the fit above
            p0 = float(np.mean([policy_at_temperature(g, 0.0, T) for g in gaps["plus_is_A"]["0"]["gap"]]))
            nll = float(np.mean([v for s in gaps.values() for blk in s.values() for v in blk["prompt_nll"]]))
            lp_plus0, lp_minus0 = _sector_avg_logp(gaps, "0")
            rows.append({"wording": wi, "format": fmt, "fit": fit, "p_plus_k0": p0,
                         "H0_bits": shannon_entropy(np.array([p0, 1 - p0])),
                         "prompt_nll": nll, "logp_plus_k0": lp_plus0, "logp_minus_k0": lp_minus0,
                         "n_tokens": gaps["plus_is_A"]["0"]["n_tokens"], "gaps": gaps})
            print(f"w{wi} {fmt:6s} h={fit['h']['estimate']:+.3f} J={fit['J']['estimate']:+.3f} "
                  f"H0={rows[-1]['H0_bits']:.3f} nll={nll:.3f} ({time.time() - t0:.0f} s)", flush=True)

    # POSIX on the two labels at k = 0: for formulation j take its preferred
    # label y_j, compare log P(y_j | x_i) across formulations i.
    n = len(rows)
    for j, rj in enumerate(rows):
        key = "logp_plus_k0" if rj["p_plus_k0"] >= 0.5 else "logp_minus_k0"
        rj["posix"] = float(np.mean([abs(rows[i][key] - rj[key]) for i in range(n) if i != j]))

    J = np.array([r["fit"]["J"]["estimate"] for r in rows])
    h = np.array([r["fit"]["h"]["estimate"] for r in rows])
    halfJ = np.mean([(r["fit"]["J"]["ci95"][1] - r["fit"]["J"]["ci95"][0]) / 2 for r in rows])
    halfh = np.mean([(r["fit"]["h"]["ci95"][1] - r["fit"]["h"]["ci95"][0]) / 2 for r in rows])
    cov = {c: np.array([r[c] for r in rows]) for c in ("prompt_nll", "H0_bits", "posix")}
    corr = {p: {c: spearmanr(vals, cov[c]).correlation for c in cov} for p, vals in (("J", J), ("h", h))}
    by_format = {f: {"J_mean": float(J[[r["format"] == f for r in rows]].mean()),
                     "h_mean": float(h[[r["format"] == f for r in rows]].mean())} for f in cfg["formats"]}
    summary = {
        "J": {"mean": float(J.mean()), "sd_between": float(J.std(ddof=1)), "ci_halfwidth_within": float(halfJ),
              "min": float(J.min()), "max": float(J.max())},
        "h": {"mean": float(h.mean()), "sd_between": float(h.std(ddof=1)), "ci_halfwidth_within": float(halfh),
              "min": float(h.min()), "max": float(h.max())},
        "spearman": corr, "by_format": by_format,
        "h_over_J_mean": float(np.mean(np.abs(h) / np.maximum(np.abs(J), 1e-9))),
    }
    print("\nJ: mean %.3f, between-formulation sd %.3f, within ci half-width %.3f, range [%.3f, %.3f]"
          % (summary["J"]["mean"], summary["J"]["sd_between"], halfJ, J.min(), J.max()))
    print("h: mean %.3f, between-formulation sd %.3f, within ci half-width %.3f, range [%.3f, %.3f]"
          % (summary["h"]["mean"], summary["h"]["sd_between"], halfh, h.min(), h.max()))
    print("Spearman J vs", {c: round(v, 2) for c, v in corr["J"].items()})
    print("Spearman h vs", {c: round(v, 2) for c, v in corr["h"].items()})
    print("by format", by_format)
    write_json(a.out, {"manifest": build_manifest(cfg, [], {"revision": reader.resolved_revision}),
                       "rows": rows, "summary": summary})
    print("written", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
