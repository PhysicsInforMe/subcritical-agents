"""Phase A prompts and tabulation at the poor channel (design notes §5, settimane 2-3).

State of an agent for the tabulated policy (Flint et al., extended §4.3):
    (peer configuration seen, own memory of the last k_mem states)
where the peer configuration is either the ordered list of the ``exposure``
peer states shown (channel "list") or just the counts (channel "count",
Cho's Count format, §10.23). The policy is P(+ | peers, memory) read from
the logits after the JSON prefix, both flip sectors.

Cost control: the flipped sector is the same set of prompts with the labels
swapped, so its gaps follow from the unflipped ones by
    gap_B(states, memory) = -gap_A(-states, -memory)
and are never recomputed (§16, lesson 1). For large exposure the number
of orderings with a given sum grows as C(e, n+); at most ``max_orderings``
per (sum, memory) are tabulated, drawn with a fixed seed, and the variance
across them is still reported (§14.1 B3).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .tabulate import LABELS, STATE_FORMATS, LogitReader


def render_peers(channel: str, states: tuple[int, ...], plus_label: str, state_format: str) -> str:
    minus_label = LABELS[1] if plus_label == LABELS[0] else LABELS[0]
    if channel == "list":
        shown = [plus_label if s == 1 else minus_label for s in states]
        return STATE_FORMATS[state_format](shown)
    if channel == "count":
        n_plus = sum(1 for s in states if s == 1)
        n_minus = len(states) - n_plus
        return f"{n_plus} of them hold {plus_label} and {n_minus} hold {minus_label}"
    raise ValueError(channel)


def render_memory(memory: tuple[int, ...], plus_label: str) -> str:
    """Own past states, oldest first; empty string when k_mem = 0."""
    if not memory:
        return ""
    minus_label = LABELS[1] if plus_label == LABELS[0] else LABELS[0]
    shown = ", ".join(plus_label if s == 1 else minus_label for s in memory)
    return f" In the previous rounds Player 1 held, oldest first: {shown}."


def build_prompt_a(wording: str, channel: str, states: tuple[int, ...], memory: tuple[int, ...],
                   plus_label: str, state_format: str) -> str:
    """Wordings use the slots {a} {b} {peers} {memory} {n_peers}."""
    return wording.format(a=LABELS[0], b=LABELS[1], n_peers=len(states),
                         peers=render_peers(channel, states, plus_label, state_format),
                         memory=render_memory(memory, plus_label))


def peer_configurations(exposure: int, channel: str, max_orderings: int, rng: np.random.Generator
                        ) -> dict[int, list[tuple[int, ...]]]:
    """Configurations to tabulate, keyed by k = sum of peer states."""
    out: dict[int, list[tuple[int, ...]]] = {}
    for n_plus in range(exposure + 1):
        k = 2 * n_plus - exposure
        if channel == "count":
            out[k] = [tuple([1] * n_plus + [-1] * (exposure - n_plus))]
            continue
        if exposure <= 10:
            # enumerate all distinct orderings, then draw max_orderings of
            # them by index: the original procedure, kept unchanged so that
            # every cell of the main grid (e <= 7) and the e = 10 cells stay
            # reproducible from their config and seed
            allo = sorted(set(itertools.permutations([1] * n_plus + [-1] * (exposure - n_plus))))
            if len(allo) > max_orderings:
                idx = rng.choice(len(allo), size=max_orderings, replace=False)
                allo = [allo[i] for i in sorted(idx)]
        else:
            # beyond e = 10 the e! permutations cannot be enumerated (the
            # 2026-09-12 run at e = 15 stalled on 15!): sample distinct
            # orderings directly as random permutations of the multiset
            base = [1] * n_plus + [-1] * (exposure - n_plus)
            n_distinct = math.comb(exposure, n_plus)
            seen: set[tuple[int, ...]] = set()
            while len(seen) < min(max_orderings, n_distinct):
                seen.add(tuple(int(s) for s in rng.permutation(base)))
            allo = sorted(seen)
        out[k] = allo
    return out


def memory_states(k_mem: int) -> list[tuple[int, ...]]:
    return [()] if k_mem == 0 else [tuple(m) for m in itertools.product((-1, 1), repeat=k_mem)]


def tabulate_cell(reader: LogitReader, wording: str, channel: str, exposure: int, k_mem: int,
                  state_format: str, max_orderings: int, seed: int = 0, with_nll: bool = False
                  ) -> dict:
    """Sector A gaps for every (memory, k, ordering); sector B derived.

    Returns {"entries": [{"memory": [...], "k": k, "states": [...],
    "gap": l_+ - l_-, "logp_plus":..., "logp_minus":..., "nll":...}],
    "n_passes": int, "n_tokens": int}.
    """
    rng = np.random.default_rng(seed)
    configs = peer_configurations(exposure, channel, max_orderings, rng)
    entries, ntok = [], None
    for mem in memory_states(k_mem):
        for k, orderings in configs.items():
            for states in orderings:
                text = build_prompt_a(wording, channel, states, mem, LABELS[0], state_format)
                lA, lB, ntok = reader.label_logits(text)
                e = {"memory": list(mem), "k": int(k), "states": list(states), "gap": lA - lB,
                     "logp_plus": lA, "logp_minus": lB}
                if with_nll:
                    e["nll"] = reader.text_nll(text)[0]
                entries.append(e)
    return {"entries": entries, "n_passes": len(entries), "n_tokens": ntok}


def policy_tables(entries: list[dict], T: float) -> tuple[dict, dict, dict]:
    """Ordering-averaged P(+ | k, memory) for sector A, sector B (derived by
    the flip symmetry) and the symmetrised table, at temperature T.

    Keys are (k, memory tuple). The flip maps (k, m) of sector A to
    (-k, -m) of sector B with P_B = 1 - P_A.
    """
    acc: dict[tuple, list[float]] = {}
    for e in entries:
        key = (e["k"], tuple(e["memory"]))
        acc.setdefault(key, []).append(1.0 / (1.0 + math.exp(-e["gap"] / T)))
    pA = {key: float(np.mean(v)) for key, v in acc.items()}
    varA = {key: float(np.var(v)) for key, v in acc.items()}
    pB = {(-k, tuple(-s for s in m)): 1.0 - p for (k, m), p in pA.items()}
    p_sym = {key: 0.5 * (pA[key] + pB[key]) for key in pA if key in pB}
    return pA, pB, p_sym | {"_var": varA}


def label_space_fit(entries: list[dict], T: float, k_mem: int, n_boot: int = 300,
                    rng: np.random.Generator | None = None) -> dict:
    """Fit logit P(+ | k, memory) = 2(h + J k + J_mem * sum(memory)) in sector A.

    The memory enters through the sum of the own past states, the simplest
    one-parameter extension; J_mem is the temporal coupling (§3.3, memory
    as the range of the temporal interaction). Ordering bootstrap as in
    fit_table. Returns estimates with ci95.
    """
    rng = rng or np.random.default_rng(0)
    groups: dict[tuple, list[float]] = {}
    for e in entries:
        groups.setdefault((e["k"], tuple(e["memory"])), []).append(e["gap"])

    def _fit(sample: dict[tuple, list[float]]):
        rows, y = [], []
        for (k, m), gaps in sample.items():
            p = float(np.mean([1.0 / (1.0 + math.exp(-g / T)) for g in gaps]))
            p = min(max(p, 1e-9), 1 - 1e-9)
            row = [1.0, float(k)] + ([float(sum(m))] if k_mem > 0 else [])
            rows.append(row)
            y.append(math.log(p / (1 - p)))
        X = np.array(rows)
        coef, *_ = np.linalg.lstsq(X, np.array(y), rcond=None)
        return coef / 2.0

    names = ["h", "J"] + (["J_mem"] if k_mem > 0 else [])
    point = _fit(groups)
    boots = []
    for _ in range(n_boot):
        sample = {key: list(rng.choice(v, size=len(v), replace=True)) for key, v in groups.items()}
        boots.append(_fit(sample))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {n: {"estimate": float(point[i]), "ci95": [float(lo[i]), float(hi[i])]} for i, n in enumerate(names)}
