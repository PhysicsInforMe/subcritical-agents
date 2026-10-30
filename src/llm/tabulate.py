"""Tabulate P(+ | neighbour configuration) from the logits of a small LLM.

Method of Flint et al. (PNAS 2026), sampled form in De Nobili et al.
(design notes §4.3): for every distinct neighbour configuration the prompt is
built, one forward pass gives the next-token distribution after a
constrained prefix, and the policy is read from it. Then the dynamics runs
in numpy at zero LLM cost.

Conventions adopted from the audit (§14.1):
    B5  constrained JSON prefix, all tokenisations of each label summed,
        thinking disabled, HF revision pinned in the manifest.
    B6  neutral labels A/B; the global-flip sector swaps which label is +1
        so that averaging the two sectors cancels the label bias.
    B4  the decoding temperature only rescales the two logits, so every
        temperature is obtained analytically from one forward pass.
    B3  all orderings of the neighbour list are enumerated and the variance
        of P(+) across orderings at fixed k is reported.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

LABELS = ["A", "B"]   # mutable on purpose: set_labels() changes it in place for every importer


def set_labels(first: str, second: str) -> None:
    """Replace the two label tokens (default A/B) for a cell whose YAML carries
    ``labels: [first, second]``; the first is the + label (slot {a}). Mutates
    the module list in place so that phase_a.py and motivation.py, which
    import the same object, see the change. Must be called before the
    LogitReader is built, because the reader caches the token variants of
    each label. Added on 2026-09-30 for the token-identity test of the label
    bias (the results notes (not included), correction of 2026-09-30)."""
    LABELS[0], LABELS[1] = str(first), str(second)


def enumerate_orderings(z: int, k: int) -> list[tuple[int, ...]]:
    """All ordered neighbour lists of length z with sum k (values +-1)."""
    n_plus = (z + k) // 2
    if (z + k) % 2 or not (0 <= n_plus <= z):
        raise ValueError(f"invalid (z, k) = ({z}, {k})")
    return sorted(set(itertools.permutations([1] * n_plus + [-1] * (z - n_plus))))


STATE_FORMATS = {
    # how the neighbour list is rendered: the format axis of Sclar et al. (§10.31)
    "comma": lambda shown: ", ".join(shown),
    "semicolon": lambda shown: "; ".join(shown),
    "lines": lambda shown: "\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(shown)),
    "brackets": lambda shown: "[" + ", ".join(shown) + "]",
}


def build_prompt(states: tuple[int, ...], plus_label: str, template: str,
                 state_format: str = "comma") -> str:
    """Fill the template with the neighbour labels for this flip sector."""
    minus_label = LABELS[1] if plus_label == LABELS[0] else LABELS[0]
    shown = [plus_label if s == 1 else minus_label for s in states]
    return template.format(states=STATE_FORMATS[state_format](shown), a=LABELS[0], b=LABELS[1])


DEFAULT_TEMPLATE = (
    "Player 1 is one of several players. Each player holds a state, either {a} or {b}. "
    "The neighbours of Player 1 currently hold the states: {states}. "
    "Answer with the state Player 1 should hold now, as JSON with a single key \"state\"."
)


@dataclass
class LogitReader:
    """Wraps a causal LM to read the two label logits after a JSON prefix."""
    model_name: str
    revision: str | None = None
    prefix: str = '{"state": "'
    device: str = "cpu"
    dtype: str = "float32"   # "bfloat16" halves memory on a 16 GB laptop; recorded in the manifest

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.revision,
                                                          dtype=getattr(torch, self.dtype),
                                                          low_cpu_mem_usage=True)
        self.model.eval()
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or self.revision
        # every tokenisation of each label that can follow the open quote
        self.variant_ids = {}
        for lab in LABELS:
            ids = set()
            for v in (lab, " " + lab, lab.lower(), " " + lab.lower()):
                enc = self.tok.encode(v, add_special_tokens=False)
                if len(enc) == 1:
                    ids.add(enc[0])
            self.variant_ids[lab] = sorted(ids)

    def text_nll(self, text: str) -> tuple[float, int]:
        """Mean per-token negative log-likelihood of ``text`` under the model
        (perplexity = exp of it). Used as the objective 'naturalness' of a
        prompt formulation (design notes §14.1 B15, Gonen et al.)."""
        import torch
        ids = self.tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
        with torch.no_grad():
            logits = self.model(input_ids=ids).logits[0, :-1].double()
        logp = torch.log_softmax(logits, dim=-1)
        target = ids[0, 1:]
        nll = -logp[torch.arange(target.numel()), target].mean()
        return float(nll), int(target.numel())

    def label_logits(self, user_text: str) -> tuple[float, float, int]:
        """Return (logit_A, logit_B, n_tokens) with logits log-sum-exp'd
        over the token variants of each label."""
        import torch
        messages = [{"role": "user", "content": user_text}]
        try:
            chat = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:  # tokenizers without the thinking switch
            chat = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(chat + self.prefix, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1].double()
        logp = torch.log_softmax(logits, dim=-1)
        out = []
        for lab in LABELS:
            out.append(float(torch.logsumexp(logp[self.variant_ids[lab]], dim=0)))
        return out[0], out[1], int(ids["input_ids"].shape[1])


def policy_at_temperature(logit_plus: float, logit_minus: float, T: float) -> float:
    """P(+) = sigma((l_+ - l_-)/T): the temperature rescales the gap (B4)."""
    return 1.0 / (1.0 + math.exp(-(logit_plus - logit_minus) / T))


def tabulate_policy(reader: LogitReader, z: int, template: str = DEFAULT_TEMPLATE,
                    state_format: str = "comma", with_nll: bool = False) -> dict[str, dict]:
    """Logit gaps for every ordering, both flip sectors.

    Returns {sector: {k: {"orderings": [...], "gap": [l_+ - l_-, ...],
    "logp_plus": [...], "logp_minus": [...], "n_tokens": int,
    "prompt_nll": [...] (if with_nll)}}} where sector "plus_is_A" shows +1
    as A and reads A as +1, and "plus_is_B" is the conjugate (global flip).
    ``logp_*`` are the log-probabilities (variants summed) of the two labels
    after the prefix, needed for the POSIX index (§14.1 B15).
    """
    out: dict[str, dict] = {}
    for sector, plus_label in (("plus_is_A", LABELS[0]), ("plus_is_B", LABELS[1])):
        out[sector] = {}
        for k in range(-z, z + 1, 2):
            gaps, lp_plus, lp_minus, nlls, ntok = [], [], [], [], None
            for states in enumerate_orderings(z, k):
                text = build_prompt(states, plus_label, template, state_format)
                lA, lB, ntok = reader.label_logits(text)
                lplus, lminus = (lA, lB) if plus_label == LABELS[0] else (lB, lA)
                gaps.append(lplus - lminus)
                lp_plus.append(lplus)
                lp_minus.append(lminus)
                if with_nll:
                    nlls.append(reader.text_nll(text)[0])
            out[sector][str(k)] = {"orderings": [list(s) for s in enumerate_orderings(z, k)],
                                   "gap": gaps, "logp_plus": lp_plus, "logp_minus": lp_minus,
                                   "n_tokens": ntok}
            if with_nll:
                out[sector][str(k)]["prompt_nll"] = nlls
    return out


def fit_table(gaps_by_k: dict[str, list[float]], T: float, n_boot: int = 500,
              rng: np.random.Generator | None = None) -> dict:
    """Two-term fit logit P(+|k) = 2(h + J k) on the ordering-averaged policy.

    P(+|k) is the mean over orderings of sigma(gap/T). The interval comes
    from resampling orderings within each k (the only randomness here is the
    ordering, the logits are exact).
    """
    rng = rng or np.random.default_rng(0)
    ks = np.array(sorted(int(k) for k in gaps_by_k), dtype=float)

    def _fit(sample):
        logit = []
        for k in ks:
            p = np.mean([policy_at_temperature(g, 0.0, T) for g in sample[str(int(k))]])
            p = min(max(p, 1e-9), 1 - 1e-9)
            logit.append(math.log(p / (1 - p)))
        A = np.vstack([np.ones_like(ks), ks]).T
        coef, *_ = np.linalg.lstsq(A, np.array(logit), rcond=None)
        return coef / 2.0

    point = _fit(gaps_by_k)
    boots = []
    for _ in range(n_boot):
        sample = {k: list(rng.choice(v, size=len(v), replace=True)) for k, v in gaps_by_k.items()}
        boots.append(_fit(sample))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"h": {"estimate": float(point[0]), "ci95": [float(lo[0]), float(hi[0])]},
            "J": {"estimate": float(point[1]), "ci95": [float(lo[1]), float(hi[1])]}}


def cycle_test(p_plus_by_k: dict[int, float], z: int = 4, L: int = 4, n_samples: int = 2000,
               seed: int = 0) -> dict[str, float]:
    """Kolmogorov cycle test of detailed balance for a k-only policy (§10.28).

    For a random configuration on an L x L periodic lattice and a random
    pair of neighbouring sites (i, j), the cycle
        c -> flip i -> flip j -> unflip i -> unflip j -> c
    has forward rates w and backward rates w'; detailed balance requires
        sum log w - sum log w' = 0.
    With P(+ | k) = sigma(2(h + J k)) the sum vanishes identically; a table
    whose logit is not linear in k gives a nonzero mean absolute sum.
    """
    rng = np.random.default_rng(seed)
    N = L * L

    def rate(s, site, k):
        """Probability that heat bath sets ``site`` to -s given current s."""
        p_plus = p_plus_by_k[int(k)]
        return (1 - p_plus) if s == 1 else p_plus

    def nbr_sum(conf, site):
        r, c = divmod(site, L)
        return conf[(r - 1) % L, c] + conf[(r + 1) % L, c] + conf[r, (c - 1) % L] + conf[r, (c + 1) % L]

    sums = []
    for _ in range(n_samples):
        conf = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
        i = int(rng.integers(N))
        r, c = divmod(i, L)
        j = ((r + 1) % L) * L + c if rng.random() < 0.5 else r * L + (c + 1) % L
        path = []
        cur = conf.copy()
        for site in (i, j, i, j):
            s = int(cur.flat[site])
            k = nbr_sum(cur, site)
            fwd = rate(s, site, k)
            cur.flat[site] = -s
            k_after = nbr_sum(cur, site)
            bwd = rate(-s, site, k_after)
            path.append(math.log(fwd) - math.log(bwd))
        sums.append(sum(path))
    sums = np.array(sums)
    return {"mean_abs_cycle_sum": float(np.abs(sums).mean()),
            "max_abs_cycle_sum": float(np.abs(sums).max()), "n_cycles": int(n_samples)}
