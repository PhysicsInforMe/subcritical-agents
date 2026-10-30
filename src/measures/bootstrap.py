"""Bootstrap over independent seeds (design notes §11.5: bands from seeds)."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean of ``values``: (mean, lo, hi)."""
    rng = rng or np.random.default_rng(0)
    v = np.asarray(values, dtype=float)
    boots = np.array([rng.choice(v, size=v.size, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def seed_bootstrap(per_seed: Sequence, statistic: Callable[[list], float],
                   n_boot: int = 1000, alpha: float = 0.05,
                   rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Bootstrap a statistic that needs the whole set of seed results.

    ``per_seed`` is a list with one entry per seed (any object). The seeds
    are resampled with replacement and ``statistic`` is recomputed on each
    resample. Returns (point estimate on all seeds, lo, hi).
    """
    rng = rng or np.random.default_rng(0)
    n = len(per_seed)
    point = float(statistic(list(per_seed)))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boots.append(float(statistic([per_seed[i] for i in idx])))
        except (ValueError, RuntimeError, FloatingPointError):
            continue
    if not boots:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
