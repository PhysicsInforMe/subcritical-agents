"""Glauber-response regressions (design notes §4.2 test 0, §4.3, §14.1 B1).

Two-term (De Nobili et al.):
    logit P(s' = +1 | k) = 2 (h + J k),          k = sum of observed peers.
Three-term (ours, with the truth-aligned designed field):
    logit P(s' = +1 | k, y) = 2 (h_bias + h_spec y + J k),   y = +-1.

Two estimators are provided.

* ``fit_glauber_counts``: De Nobili's estimator. Updates are binned by
  (k, y), P_+ is estimated per bin with Jeffreys smoothing
  (n_+ + 1/2)/(n + 1), and a weighted linear regression of logit P_+ on the
  regressors with weights n(bin) gives the coefficients (divided by 2).
  Bins with n < min_count are dropped (App. B of De Nobili).
* ``fit_glauber_mle``: maximum-likelihood logistic regression on the
  individual updates (El et al., Okawa App. C.3). Preferred when the
  raw updates are available; the two must agree on synthetic data.

Both return a dict with point estimates and, if ``n_boot`` > 0, percentile
bootstrap intervals over resampled updates.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _design(k: np.ndarray, y: np.ndarray | None) -> tuple[np.ndarray, list[str]]:
    k = np.asarray(k, dtype=float)
    cols = [np.ones_like(k), k]
    names = ["h", "J"]
    if y is not None:
        cols.insert(1, np.asarray(y, dtype=float))
        names = ["h_bias", "h_spec", "J"]
    return np.vstack(cols).T, names


def jeffreys_logit_table(k: np.ndarray, s_new: np.ndarray, y: np.ndarray | None = None
                         ) -> dict[tuple, dict[str, float]]:
    """Per-bin Jeffreys-smoothed P_+ and counts, keyed by (k,) or (k, y)."""
    k = np.asarray(k)
    s_new = np.asarray(s_new)
    keys = list(zip(k, y)) if y is not None else [(kk,) for kk in k]
    table: dict[tuple, dict[str, float]] = {}
    for key in set(keys):
        mask = np.array([kk == key for kk in keys])
        n = int(mask.sum())
        n_plus = int((s_new[mask] == 1).sum())
        p = (n_plus + 0.5) / (n + 1.0)
        table[key] = {"n": n, "n_plus": n_plus, "p_plus": p}
    return table


def fit_glauber_counts(k: np.ndarray, s_new: np.ndarray, y: np.ndarray | None = None,
                       min_count: int = 5, n_boot: int = 0,
                       rng: np.random.Generator | None = None) -> dict:
    """De Nobili's weighted logit regression on Jeffreys-smoothed bins."""
    def _fit(kk, ss, yy):
        table = jeffreys_logit_table(kk, ss, yy)
        keys = [key for key, v in table.items() if v["n"] >= min_count]
        if len(keys) < (3 if yy is not None else 2):
            raise ValueError("too few populated bins")
        kb = np.array([key[0] for key in keys], dtype=float)
        yb = np.array([key[1] for key in keys], dtype=float) if yy is not None else None
        X, names = _design(kb, yb)
        logit = np.array([np.log(table[key]["p_plus"] / (1 - table[key]["p_plus"])) for key in keys])
        w = np.array([table[key]["n"] for key in keys], dtype=float)
        XtW = X.T * w
        coef = np.linalg.solve(XtW @ X, XtW @ logit) / 2.0
        return dict(zip(names, coef))

    point = _fit(k, s_new, y)
    out = {name: {"estimate": float(v)} for name, v in point.items()}
    if n_boot > 0:
        rng = rng or np.random.default_rng(0)
        _add_bootstrap(out, _fit, k, s_new, y, n_boot, rng)
    return out


def fit_glauber_mle(k: np.ndarray, s_new: np.ndarray, y: np.ndarray | None = None,
                    n_boot: int = 0, rng: np.random.Generator | None = None,
                    ridge: float = 1e-6) -> dict:
    """Maximum-likelihood logistic regression, coefficients divided by 2."""
    def _fit(kk, ss, yy):
        X, names = _design(kk, yy)
        t = (np.asarray(ss) == 1).astype(float)

        def nll(beta):
            z = X @ beta
            # log-likelihood of Bernoulli with logit z, numerically stable
            ll = t * z - np.logaddexp(0.0, z)
            return -ll.sum() + ridge * (beta @ beta)

        def grad(beta):
            z = X @ beta
            return -X.T @ (t - expit(z)) + 2 * ridge * beta

        res = minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B")
        if not res.success:  # pragma: no cover
            raise RuntimeError(res.message)
        return dict(zip(names, res.x / 2.0))

    point = _fit(k, s_new, y)
    out = {name: {"estimate": float(v)} for name, v in point.items()}
    if n_boot > 0:
        rng = rng or np.random.default_rng(0)
        _add_bootstrap(out, _fit, k, s_new, y, n_boot, rng)
    return out


def _add_bootstrap(out, fit_fn, k, s_new, y, n_boot, rng) -> None:
    n = len(k)
    k = np.asarray(k)
    s_new = np.asarray(s_new)
    y = None if y is None else np.asarray(y)
    boots: dict[str, list[float]] = {name: [] for name in out}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            b = fit_fn(k[idx], s_new[idx], None if y is None else y[idx])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        for name, v in b.items():
            boots[name].append(v)
    for name in out:
        if boots[name]:
            lo, hi = np.percentile(boots[name], [2.5, 97.5])
            out[name]["ci95"] = [float(lo), float(hi)]
            out[name]["boot_se"] = float(np.std(boots[name]))


def synthetic_updates(n: int, J: float, h_bias: float, h_spec: float = 0.0,
                      k_values: tuple[int, ...] = (-4, -2, 0, 2, 4),
                      k_probs: np.ndarray | None = None, with_truth: bool = True,
                      rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Draw n Glauber updates from the three-term model on a k distribution.

    The default k distribution mimics a 2D lattice near the crossover, where
    the extreme bins k = +-4 are rarer than k = 0 (De Nobili App. B on rare
    bins). Returns k, y (or None), s_new.
    """
    rng = rng or np.random.default_rng(0)
    k_values = np.asarray(k_values)
    if k_probs is None:
        k_probs = np.array([1, 4, 6, 4, 1], dtype=float)[: k_values.size]
        k_probs = k_probs / k_probs.sum()
    k = rng.choice(k_values, size=n, p=k_probs)
    y = rng.choice(np.array([-1, 1]), size=n) if with_truth else None
    field = h_bias + J * k + (h_spec * y if with_truth else 0.0)
    p_plus = expit(2.0 * field)
    s_new = np.where(rng.random(n) < p_plus, 1, -1)
    return {"k": k, "y": y, "s_new": s_new}


def recovery_study(true: dict[str, float], n_updates: int, n_draws: int,
                   estimator=fit_glauber_mle, n_boot: int = 200,
                   rng: np.random.Generator | None = None) -> dict:
    """Repeated synthetic draws: coverage of the 95 % interval and relative
    bias per parameter (standard of Brockers et al., design notes §10.12)."""
    rng = rng or np.random.default_rng(0)
    names = list(true)
    hits = {n: 0 for n in names}
    ests = {n: [] for n in names}
    done = 0
    for _ in range(n_draws):
        d = synthetic_updates(n_updates, true["J"], true.get("h_bias", true.get("h", 0.0)),
                              true.get("h_spec", 0.0), with_truth="h_spec" in true, rng=rng)
        try:
            fit = estimator(d["k"], d["s_new"], d["y"], n_boot=n_boot, rng=rng)
        except (ValueError, RuntimeError):
            continue
        done += 1
        for n in names:
            key = n if n in fit else {"h": "h_bias", "h_bias": "h"}.get(n, n)
            est = fit[key]["estimate"]
            lo, hi = fit[key]["ci95"]
            ests[n].append(est)
            hits[n] += int(lo <= true[n] <= hi)
    out = {"n_draws": done, "coverage95": {}, "relative_bias": {}}
    for n in names:
        out["coverage95"][n] = hits[n] / max(done, 1)
        denom = abs(true[n]) if abs(true[n]) > 1e-12 else 1.0
        out["relative_bias"][n] = (float(np.mean(ests[n])) - true[n]) / denom
    return out
