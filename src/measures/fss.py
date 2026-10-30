"""Finite-size scaling helpers.

    chi_max(L) ~ L^(gamma/nu)          (peak height)
    T_c(L)     = T_c + a L^(-1/nu)     (peak position, nu = 1 in 2D Ising)
    <|m|>(T_c) ~ L^(-beta/nu)
    U_4(L, T) curves cross at T_c      (h = 0 only, §14.1 B9)
"""
from __future__ import annotations

import numpy as np


def peak_parabolic(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Position and height of the maximum of y(x) by a parabola through the
    maximising grid point and its two neighbours (El et al. E.3 use the same
    interpolation for T_c). Falls back to the grid point at the edges."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    i = int(np.argmax(y))
    if i == 0 or i == y.size - 1:
        return float(x[i]), float(y[i])
    x0, x1, x2 = x[i - 1:i + 2]
    y0, y1, y2 = y[i - 1:i + 2]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / denom
    if a >= 0:
        return float(x[i]), float(y[i])
    xp = -b / (2 * a)
    yp = y1 + a * (xp - x1) ** 2 + b * (xp - x1) - (a * x1 ** 2 + b * x1) + (a * x1 ** 2 + b * x1)
    # evaluate the parabola exactly at xp
    c = y0 - a * x0 ** 2 - b * x0
    yp = a * xp ** 2 + b * xp + c
    return float(xp), float(yp)


def powerlaw_fit(L: np.ndarray, y: np.ndarray, y_err: np.ndarray | None = None) -> tuple[float, float]:
    """Weighted least squares of log y = c + s log L. Returns (s, se_s).

    Weights 1/sigma^2 with sigma = y_err / y on the log scale, as in
    De Nobili et al. App. C. Unweighted if y_err is None.
    """
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.vstack([np.ones_like(L), np.log(L)]).T
    Y = np.log(y)
    if y_err is None:
        W = np.ones_like(Y)
    else:
        W = (y / np.asarray(y_err, dtype=float)) ** 2
    XtW = X.T * W
    cov = np.linalg.inv(XtW @ X)
    coef = cov @ XtW @ Y
    resid = Y - X @ coef
    dof = max(Y.size - 2, 1)
    s2 = float((W * resid ** 2).sum() / dof) if y_err is None else 1.0
    se = float(np.sqrt(s2 * cov[1, 1]))
    return float(coef[1]), se


def tc_from_peaks(L: np.ndarray, T_peak: np.ndarray, nu: float = 1.0) -> tuple[float, float]:
    """Extrapolate T_c from peak positions with T_c(L) = T_c + a L^(-1/nu).
    Returns (T_c, a) by linear least squares in L^(-1/nu)."""
    L = np.asarray(L, dtype=float)
    x = L ** (-1.0 / nu)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, np.asarray(T_peak, dtype=float), rcond=None)
    return float(coef[0]), float(coef[1])


def binder_crossing(T: np.ndarray, U_small: np.ndarray, U_large: np.ndarray) -> float:
    """Temperature where two Binder curves cross, by linear interpolation of
    the sign change of U_small - U_large on the grid. Returns nan if they
    do not cross. Above T_c the smaller lattice has the larger U_4? No: for
    T > T_c the larger lattice is more disordered, U_large < U_small; for
    T < T_c, U_large > U_small. So the difference changes sign at T_c."""
    T = np.asarray(T, dtype=float)
    d = np.asarray(U_small, dtype=float) - np.asarray(U_large, dtype=float)
    sign = np.sign(d)
    idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    if idx.size == 0:
        return float("nan")
    # take the crossing closest to the middle of the window if several
    i = idx[np.argmin(np.abs(idx - T.size / 2))]
    t0, t1, d0, d1 = T[i], T[i + 1], d[i], d[i + 1]
    return float(t0 - d0 * (t1 - t0) / (d1 - d0))
