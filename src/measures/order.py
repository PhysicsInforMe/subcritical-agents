"""Order parameters and susceptibilities from a magnetisation time series."""
from __future__ import annotations

import numpy as np


def moments_from_m(m: np.ndarray) -> dict[str, float]:
    """<m>, <|m|>, <m^2>, <m^4> from a series of per-spin magnetisations."""
    m = np.asarray(m, dtype=float)
    return {
        "m": float(m.mean()),
        "abs_m": float(np.abs(m).mean()),
        "m2": float((m * m).mean()),
        "m4": float((m ** 4).mean()),
    }


def chi_signed(m: np.ndarray, N: int, T: float) -> float:
    """chi = (N/T) (<m^2> - <m>^2), the usual (signed) susceptibility."""
    mo = moments_from_m(m)
    return N / T * (mo["m2"] - mo["m"] ** 2)


def chi_abs(m: np.ndarray, N: int, T: float) -> float:
    """chi_|m| = (N/T) (<m^2> - <|m|>^2).

    Uses |m| so that the ordered phase of a finite system, where m flips
    sign, does not produce a trivial 1/T divergence (De Nobili et al.,
    design notes §4.2 test 2 and §11.5).
    """
    mo = moments_from_m(m)
    return N / T * (mo["m2"] - mo["abs_m"] ** 2)


def binder_cumulant(m: np.ndarray) -> float:
    """U_4 = 1 - <m^4> / (3 <m^2>^2) (Binder 1981).

    U_4 -> 2/3 deep in the ordered phase and -> 0 in the disordered phase for
    a Gaussian m; curves for different L cross at T_c when h = 0. Used only
    for the Ising calibration (design notes §14.1 B9).
    """
    mo = moments_from_m(m)
    return 1.0 - mo["m4"] / (3.0 * mo["m2"] ** 2)


def shannon_entropy(p: np.ndarray, base: float = 2.0) -> float:
    """H(p) = -sum p log p with 0 log 0 = 0."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(base))
