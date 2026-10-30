"""Measurement library (design notes §4).

Every function has a unit test against an analytic case or a known value
(tests/). Measures are computed downstream from raw states, never inline.
"""
from .order import (binder_cumulant, chi_abs, chi_signed, moments_from_m,
                    shannon_entropy)
from .bootstrap import bootstrap_ci, seed_bootstrap
from .fss import (binder_crossing, peak_parabolic, powerlaw_fit, tc_from_peaks)
from .regression import (fit_glauber_counts, fit_glauber_mle, jeffreys_logit_table,
                         recovery_study, synthetic_updates)
from .info import o_information_binary, total_correlation_binary
from .damage import damage_spreading
from .hysteresis import stubborn_sweep

__all__ = [
    "binder_cumulant", "chi_abs", "chi_signed", "moments_from_m", "shannon_entropy",
    "bootstrap_ci", "seed_bootstrap",
    "binder_crossing", "peak_parabolic", "powerlaw_fit", "tc_from_peaks",
    "fit_glauber_counts", "fit_glauber_mle", "jeffreys_logit_table",
    "recovery_study", "synthetic_updates",
    "o_information_binary", "total_correlation_binary",
    "damage_spreading", "stubborn_sweep",
]
