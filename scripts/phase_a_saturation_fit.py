#!/usr/bin/env python
"""Test D of the results notes (not included) (protocol written before the data):
is the unanimity jump a saturating function of the number of peers,
    jump(e) = G * e / (e + e0),
with ceiling G and saturation scale e0 depending on the formulation?

For every formulation (list, no memory), all tabulated exposures
(2, 4, 7, 10, 15, 20, 30), direct jump gap(+e) - gap(-e):
    * least-squares fit of the saturating form and of the power law
      a * e^x, AIC of both (same number of parameters, so AIC compares
      residual sums of squares), and which one wins;
    * the apparent exponent over e >= 7 (as in exposure_shape) against
      the fitted e0: Spearman correlation (prediction D2);
    * dispersion across formulations of G against that of jump(7)
      (prediction D3).
Formulations whose jump is below 0.5 logit at some e are excluded from
the exponent, as in exposure_shape, but kept in the fits.
Output: results/phase_a/saturation_fit.json and a text summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402


def sat(e, G, e0):
    return G * e / (e + e0)


def pow_(e, a, x):
    return a * e ** x


def aic(y, pred, k):
    rss = float(((y - pred) ** 2).sum()); n = len(y)
    return n * np.log(max(rss, 1e-12) / n) + 2 * k


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase_a")
    ap.add_argument("--models", default="qwen0.6b,qwen1.7b")
    a = ap.parse_args(argv)
    d = json.load(open(Path(a.results) / "exposure_shape.json", encoding="utf-8"))
    out = {"models": {}}
    for model in a.models.split(","):
        r = d["models"][model]
        es = np.array(r["exposures"], float)
        n_f = r["n_formulations"]
        rows = []
        for i in range(n_f):
            y = np.array([r["jump"][str(int(e))]["by_formulation"][i] for e in es])
            row = {"formulation": i, "jumps": y.tolist()}
            try:
                (G, e0), _ = curve_fit(sat, es, y, p0=[max(y.max(), 1.0), 5.0], bounds=([-np.inf, 0.01], [np.inf, 1e4]), maxfev=20000)
                row["sat"] = {"G": float(G), "e0": float(e0), "aic": aic(y, sat(es, G, e0), 2)}
            except Exception as exc:
                row["sat"] = {"error": str(exc)}
            pos = y > 0
            if pos.sum() >= 3:
                try:
                    (aa, x), _ = curve_fit(pow_, es[pos], y[pos], p0=[max(y[pos].min(), 0.1), 0.2], maxfev=20000)
                    row["pow"] = {"a": float(aa), "x": float(x), "aic": aic(y[pos], pow_(es[pos], aa, x), 2), "n": int(pos.sum())}
                except Exception as exc:
                    row["pow"] = {"error": str(exc)}
            big = es >= 7
            row["exponent_e_ge_7"] = float(np.polyfit(np.log(es[big]), np.log(y[big]), 1)[0]) if (y[big] > 0.5).all() else None
            rows.append(row)
        ok = [w for w in rows if "aic" in w.get("sat", {}) and "aic" in w.get("pow", {})]
        wins = sum(w["sat"]["aic"] < w["pow"]["aic"] for w in ok)
        ex = [w for w in rows if w["exponent_e_ge_7"] is not None and "e0" in w.get("sat", {})]
        rho, p = spearmanr([w["exponent_e_ge_7"] for w in ex], [w["sat"]["e0"] for w in ex]) if len(ex) >= 4 else (None, None)
        Gs = [w["sat"]["G"] for w in rows if "G" in w.get("sat", {})]
        j7 = [w["jumps"][list(es).index(7.0)] for w in rows]
        summary = {"sat_beats_pow_aic": f"{wins}/{len(ok)}",
                   "spearman_exponent_e0": None if rho is None else {"rho": float(rho), "p": float(p), "n": len(ex)},
                   "G_sd": float(np.std(Gs, ddof=1)), "G_mean": float(np.mean(Gs)), "jump7_sd": float(np.std(j7, ddof=1)),
                   "e0_by_formulation": [w["sat"].get("e0") for w in rows], "G_by_formulation": Gs}
        out["models"][model] = {"exposures": es.tolist(), "rows": rows, "summary": summary}
        print(f"== {model}")
        for w in rows:
            s, pw = w.get("sat", {}), w.get("pow", {})
            print(f"  f{w['formulation']}: G {s.get('G', float('nan')):6.2f} e0 {s.get('e0', float('nan')):6.2f} AIC sat {s.get('aic', float('nan')):6.1f} | pow x {pw.get('x', float('nan')):5.2f} AIC {pw.get('aic', float('nan')):6.1f} | exponent(7-30) {w['exponent_e_ge_7']}")
        print(f"  D1 saturating beats power law (AIC): {summary['sat_beats_pow_aic']}")
        print(f"  D2 Spearman(exponent 7-30, e0): {summary['spearman_exponent_e0']}")
        print(f"  D3 sd of G across formulations {summary['G_sd']:.2f} (mean G {summary['G_mean']:.2f}) vs sd of jump(7) {summary['jump7_sd']:.2f}")
    write_json(Path(a.results) / "saturation_fit.json", {"manifest": build_manifest(vars(a), []), **out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
