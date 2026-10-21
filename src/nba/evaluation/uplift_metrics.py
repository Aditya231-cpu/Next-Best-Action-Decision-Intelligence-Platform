"""Held-out uplift ranking metrics for randomized treatment-control data."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

def qini_curve(scores, treatment, outcome) -> pd.DataFrame:
    """Build a tie-aware Qini curve (higher score means larger expected treatment effect).

    Points are emitted at score-group boundaries. Thus tied scores do not acquire a
    fictitious within-group ordering; downstream interpolation represents random
    ordering inside a tied group. Qini is cumulative treated outcome minus the
    treated count times the cumulative control mean.
    """
    score, t, y = np.asarray(scores, float), np.asarray(treatment, bool), np.asarray(outcome, float)
    if not (len(score) == len(t) == len(y)) or not len(y): raise ValueError("Inputs must have equal nonzero length")
    if not (np.isfinite(score).all() and np.isfinite(y).all()): raise ValueError("Scores and outcomes must be finite")
    order = np.argsort(-score, kind="stable")
    score, t, y = score[order], t[order], y[order]
    n_t, n_c = int(t.sum()), int((~t).sum())
    if n_t == 0 or n_c == 0: raise ValueError("Both treatment and control groups are required")
    ends = np.r_[np.flatnonzero(score[1:] != score[:-1]), len(score) - 1]
    c_t = np.cumsum(t)[ends]; c_c = np.cumsum(~t)[ends]
    s_t = np.cumsum(y * t)[ends]; s_c = np.cumsum(y * (~t))[ends]
    control_mean = np.divide(s_c, c_c, out=np.zeros(len(ends), float), where=c_c > 0)
    uplift = s_t - c_t * control_mean
    return pd.DataFrame({"fraction_targeted": (ends + 1) / len(y), "qini": uplift})

def qini_coefficient(scores, treatment, outcome) -> float:
    """Area above the random-ranking line, normalized by total sample uplift."""
    curve = qini_curve(scores, treatment, outcome)
    x, y = np.r_[0., curve.fraction_targeted.to_numpy()], np.r_[0., curve.qini.to_numpy()]
    random = x * y[-1]
    denom = abs(y[-1])
    return float(trapezoid(y - random, x) / denom) if denom > 0 else 0.0

def auuc(scores, treatment, outcome) -> float:
    """Area under the cumulative incremental-outcome curve (units: outcome × sample fraction)."""
    curve = qini_curve(scores, treatment, outcome)
    x = np.r_[0., curve.fraction_targeted.to_numpy()]
    y = np.r_[0., curve.qini.to_numpy()]
    return float(trapezoid(y, x))

def uplift_at_fraction(scores, treatment, outcome, fractions=(.05, .10, .20)) -> dict[float, float]:
    curve = qini_curve(scores, treatment, outcome)
    x = np.r_[0., curve.fraction_targeted.to_numpy()]
    y = np.r_[0., curve.qini.to_numpy()]
    return {float(frac): float(np.interp(frac, x, y)) for frac in fractions}


def ranking_bootstrap(scores, treatment, outcome, *, draws: int = 300, seed: int = 0,
                      fractions=(.05, .10, .20)) -> dict:
    """Stratified customer bootstrap intervals for Qini and uplift-at-k."""
    score, t, y = np.asarray(scores, float), np.asarray(treatment, bool), np.asarray(outcome, float)
    treated, control = np.flatnonzero(t), np.flatnonzero(~t)
    if not len(treated) or not len(control): raise ValueError("Both randomized arms are required")
    rng = np.random.default_rng(seed)
    qini_values, at_k = [], {float(f): [] for f in fractions}
    for _ in range(draws):
        idx = np.r_[rng.choice(treated, len(treated), replace=True), rng.choice(control, len(control), replace=True)]
        qini_values.append(qini_coefficient(score[idx], t[idx], y[idx]))
        values = uplift_at_fraction(score[idx], t[idx], y[idx], fractions)
        for frac in fractions: at_k[float(frac)].append(values[float(frac)])
    alpha = .025
    return {"draws": draws, "confidence": .95,
        "qini_coefficient": {"low": float(np.quantile(qini_values, alpha)), "high": float(np.quantile(qini_values, 1-alpha))},
        "uplift_at_fraction": {str(frac): {"low": float(np.quantile(vals, alpha)), "high": float(np.quantile(vals, 1-alpha))}
            for frac, vals in at_k.items()}}
