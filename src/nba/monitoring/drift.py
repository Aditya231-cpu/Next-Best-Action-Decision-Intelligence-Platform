"""Offline feature and observed-outcome drift summaries, kept separate from causal drift."""
from __future__ import annotations
import numpy as np
import pandas as pd

def _psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = reference.dropna().to_numpy(float); cur = current.dropna().to_numpy(float)
    if not len(ref) or not len(cur): return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(cur, ref[0]) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=edges)[0].astype(float); c = np.histogram(cur, bins=edges)[0].astype(float)
    r = np.clip(r / len(ref), 1e-6, 1); c = np.clip(c / len(cur), 1e-6, 1)
    return float(np.sum((c-r) * np.log(c/r)))

def feature_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Compare feature distributions and missingness; drift is not causal-effect drift."""
    shared = sorted(set(reference.columns) & set(current.columns))
    features = {}
    for col in shared:
        missing_ref, missing_cur = float(reference[col].isna().mean()), float(current[col].isna().mean())
        if pd.api.types.is_numeric_dtype(reference[col]) and pd.api.types.is_numeric_dtype(current[col]):
            score = _psi_numeric(reference[col], current[col])
            kind = "numeric_population_stability_index"
        else:
            levels = set(reference[col].dropna().astype(str)) | set(current[col].dropna().astype(str))
            pr = reference[col].astype("string").value_counts(normalize=True)
            pc = current[col].astype("string").value_counts(normalize=True)
            score = float(sum(abs(float(pr.get(level, 0))-float(pc.get(level, 0))) for level in levels) / 2)
            kind = "categorical_total_variation_distance"
        features[col] = {"metric": kind, "score": score, "reference_missing_rate": missing_ref,
            "current_missing_rate": missing_cur, "missing_rate_change": missing_cur-missing_ref}
    return {"reference_rows": len(reference), "current_rows": len(current), "feature_drift": features,
        "causal_effect_drift_assessed": False,
        "interpretation": "Feature distribution changes are not evidence of causal treatment-effect changes."}

def observed_outcome_shift(reference: pd.Series, current: pd.Series) -> dict:
    """Summarize observed outcome mean shift without causal interpretation."""
    a, b = reference.dropna().to_numpy(float), current.dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2: raise ValueError("At least two nonmissing outcomes per period are required")
    diff = float(b.mean()-a.mean())
    se = float(np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b)))
    return {"reference_mean": float(a.mean()), "current_mean": float(b.mean()), "observed_mean_change": diff,
        "standard_error": se, "ci95_low": diff-1.96*se, "ci95_high": diff+1.96*se,
        "causal_effect_drift_assessed": False,
        "interpretation": "Observed outcome movement may reflect population mix, exposure, or other changes; it is not an isolated treatment-effect estimate."}

def assignment_rate_drift(reference: pd.Series, current: pd.Series) -> dict:
    """Compare treatment assignment proportions and expose unseen/vanished arms."""
    ref = reference.astype("string").value_counts(normalize=True)
    cur = current.astype("string").value_counts(normalize=True)
    arms = sorted(set(ref.index) | set(cur.index))
    return {"reference_rates": {a: float(ref.get(a, 0.)) for a in arms},
        "current_rates": {a: float(cur.get(a, 0.)) for a in arms},
        "absolute_rate_changes": {a: float(cur.get(a, 0.)-ref.get(a, 0.)) for a in arms},
        "causal_effect_drift_assessed": False}
