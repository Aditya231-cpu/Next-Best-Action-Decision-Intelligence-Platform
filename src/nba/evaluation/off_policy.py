"""IPS and doubly robust estimators for known-propensity randomized policies."""
from __future__ import annotations
import numpy as np
import pandas as pd

def _prob_for_action(actions: np.ndarray, probabilities: dict[str, float] | pd.DataFrame, n: int) -> np.ndarray:
    if isinstance(probabilities, pd.DataFrame):
        if len(probabilities) != n: raise ValueError("Probability matrix row count does not match outcomes")
        return np.array([probabilities.iloc[i].get(a, 0.0) for i, a in enumerate(actions)], dtype=float)
    return np.array([probabilities.get(a, 0.0) for a in actions], dtype=float)

def inverse_propensity_value(observed_action, outcome, policy_action, propensity, *, clip: float | None = None) -> dict:
    """Estimate E[Y(policy(X))] for known randomization probabilities."""
    a, y, pi = np.asarray(observed_action), np.asarray(outcome, dtype=float), np.asarray(policy_action)
    if not (len(a) == len(y) == len(pi)) or len(y) == 0: raise ValueError("Inputs must have equal nonzero length")
    p = _prob_for_action(a, propensity, len(y))
    if np.any(p <= 0): raise ValueError("Observed assignments must have positive documented propensity")
    if np.any((p <= 0) & (a == pi)): raise ValueError("A matched policy action has zero assignment probability")
    matched = a == pi
    weights = np.divide(matched.astype(float), p, out=np.zeros_like(p), where=p > 0)
    if clip is not None: weights = np.minimum(weights, clip)
    contributions = weights * y
    ess = float(weights.sum() ** 2 / np.square(weights).sum()) if np.square(weights).sum() else 0.0
    return {"value": float(contributions.mean()), "standard_error": float(contributions.std(ddof=1) / np.sqrt(len(y))),
        "effective_sample_size": ess, "n": len(y), "estimator": "IPS", "clipped": clip is not None}

def doubly_robust_value(observed_action, outcome, policy_action, propensity, q_values: dict[str, np.ndarray]) -> dict:
    """DR policy value using outcome regression q_values[X, action] and known propensities."""
    a, y, pi = np.asarray(observed_action), np.asarray(outcome, dtype=float), np.asarray(policy_action)
    if not (len(a) == len(y) == len(pi)): raise ValueError("Inputs must have equal length")
    p = _prob_for_action(a, propensity, len(y))
    p_match = _prob_for_action(pi, propensity, len(y))
    if np.any(p <= 0): raise ValueError("Observed assignments must have positive documented propensity")
    if np.any(p_match <= 0): raise ValueError("Policy selected an action with zero assignment probability")
    mu_pi = np.array([q_values[action][i] for i, action in enumerate(pi)], dtype=float)
    mu_obs = np.array([q_values[action][i] for i, action in enumerate(a)], dtype=float)
    correction = np.divide((a == pi) * (y - mu_obs), p, out=np.zeros_like(y), where=p > 0)
    score = mu_pi + correction
    return {"value": float(score.mean()), "standard_error": float(score.std(ddof=1) / np.sqrt(len(y))),
        "n": len(y), "estimator": "doubly_robust"}

def paired_bootstrap_difference(policy_a, policy_b, *, draws: int = 2000, seed: int = 0, confidence: float = .95) -> dict:
    """Paired row bootstrap of two policy contribution vectors."""
    a, b = np.asarray(policy_a, dtype=float), np.asarray(policy_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2: raise ValueError("Paired contributions must be aligned one-dimensional arrays")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(draws, len(a)))
    diff = (a[idx] - b[idx]).mean(axis=1)
    alpha = (1 - confidence) / 2
    return {"difference": float((a - b).mean()), "ci_low": float(np.quantile(diff, alpha)),
        "ci_high": float(np.quantile(diff, 1 - alpha)), "confidence": confidence, "draws": draws}
