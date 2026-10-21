"""Stratified bootstrap uncertainty and effect-ranking stability utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd

def bootstrap_ate(frame: pd.DataFrame, *, treatment: str, control: str, outcome: str,
                  draws: int = 2000, seed: int = 0, confidence: float = .95) -> dict:
    """Bootstrap difference in arm means, resampling customers within randomized arms."""
    treated = frame.loc[frame.treatment == treatment, outcome].to_numpy(float)
    untreated = frame.loc[frame.treatment == control, outcome].to_numpy(float)
    if len(treated) < 2 or len(untreated) < 2: raise ValueError("At least two observations are required in each arm")
    rng = np.random.default_rng(seed)
    t_idx = rng.integers(0, len(treated), size=(draws, len(treated)))
    c_idx = rng.integers(0, len(untreated), size=(draws, len(untreated)))
    effects = treated[t_idx].mean(axis=1) - untreated[c_idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {"estimate": float(treated.mean() - untreated.mean()), "ci_low": float(np.quantile(effects, alpha)),
        "ci_high": float(np.quantile(effects, 1-alpha)), "bootstrap_sd": float(effects.std(ddof=1)),
        "draws": draws, "confidence": confidence, "n_treatment": len(treated), "n_control": len(untreated)}

def ranking_stability(effect_draws: np.ndarray, *, top_fraction: float = .1) -> dict:
    """Summarize chosen-action frequency and top-customer overlap over bootstrap prediction draws.

    `effect_draws` has shape (replicates, customers, actions); no latent true ITE is implied.
    """
    draws = np.asarray(effect_draws, dtype=float)
    if draws.ndim != 3 or draws.shape[0] < 2 or draws.shape[1] < 1 or draws.shape[2] < 1:
        raise ValueError("effect_draws must have shape (replicates>=2, customers>=1, actions>=1)")
    if not 0 < top_fraction <= 1: raise ValueError("top_fraction must be in (0, 1]")
    winners = np.argmax(draws, axis=2)
    action_freq = np.array([(winners == action).mean() for action in range(draws.shape[2])])
    k = max(1, int(np.ceil(draws.shape[1] * top_fraction)))
    top_sets = [set(np.argsort(-draws[r].max(axis=1), kind="stable")[:k]) for r in range(draws.shape[0])]
    overlaps = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            union = top_sets[i] | top_sets[j]
            overlaps.append(len(top_sets[i] & top_sets[j]) / len(union) if union else 1.0)
    return {"action_selection_frequency": action_freq.tolist(), "top_k": k,
        "mean_top_customer_jaccard": float(np.mean(overlaps)) if overlaps else 1.0,
        "replicates": draws.shape[0], "customers": draws.shape[1], "actions": draws.shape[2],
        "interpretation": "Bootstrap stability describes sensitivity of estimated decisions, not true individual effect accuracy."}
