"""Convert causal effect estimates into net-value estimates under explicit assumptions."""
from __future__ import annotations
import numpy as np

def net_value_from_spend(incremental_spend, action_cost) -> np.ndarray:
    """For monetary spend effects, subtract cost once; do not multiply by another margin."""
    return np.asarray(incremental_spend, float) - np.asarray(action_cost, float)

def net_value_from_conversion(incremental_probability, value_per_incremental_conversion: float, action_cost) -> np.ndarray:
    """Convert a binary outcome effect using an explicit assumed conversion value."""
    if value_per_incremental_conversion < 0: raise ValueError("Conversion value must be nonnegative")
    return np.asarray(incremental_probability, float) * value_per_incremental_conversion - np.asarray(action_cost, float)

def break_even_cost(incremental_spend) -> np.ndarray:
    """Maximum per-customer intervention cost before expected net value turns negative."""
    return np.asarray(incremental_spend, dtype=float)
