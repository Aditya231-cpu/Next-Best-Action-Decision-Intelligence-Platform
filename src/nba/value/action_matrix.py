"""Build an action-by-customer net-value matrix from effects and explicit costs."""
from __future__ import annotations
import numpy as np
import pandas as pd
from nba.value.economics import net_value_from_spend

def spend_value_matrix(effects: dict[str, np.ndarray], costs: dict[str, float], *, index=None) -> pd.DataFrame:
    """Return net incremental spend values; absent treatment effects are never fabricated."""
    if not effects: raise ValueError("At least one empirically estimated action effect is required")
    if len({len(v) for v in effects.values()}) != 1: raise ValueError("Effect arrays must have equal length")
    n = len(next(iter(effects.values())))
    values = {"no_action": np.zeros(n)}
    for action, effect in effects.items():
        if action not in costs: raise KeyError(f"Missing explicit cost for action {action!r}")
        values[action] = net_value_from_spend(effect, costs[action])
    return pd.DataFrame(values, index=index)
