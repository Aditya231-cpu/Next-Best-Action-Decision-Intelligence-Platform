"""Deterministic approximate allocator ranked by positive net benefit."""
from __future__ import annotations
import numpy as np
import pandas as pd
from decimal import Decimal

def allocate_greedy(values: pd.DataFrame, costs: dict[str, float], *, budget: float = float("inf"), capacities: dict[str, int | None] | None = None,
                    eligibility: pd.DataFrame | None = None) -> pd.DataFrame:
    """Choose at most one positive-net-benefit action/customer within constraints.

    `values` must already be incremental net benefits (gross effect less action
    cost exactly once). Costs here are only consumed by the budget constraint.
    The heuristic sorts by benefit, so it is feasible but not generally optimal
    when action costs differ.
    """
    capacities = capacities or {}
    if not values.index.is_unique: raise ValueError("Customer index must be unique")
    if not np.isfinite(values.to_numpy(dtype=float)).all(): raise ValueError("Action values must be finite")
    if np.isnan(budget) or budget < 0: raise ValueError("Budget must be nonnegative")
    if "no_action" not in values: values = values.assign(no_action=0.0)
    if not np.allclose(values["no_action"].to_numpy(dtype=float), 0.0): raise ValueError("no_action must have zero incremental net benefit")
    actions = [a for a in values if a != "no_action"]
    for a in actions:
        if a not in costs or not np.isfinite(costs[a]) or costs[a] < 0: raise ValueError(f"A finite nonnegative cost is required for action {a!r}")
    if any(cap is not None and cap < 0 for cap in capacities.values()): raise ValueError("Action capacities must be nonnegative")
    allowed = eligibility.reindex(index=values.index, columns=actions, fill_value=False).fillna(False).astype(bool) if eligibility is not None else pd.DataFrame(True, index=values.index, columns=actions)
    chosen = pd.Series("no_action", index=values.index, dtype="object")
    ranked = []
    for i, customer in enumerate(values.index):
        candidates = [(float(values.loc[customer, action]), action) for action in actions
                      if allowed.loc[customer, action] and float(values.loc[customer, action]) > 0]
        candidates.sort(key=lambda x: (-x[0], x[1]))
        if candidates: ranked.append((candidates[0][0], i, customer, candidates[0][1]))
    ranked.sort(key=lambda item: (-item[0], item[1], item[3]))
    total_cost_exact, counts = Decimal("0"), {a: 0 for a in actions}
    budget_exact = Decimal(str(budget))
    for _, _, customer, action in ranked:
        cap = capacities.get(action)
        cost = float(costs[action])
        if cap is not None and counts[action] >= cap: continue
        cost_exact = Decimal(str(cost))
        if total_cost_exact + cost_exact > budget_exact: continue
        chosen.loc[customer] = action; counts[action] += 1; total_cost_exact += cost_exact
    total_cost = float(total_cost_exact)
    return pd.DataFrame({"action": chosen, "cost": [float(costs.get(a, 0)) for a in chosen],
        "net_value": [float(values.loc[i, a]) if a in values else 0.0 for i, a in chosen.items()]}, index=values.index)
