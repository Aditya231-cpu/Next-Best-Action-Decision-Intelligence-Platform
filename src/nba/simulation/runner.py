"""Reproducible policy comparison in the explicitly synthetic environment."""
from __future__ import annotations
import numpy as np
import pandas as pd
from nba.simulation.customer_state import CustomerState
from nba.simulation.environment import run_period
from nba.simulation.response_model import SIM_ACTIONS, response_probabilities

def run_simulation(*, customers: int = 2000, periods: int = 8, seed: int = 20261018,
                   contact_cap: int = 3, contact_capacity_per_period: int = 400,
                   fatigue_strength: float = .12, distribution_shift: float = .15,
                   conversion_value: float = 45., action_costs: dict[str, float] | None = None,
                   policy: str = "incremental_net_value") -> pd.DataFrame:
    """Run one synthetic cohort trajectory; output explicitly represents simulation assumptions."""
    costs = action_costs or {"mens_email": .01, "womens_email": .01, "retention_offer": 5., "service_reminder": .25}
    if customers <= 0 or periods <= 0: raise ValueError("customers and periods must be positive")
    if contact_cap < 0 or contact_capacity_per_period < 0: raise ValueError("Contact caps and capacity must be nonnegative")
    if fatigue_strength < 0 or not 0 <= distribution_shift <= 1: raise ValueError("Invalid fatigue or distribution-shift intensity")
    if policy not in {"random", "response", "uplift", "incremental_net_value", "segment_rule"}:
        raise ValueError(f"Unknown simulator policy: {policy}")
    rng = np.random.default_rng(seed)
    state = CustomerState.sample(customers, rng)
    rows = []
    for period in range(periods):
        eligible = np.flatnonzero(state.contacts < contact_cap)
        base, potential = response_probabilities(state, period=period, periods=periods,
            fatigue_strength=fatigue_strength, distribution_shift=distribution_shift)
        capacity = min(contact_capacity_per_period, len(eligible))
        if policy == "random": rank = rng.random(len(eligible))
        elif policy == "response": rank = base[eligible]
        elif policy == "uplift": rank = np.max(np.column_stack([potential[a][eligible] - base[eligible] for a in SIM_ACTIONS]), axis=1)
        else:
            rank = np.max(np.column_stack([(potential[a][eligible] - base[eligible]) * conversion_value - costs[a] for a in SIM_ACTIONS]), axis=1)
        selected = eligible[np.argsort(-rank, kind="stable")[:capacity]]
        action_list = []
        for customer in selected:
            effects = {a: potential[a][customer] - base[customer] for a in SIM_ACTIONS}
            if policy == "segment_rule":
                action = "mens_email" if state.mens[customer] else "womens_email" if state.womens[customer] else "service_reminder"
            else:
                action = max(SIM_ACTIONS, key=lambda a: ((effects[a] * conversion_value - costs[a]) if policy == "incremental_net_value" else effects[a], a))
            action_list.append(action)
        rows.append(run_period(state, selected, action_list, period=period, periods=periods,
            fatigue_strength=fatigue_strength, distribution_shift=distribution_shift,
            conversion_value=conversion_value, action_costs=costs, rng=rng))
    result = pd.DataFrame(rows)
    result["policy"] = policy
    result["result_type"] = "SIMULATED — synthetic assumptions; not real banking or Hillstrom performance"
    result["cumulative_net_value"] = result.net_value.cumsum()
    result["cumulative_cost"] = result.cost.cumsum()
    result["cumulative_conversions"] = result.conversions.cumsum()
    return result
