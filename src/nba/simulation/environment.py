"""One synthetic period transition; no function here estimates real intervention effects."""
from __future__ import annotations
import numpy as np
from nba.simulation.customer_state import CustomerState
from nba.simulation.response_model import response_probabilities, draw_outcome

def run_period(state: CustomerState, selected: np.ndarray, actions: list[str], *, period: int,
               periods: int, fatigue_strength: float, distribution_shift: float,
               conversion_value: float, action_costs: dict[str, float], rng: np.random.Generator) -> dict:
    selected = np.asarray(selected, dtype=int)
    if len(selected) != len(actions): raise ValueError("Each selected customer must have exactly one action")
    if np.any((selected < 0) | (selected >= len(state.contacts))): raise IndexError("Selected customer index is out of bounds")
    base, probabilities = response_probabilities(state, period=period, periods=periods,
        fatigue_strength=fatigue_strength, distribution_shift=distribution_shift)
    conversions, spend = 0, 0.0
    for customer, action in zip(selected, actions):
        p = base[customer] if action == "no_action" else probabilities[action][customer]
        converted, amount = draw_outcome(float(p), conversion_value, rng)
        conversions += converted; spend += amount
        if action != "no_action": state.contacts[customer] += 1
    cost = float(sum(action_costs.get(action, 0.0) for action in actions))
    return {"period": period, "contacts": len(selected), "interventions": sum(a != "no_action" for a in actions),
        "conversions": conversions, "spend": spend, "cost": cost, "net_value": spend - cost,
        "mean_contact_count": float(state.contacts.mean())}
