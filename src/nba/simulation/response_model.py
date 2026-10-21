"""Synthetic potential-response probabilities and sampled outcomes."""
from __future__ import annotations
import numpy as np
from nba.simulation.customer_state import CustomerState

SIM_ACTIONS = ("mens_email", "womens_email", "retention_offer", "service_reminder")
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def response_probabilities(state: CustomerState, *, period: int, periods: int, fatigue_strength: float,
                           distribution_shift: float, contact_count=None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return baseline and action probabilities from declared synthetic rules."""
    count = state.contacts if contact_count is None else np.asarray(contact_count)
    shift = 1 - distribution_shift * (period / max(periods - 1, 1))
    base_logit = -3.9 + .16 * np.log1p(state.history) + .04 * (6 - state.recency) + .18 * state.newbie
    base = sigmoid(base_logit)
    heterogeneity = {
        "mens_email": .20 + .42 * state.mens - .10 * state.womens,
        "womens_email": .16 + .48 * state.womens - .31 * state.mens,
        "retention_offer": .28 + .35 * state.newbie + .12 * (state.recency >= 9),
        "service_reminder": .11 + .10 * (state.recency >= 10),
    }
    fatigue = np.exp(-fatigue_strength * count)
    result = {action: sigmoid(base_logit + shift * heterogeneity[action] * fatigue) for action in SIM_ACTIONS}
    return base, result

def draw_outcome(probability: float, conversion_value: float, rng: np.random.Generator) -> tuple[int, float]:
    converted = int(rng.random() < probability)
    amount = float(rng.lognormal(np.log(conversion_value), .45)) if converted else 0.0
    return converted, amount
