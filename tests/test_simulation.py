from nba.simulation.runner import run_simulation
from nba.simulation.customer_state import CustomerState
from nba.simulation.response_model import response_probabilities
import numpy as np

def test_simulation_reproducible_and_explicitly_labeled():
    kwargs = dict(customers=120, periods=4, seed=77, contact_capacity_per_period=30)
    first = run_simulation(**kwargs)
    second = run_simulation(**kwargs)
    assert first.equals(second)
    assert first.result_type.str.contains("SIMULATED").all()
    assert first.cumulative_cost.is_monotonic_increasing
    assert first.cumulative_conversions.is_monotonic_increasing

def test_contact_cap_prevents_unbounded_repeated_interventions():
    result = run_simulation(customers=40, periods=8, seed=3, contact_cap=1, contact_capacity_per_period=40)
    assert result.interventions.sum() <= 40
    assert result.mean_contact_count.iloc[-1] <= 1

def test_simulation_invalid_settings_are_rejected():
    try:
        run_simulation(customers=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected invalid customer count to fail")

def test_fatigue_and_shift_reduce_synthetic_treatment_response():
    state = CustomerState(recency=np.array([6]), history=np.array([100.]), mens=np.array([1]),
        womens=np.array([0]), newbie=np.array([0]), contacts=np.array([0]))
    _, fresh = response_probabilities(state, period=0, periods=8, fatigue_strength=.2, distribution_shift=.4)
    contacted = CustomerState(state.recency, state.history, state.mens, state.womens, state.newbie, np.array([3]))
    _, fatigued = response_probabilities(contacted, period=7, periods=8, fatigue_strength=.2, distribution_shift=.4)
    assert fatigued["mens_email"][0] < fresh["mens_email"][0]
