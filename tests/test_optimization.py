import numpy as np
import pandas as pd
import pytest
from nba.optimization.constrained_solver import allocate_constrained
from nba.optimization.greedy import allocate_greedy
from nba.value.action_matrix import spend_value_matrix
from nba.value.economics import net_value_from_spend

def test_net_value_subtracts_cost_once_and_matrix_keeps_no_action():
    assert net_value_from_spend([3.0], 1.0)[0] == 2.0
    values = spend_value_matrix({"mens_email": np.array([2., -1.])}, {"mens_email": .1}, index=["a", "b"])
    assert list(values.columns) == ["no_action", "mens_email"]
    assert values.loc["b", "mens_email"] < values.loc["b", "no_action"]

def test_greedy_enforces_budget_capacity_and_eligibility():
    values = pd.DataFrame({"no_action": [0., 0., 0.], "mens": [10., 9., 8.], "womens": [8., 7., 6.]}, index=list("abc"))
    eligible = pd.DataFrame({"mens": [True, False, True], "womens": [True, True, True]}, index=values.index)
    allocation = allocate_greedy(values, {"mens": 2., "womens": 1.}, budget=2., capacities={"mens": 1, "womens": 2}, eligibility=eligible)
    assert allocation.cost.sum() <= 2.
    assert (allocation.action == "mens").sum() <= 1
    assert allocation.loc["b", "action"] != "mens"

def test_milp_solver_respects_allocation_constraints():
    values = pd.DataFrame({"no_action": [0., 0., 0.], "mens": [10., 9., 8.], "womens": [7., 7., 7.]}, index=list("abc"))
    eligibility = pd.DataFrame({"mens": [True, False, True], "womens": [True, True, True]}, index=values.index)
    result = allocate_constrained(values, {"mens": 2., "womens": 1.}, budget=2., capacities={"mens": 1, "womens": 2}, eligibility=eligibility)
    assert result["constraints_satisfied"]
    assert result["total_cost"] <= 2.
    assert result["action_counts"]["mens"] <= 1
    assert result["assignments"].loc["b"] != "mens"

def test_exact_optimizer_beats_greedy_on_small_knapsack_case():
    # Greedy takes value 10 at cost 10; the exact solution takes 9 + 7 at total cost 10.
    values = pd.DataFrame({"no_action": [0., 0., 0.], "expensive": [10., 0., 0.],
        "medium": [0., 9., 0.], "cheap": [0., 0., 7.]}, index=["a", "b", "c"])
    costs = {"expensive": 10., "medium": 6., "cheap": 4.}
    greedy = allocate_greedy(values, costs, budget=10.)
    exact = allocate_constrained(values, costs, budget=10.)
    assert greedy.net_value.sum() == 10.
    assert exact["status"] == "optimal"
    assert exact["total_value"] == 16.
    assert exact["total_cost"] == 10.

def test_optimizer_keeps_no_action_for_nonpositive_benefits_and_is_deterministic():
    values = pd.DataFrame({"no_action": [0., 0.], "offer": [-0.1, 0.0]}, index=["a", "b"])
    first = allocate_constrained(values, {"offer": 0.01}, budget=1.)
    second = allocate_constrained(values, {"offer": 0.01}, budget=1.)
    assert first["assignments"].equals(second["assignments"])
    assert first["assignments"].eq("no_action").all()
    assert first["constraints_satisfied"]

def test_optimizer_rejects_duplicate_customers_and_nonfinite_values():
    duplicated = pd.DataFrame({"offer": [1., 2.]}, index=["x", "x"])
    with pytest.raises(ValueError, match="unique"):
        allocate_constrained(duplicated, {"offer": 0.1}, budget=1.)
    nonfinite = pd.DataFrame({"offer": [np.inf]}, index=["x"])
    with pytest.raises(ValueError, match="finite"):
        allocate_greedy(nonfinite, {"offer": 0.1}, budget=1.)
    with pytest.raises(ValueError, match="no_action"):
        allocate_constrained(pd.DataFrame({"no_action": [1.], "offer": [2.]}, index=["x"]), {"offer": .1}, budget=1.)

def test_equal_cost_exact_shortcut_selects_best_positive_top_k():
    values = pd.DataFrame({"no_action": [0., 0., 0.], "a": [2., 8., -1.], "b": [3., 1., 9.]}, index=list("xyz"))
    result = allocate_constrained(values, {"a": 0.5, "b": 0.5}, budget=1.)
    assert result["solver_method"] == "exact_equal_cost_top_k"
    assert result["total_cost"] == 1.0
    assert result["total_value"] == 17.0
    assert result["assignments"].to_dict() == {"x": "no_action", "y": "a", "z": "b"}

def test_greedy_decimal_costs_use_exactly_affordable_budget():
    values = pd.DataFrame({"no_action": [0.] * 4, "offer": [1.] * 4}, index=list("abcd"))
    allocation = allocate_greedy(values, {"offer": 0.1}, budget=0.3)
    assert allocation.action.eq("offer").sum() == 3
    assert allocation.cost.sum() <= 0.3 + 1e-12
