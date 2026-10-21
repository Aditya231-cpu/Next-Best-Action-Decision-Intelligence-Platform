import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_policies import (
    _allocate_by_benefit_cost,
    _policy_contributions,
    _random_budget_policy,
)


def test_ips_gross_net_and_incremental_cost_convention_subtracts_cost_once():
    observed = np.array(["mens_email", "no_action", "womens_email"])
    policy = np.array(["mens_email", "no_action", "no_action"])
    result = _policy_contributions(policy, observed, [3.0, 1.0, 4.0], {"mens_email": 0.5, "womens_email": 0.25})
    assert np.allclose(result["gross"], [9.0, 3.0, 0.0])
    assert np.allclose(result["cost"], [0.5, 0.0, 0.0])
    assert np.allclose(result["net"], [8.5, 3.0, 0.0])


def test_policy_label_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Unknown policy or observed action"):
        _policy_contributions(["Mens E-Mail"], ["mens_email"], [1.0], {"mens_email": 0.01})


def test_random_targeting_is_reproducible_one_action_per_customer_and_budgeted():
    index = pd.Index(["a", "b", "c", "d"])
    actions = ["mens_email", "womens_email"]
    eligibility = pd.DataFrame(True, index=index, columns=actions)
    first = _random_budget_policy(index, actions, {a: 1.0 for a in actions}, 2.0, eligibility, seed=9)
    second = _random_budget_policy(index, actions, {a: 1.0 for a in actions}, 2.0, eligibility, seed=9)
    assert first.equals(second)
    assert sum(first.isin(actions)) <= 2
    assert sum(first.map({"mens_email": 1.0, "womens_email": 1.0}).fillna(0)) <= 2.0
    assert set(first) <= {"no_action", *actions}


def test_net_benefit_cost_ratio_allocator_respects_budget_and_leaves_unprofitable_unassigned():
    values = pd.DataFrame({"no_action": [0.0, 0.0, 0.0], "offer": [8.0, 6.0, -1.0]}, index=["a", "b", "c"])
    eligibility = pd.DataFrame({"offer": True}, index=values.index)
    assigned = _allocate_by_benefit_cost(values, {"offer": 3.0}, 3.0, eligibility)
    assert assigned.to_dict() == {"a": "offer", "b": "no_action", "c": "no_action"}
