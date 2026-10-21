"""Exact mixed-integer allocation using SciPy HiGHS, when available."""
from __future__ import annotations
import numpy as np
import pandas as pd
from decimal import Decimal
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

def allocate_constrained(values: pd.DataFrame, costs: dict[str, float], *, budget: float,
                         capacities: dict[str, int | None] | None = None,
                         eligibility: pd.DataFrame | None = None) -> dict:
    """Maximize supplied net benefits with one action/customer and cost/capacity limits.

    `values` must already be incremental net benefits (gross effect less action
    cost exactly once). `costs` are also used in the spend constraint, not
    subtracted again from the objective. No action has zero value and is always
    feasible; the budget is a maximum and need not be fully spent.
    """
    capacities = capacities or {}
    if not np.isfinite(budget) or budget < 0: raise ValueError("Budget must be finite and nonnegative")
    if not values.index.is_unique: raise ValueError("Customer index must be unique")
    if "no_action" not in values: values = values.assign(no_action=0.0)
    if not np.isfinite(values.to_numpy(dtype=float)).all(): raise ValueError("Action values must be finite")
    if not np.allclose(values["no_action"].to_numpy(dtype=float), 0.0): raise ValueError("no_action must have zero incremental net benefit")
    customers, actions = list(values.index), list(values.columns)
    n, k = len(customers), len(actions)
    if n == 0: return {"assignments": pd.Series(dtype=str), "status": "optimal", "total_cost": 0., "total_value": 0., "action_counts": {a: 0 for a in actions}}
    if any(a != "no_action" and (a not in costs or not np.isfinite(costs[a]) or costs[a] < 0) for a in actions): raise ValueError("Every action needs a finite nonnegative cost")
    allowed = eligibility.reindex(index=customers, columns=[a for a in actions if a != "no_action"], fill_value=False).fillna(False).astype(bool) if eligibility is not None else pd.DataFrame(True, index=customers, columns=[a for a in actions if a != "no_action"])
    if any(cap is not None and cap < 0 for cap in capacities.values()): raise ValueError("Action capacities must be nonnegative")
    paid_actions = [a for a in actions if a != "no_action"]
    paid_costs = [float(costs[a]) for a in paid_actions]
    no_finite_capacity = all(capacities.get(a) is None for a in paid_actions)
    equal_costs = not paid_costs or max(paid_costs) - min(paid_costs) <= 1e-12
    if equal_costs and no_finite_capacity:
        # With one common per-contact cost and no action caps, this is exactly a
        # top-k problem. Solve it directly instead of sending a large symmetric
        # mixed-integer model to HiGHS.
        cost = paid_costs[0] if paid_costs else 0.0
        max_count = n if cost == 0 else min(n, int(Decimal(str(budget)) // Decimal(str(cost))))
        candidates = []
        for pos, customer in enumerate(customers):
            options = [(float(values.loc[customer, a]), a) for a in paid_actions
                       if bool(allowed.loc[customer, a]) and float(values.loc[customer, a]) > 0]
            if options:
                value, action = max(options, key=lambda item: (item[0], item[1]))
                candidates.append((value, pos, customer, action))
        candidates.sort(key=lambda item: (-item[0], item[1], item[3]))
        assignment = pd.Series("no_action", index=customers, name="action", dtype="object")
        for _, _, customer, action in candidates[:max_count]: assignment.loc[customer] = action
        counts = assignment.value_counts().reindex(actions, fill_value=0).astype(int).to_dict()
        total_cost = float(sum(costs.get(a, 0.0) * count for a, count in counts.items()))
        total_value = float(sum(values.loc[customer, action] for customer, action in assignment.items()))
        return {"assignments": assignment, "status": "optimal", "solver_method": "exact_equal_cost_top_k",
            "total_cost": total_cost, "remaining_budget": float(budget-total_cost), "total_value": total_value,
            "action_counts": counts, "constraints_satisfied": total_cost <= budget + 1e-7
                and all(assignment.loc[customer] == "no_action" or bool(allowed.loc[customer, assignment.loc[customer]]) for customer in customers)}
    c = -values.to_numpy(dtype=float).reshape(-1)
    integrality = np.ones(n * k, dtype=int)
    lower = np.zeros(n * k); upper = np.ones(n * k)
    for i, customer in enumerate(customers):
        for j, action in enumerate(actions):
            if action != "no_action" and not allowed.loc[customer, action]: upper[i * k + j] = 0
    rows = 1 + n + sum(capacities.get(a) is not None for a in actions if a != "no_action")
    A = lil_matrix((rows, n * k), dtype=float); lb = np.full(rows, -np.inf); ub = np.full(rows, np.inf)
    for i in range(n):
        A[1 + i, i*k:(i+1)*k] = 1.; ub[1 + i] = 1.
    for j, action in enumerate(actions):
        A[0, j::k] = float(costs.get(action, 0.0))
    ub[0] = budget
    row = n + 1
    for j, action in enumerate(actions):
        cap = capacities.get(action)
        if action != "no_action" and cap is not None:
            A[row, j::k] = 1.; ub[row] = int(cap); row += 1
    result = milp(c=c, integrality=integrality, bounds=Bounds(lower, upper), constraints=LinearConstraint(A.tocsr(), lb, ub),
                  options={"time_limit": 30.0, "mip_rel_gap": 1e-8})
    if result.x is None:
        return {"assignments": pd.Series("no_action", index=customers), "status": str(result.message), "total_cost": 0., "total_value": 0., "action_counts": {a: 0 for a in actions}}
    chosen_idx = result.x.reshape(n, k).argmax(axis=1)
    assignment = pd.Series([actions[j] for j in chosen_idx], index=customers, name="action")
    counts = assignment.value_counts().reindex(actions, fill_value=0).astype(int).to_dict()
    total_cost = float(sum(costs.get(a, 0.0) * count for a, count in counts.items()))
    total_value = float(sum(values.loc[customer, action] for customer, action in assignment.items()))
    one_action = bool((assignment.isin(actions)).all()) and bool(assignment.map(lambda a: int(a != "no_action")).max() <= 1)
    eligible_ok = all(assignment.loc[customer] == "no_action" or bool(allowed.loc[customer, assignment.loc[customer]]) for customer in customers)
    return {"assignments": assignment, "status": "optimal" if result.success else f"feasible: {result.message}", "solver_method": "scipy_highs_milp",
        "total_cost": total_cost, "remaining_budget": float(budget - total_cost), "total_value": total_value,
        "action_counts": counts, "constraints_satisfied": total_cost <= budget + 1e-7 and one_action and eligible_ok
            and all(capacities.get(a) is None or counts[a] <= capacities[a] for a in actions if a != "no_action")}
