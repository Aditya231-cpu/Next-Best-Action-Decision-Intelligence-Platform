"""Evaluate locked policies once on the final randomized test split using IPS."""
from __future__ import annotations
import argparse, json
from decimal import Decimal
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from nba.causal.baselines import ResponsePropensity
from nba.data.split import split_customers
from nba.evaluation.off_policy import paired_bootstrap_difference
from nba.evaluation.uplift_metrics import auuc, qini_coefficient, qini_curve, ranking_bootstrap, uplift_at_fraction
from nba.optimization.constrained_solver import allocate_constrained
from nba.optimization.greedy import allocate_greedy
from nba.value.economics import net_value_from_spend

FEATURES = ["recency", "history", "history_segment", "mens", "womens", "newbie", "zip_code", "channel"]
ARMS = {"mens_email": "Mens E-Mail", "womens_email": "Womens E-Mail"}
LABELS = {"No E-Mail": "no_action", "Mens E-Mail": "mens_email", "Womens E-Mail": "womens_email"}
PROB = {"No E-Mail": 1/3, "Mens E-Mail": 1/3, "Womens E-Mail": 1/3}
ACTION_PROB = {"no_action": 1/3, "mens_email": 1/3, "womens_email": 1/3}

def _policy_contributions(policy, observed, outcome, costs):
    """Return rowwise IPS gross spend and net value; subtract action cost once."""
    policy, observed, outcome = np.asarray(policy), np.asarray(observed), np.asarray(outcome, float)
    if not (len(policy) == len(observed) == len(outcome)) or len(policy) == 0:
        raise ValueError("Policy, observed actions, and outcomes must have equal nonzero length")
    if set(policy) - set(ACTION_PROB) or set(observed) - set(LABELS.values()):
        raise ValueError("Unknown policy or observed action label")
    p = np.array([PROB[next(label for label, action in LABELS.items() if action == a)] for a in observed])
    gross = (policy == observed) * outcome / p
    action_cost = np.array([costs.get(a, 0.) for a in policy], dtype=float)
    return {"gross": gross, "cost": action_cost, "net": gross - action_cost}


def _exact_action_cost(actions, costs):
    counts = pd.Series(actions).value_counts().to_dict()
    total = sum((Decimal(str(costs.get(action, 0.0))) * int(count) for action, count in counts.items()), Decimal("0"))
    return float(total)


def _random_budget_policy(index, actions, costs, budget, eligibility, *, seed):
    """Uniform random ordering of eligible customer/action offers under a max budget."""
    rng = np.random.default_rng(seed)
    candidates = [(customer, action) for customer in index for action in actions
                  if bool(eligibility.loc[customer, action])]
    if candidates:
        candidates = [candidates[i] for i in rng.permutation(len(candidates))]
    assignment = pd.Series("no_action", index=index, dtype="object")
    total = Decimal("0")
    budget_exact = Decimal(str(budget))
    for customer, action in candidates:
        cost = Decimal(str(float(costs[action])))
        if assignment.loc[customer] == "no_action" and total + cost <= budget_exact:
            assignment.loc[customer] = action
            total += cost
    return assignment


def _allocate_by_benefit_cost(values, costs, budget, eligibility):
    """Rank each customer's best positive net benefit by benefit/cost ratio."""
    actions = [a for a in values.columns if a != "no_action"]
    candidates = []
    for pos, customer in enumerate(values.index):
        options = [(float(values.loc[customer, a]), a) for a in actions
                   if bool(eligibility.loc[customer, a]) and float(values.loc[customer, a]) > 0]
        if options:
            benefit, action = max(options, key=lambda item: (item[0], item[1]))
            cost = float(costs[action])
            ratio = benefit / cost if cost > 0 else float("inf")
            candidates.append((ratio, benefit, pos, customer, action, cost))
    candidates.sort(key=lambda row: (-row[0], -row[1], row[2], row[4]))
    assignment = pd.Series("no_action", index=values.index, dtype="object")
    remaining = Decimal(str(budget))
    for _, _, _, customer, action, cost in candidates:
        cost_exact = Decimal(str(cost))
        if cost_exact <= remaining:
            assignment.loc[customer] = action
            remaining -= cost_exact
    return assignment


def _ranking_diagnostics(scores, treatment, outcome, *, seed, bootstrap=False):
    scores, treatment, outcome = np.asarray(scores, float), np.asarray(treatment, bool), np.asarray(outcome, float)
    curve = qini_curve(scores, treatment, outcome)
    fractions = (.01, .05, .10, .20, .50, 1.0)
    curve_x = np.r_[0., curve.fraction_targeted.to_numpy()]
    curve_y = np.r_[0., curve.qini.to_numpy()]
    display_x = np.linspace(0., 1., 101)
    result = {"qini_coefficient": qini_coefficient(scores, treatment, outcome),
        "auuc_outcome_times_fraction": auuc(scores, treatment, outcome),
        "uplift_at_fraction": {str(k): v for k, v in uplift_at_fraction(scores, treatment, outcome, fractions).items()},
        "qini_curve": [{"fraction_targeted": float(x), "qini": float(y)} for x, y in zip(display_x, np.interp(display_x, curve_x, curve_y))],
        "ranking_direction": "descending; higher score means larger treatment effect",
        "treatment_encoding": {"treated": True, "control": False},
        "tie_diagnostics": {"unique_scores": int(np.unique(scores).size),
            "largest_tie_fraction": float(pd.Series(scores).value_counts(normalize=True).max()),
            "ties_handled_as": "score groups; no arbitrary within-tie ranking"}}
    if bootstrap:
        result["bootstrap_95_ci"] = ranking_bootstrap(scores, treatment, outcome, draws=200, seed=seed, fractions=(.05, .10, .20))
    return result

def evaluate(data_path: Path, artifact_dir: Path, out_dir: Path, *, seed: int = 20261018,
             costs: dict[str, float] | None = None, budget: float = 25.) -> dict:
    if not data_path.is_file(): raise FileNotFoundError(f"Missing real processed data: {data_path.resolve()}")
    if costs is None: costs = {"mens_email": .01, "womens_email": .01}
    frame = pd.read_parquet(data_path)
    train, validation, test = split_customers(frame, seed=seed)
    if budget < 0 or any(not np.isfinite(v) or v < 0 for v in costs.values()):
        raise ValueError("Budget and action costs must be finite and nonnegative")
    features = [c for c in FEATURES if c in frame]
    X_train, X_val, X = train[features], validation[features], test[features]
    tau, responses = {}, {}
    ranking = {}
    training_report_path = artifact_dir / "training_report.json"
    training_report = json.loads(training_report_path.read_text(encoding="utf-8")) if training_report_path.is_file() else {}
    for key, arm in ARMS.items():
        model_path = artifact_dir / f"{key}_spend.joblib"
        if not model_path.is_file(): raise FileNotFoundError(f"Missing trained model artifact {model_path}; run scripts/train_models.py first")
        model = joblib.load(model_path)
        tau[key] = model.predict_effect(X)
        tau_val = model.predict_effect(X_val)
        train_pair = train.treatment.isin(["No E-Mail", arm])
        train_arm = train.loc[train_pair]
        train_effect = float(train_arm.loc[train_arm.treatment == arm, "spend"].mean() - train_arm.loc[train_arm.treatment == "No E-Mail", "spend"].mean())
        segment_effect = {}
        for segment, group in train_arm.groupby("history_segment", dropna=False):
            arm_mean = group.loc[group.treatment == arm, "spend"].mean()
            control_mean = group.loc[group.treatment == "No E-Mail", "spend"].mean()
            if pd.notna(arm_mean) and pd.notna(control_mean): segment_effect[str(segment)] = float(arm_mean - control_mean)
        val_pair = validation.treatment.isin(["No E-Mail", arm]).to_numpy()
        test_pair = test.treatment.isin(["No E-Mail", arm]).to_numpy()
        val_t = (validation.loc[val_pair, "treatment"].to_numpy() == arm)
        test_t = (test.loc[test_pair, "treatment"].to_numpy() == arm)
        segment_val = validation.loc[val_pair, "history_segment"].astype(str).map(segment_effect).fillna(train_effect).to_numpy(float)
        segment_test = test.loc[test_pair, "history_segment"].astype(str).map(segment_effect).fillna(train_effect).to_numpy(float)
        ranking[key] = {"selected_model": training_report.get("models", {}).get(key, {}).get("selected"),
            "model_selection_partition": "validation", "validation": {
                "learned_model": _ranking_diagnostics(tau_val[val_pair], val_t, validation.loc[val_pair, "spend"].to_numpy(float), seed=seed + 1),
                "constant_effect_baseline": _ranking_diagnostics(np.full(val_t.size, train_effect), val_t, validation.loc[val_pair, "spend"].to_numpy(float), seed=seed + 2),
                "history_segment_baseline": _ranking_diagnostics(segment_val, val_t, validation.loc[val_pair, "spend"].to_numpy(float), seed=seed + 3)},
            "test": {"learned_model": _ranking_diagnostics(tau[key][test_pair], test_t, test.loc[test_pair, "spend"].to_numpy(float), seed=seed + 4, bootstrap=True),
                "constant_effect_baseline": _ranking_diagnostics(np.full(test_t.size, train_effect), test_t, test.loc[test_pair, "spend"].to_numpy(float), seed=seed + 5),
                "history_segment_baseline": _ranking_diagnostics(segment_test, test_t, test.loc[test_pair, "spend"].to_numpy(float), seed=seed + 6)}}
        y_train = train.visit.to_numpy()
        ctl = train.treatment == "No E-Mail"
        arm_mask = train.treatment == arm
        control_model = ResponsePropensity(seed).fit(X_train.loc[ctl], np.ones(int(ctl.sum()), dtype=bool), y_train[ctl])
        arm_model = ResponsePropensity(seed + 1).fit(X_train.loc[arm_mask], np.ones(int(arm_mask.sum()), dtype=bool), y_train[arm_mask])
        responses[f"control_{key}"] = control_model.predict(X)
        responses[key] = arm_model.predict(X)
    net = {a: np.asarray(net_value_from_spend(effect, costs[a])) for a, effect in tau.items()}
    observed_raw = test.treatment.to_numpy()
    observed = np.array([LABELS[a] for a in observed_raw])
    index = test.customer_id
    action_names = list(ARMS)
    elig = pd.DataFrame(True, index=index, columns=action_names)
    budget_caps = {a: None for a in action_names}
    random_action = _random_budget_policy(index, action_names, costs, budget, elig, seed=seed + 99)
    response_values = pd.DataFrame({"no_action": np.zeros(len(test)), **{a: responses[a] for a in action_names}}, index=index)
    response_action = allocate_greedy(response_values, costs, budget=budget, capacities=budget_caps, eligibility=elig).action
    rule_choice = np.where(test.mens.to_numpy() == 1, "mens_email",
        np.where(test.womens.to_numpy() == 1, "womens_email", "no_action"))
    rng = np.random.default_rng(seed + 101)
    rule_values = pd.DataFrame(0.0, index=index, columns=["no_action", *action_names])
    rule_elig = pd.DataFrame(False, index=index, columns=action_names)
    for i, (customer, action) in enumerate(zip(index, rule_choice)):
        if action != "no_action":
            rule_values.loc[customer, action] = 1.0 + float(rng.random()) * 1e-9
            rule_elig.loc[customer, action] = True
    rule_action = allocate_greedy(rule_values, costs, budget=budget, capacities=budget_caps, eligibility=rule_elig).action
    uplift_values = pd.DataFrame({"no_action": np.zeros(len(test)), **{a: tau[a] for a in action_names}}, index=index)
    uplift_action = allocate_greedy(uplift_values, costs, budget=budget, capacities=budget_caps, eligibility=elig).action
    net_values = pd.DataFrame({"no_action": np.zeros(len(test)), **{a: net[a] for a in action_names}}, index=index)
    greedy_frame = allocate_greedy(net_values, costs, budget=budget, capacities=budget_caps, eligibility=elig)
    constrained_result = allocate_constrained(net_values, costs, budget=budget, capacities=budget_caps, eligibility=elig)
    net_benefit_action = _allocate_by_benefit_cost(net_values, costs, budget, elig)
    policies = {"no_email": np.repeat("no_action", len(test)), "budget_matched_random": random_action.to_numpy(),
        "response_targeting_budgeted": response_action.to_numpy(), "business_rule_budgeted": rule_action.to_numpy(),
        "uplift_targeting_budgeted": uplift_action.to_numpy(), "incremental_net_benefit_budgeted": net_benefit_action.to_numpy(),
        "greedy_budgeted": greedy_frame.action.to_numpy(), "constrained_optimized": constrained_result["assignments"].reindex(index).to_numpy()}
    out_dir.mkdir(parents=True, exist_ok=True)
    records, contributions = [], {}
    no_email_contrib = _policy_contributions(policies["no_email"], observed, test.spend.to_numpy(), costs)["net"]
    for name, actions in policies.items():
        components = _policy_contributions(actions, observed, test.spend.to_numpy(), costs)
        gross, net_contrib = components["gross"], components["net"]
        contributions[name] = net_contrib
        incremental = net_contrib - no_email_contrib
        n = len(actions)
        se_gross = float(gross.std(ddof=1) / np.sqrt(n)); se_net = float(net_contrib.std(ddof=1) / np.sqrt(n))
        se_inc = float(incremental.std(ddof=1) / np.sqrt(n))
        counts = pd.Series(actions).value_counts().reindex(["no_action", *action_names], fill_value=0).astype(int).to_dict()
        total_cost = _exact_action_cost(actions, costs)
        paired_no_email = paired_bootstrap_difference(net_contrib, no_email_contrib, draws=1000, seed=seed + 201)
        records.append({"policy": name, "gross_policy_value_per_customer": float(gross.mean()),
            "gross_se": se_gross, "gross_ci95_normal": [float(gross.mean()-1.96*se_gross), float(gross.mean()+1.96*se_gross)],
            "net_policy_value_per_customer": float(net_contrib.mean()), "net_se": se_net,
            "net_ci95_normal": [float(net_contrib.mean()-1.96*se_net), float(net_contrib.mean()+1.96*se_net)],
            "incremental_net_value_vs_no_email_per_customer": float(incremental.mean()), "incremental_net_se": se_inc,
            "incremental_net_ci95_normal": [float(incremental.mean()-1.96*se_inc), float(incremental.mean()+1.96*se_inc)],
            "paired_bootstrap_vs_no_email": paired_no_email,
            "ips_net_value_per_customer": float(net_contrib.mean()), "standard_error": se_net,
            "ci95_normal_low": float(net_contrib.mean()-1.96*se_net), "ci95_normal_high": float(net_contrib.mean()+1.96*se_net),
            "action_counts": counts, "action_costs_usd_per_send": costs, "total_action_cost_usd": total_cost,
            "budget_usd": budget, "budget_remaining_usd": float(budget-total_cost),
            "within_budget": total_cost <= budget + 1e-8})
    baseline = contributions["budget_matched_random"]
    for record in records:
        record["paired_bootstrap_vs_budget_matched_random"] = {"difference": 0., "ci_low": 0., "ci_high": 0., "confidence": .95, "draws": 1000} if record["policy"] == "budget_matched_random" else paired_bootstrap_difference(contributions[record["policy"]], baseline, draws=1000, seed=seed + 202)
    cost_sensitivity = []
    for multiplier in (0., .5, 1., 2., 5.):
        scenario_costs = {a: c * multiplier for a, c in costs.items()}
        scenario_net = {a: tau[a] - scenario_costs[a] for a in ARMS}
        scenario_frame = pd.DataFrame({"no_action": np.zeros(len(test)), **scenario_net}, index=index)
        scenario_actions = allocate_constrained(scenario_frame, scenario_costs, budget=budget,
            capacities=budget_caps, eligibility=elig)["assignments"].reindex(index).to_numpy()
        component = _policy_contributions(scenario_actions, observed, test.spend.to_numpy(), scenario_costs)
        incremental = component["net"] - no_email_contrib
        se = float(incremental.std(ddof=1) / np.sqrt(len(test)))
        cost_sensitivity.append({"cost_multiplier": multiplier, "costs_usd_per_send": scenario_costs,
            "budget_usd_maximum": budget, "gross_policy_value_per_customer": float(component["gross"].mean()),
            "net_policy_value_per_customer": float(component["net"].mean()),
            "incremental_net_value_vs_no_email_per_customer": float(incremental.mean()),
            "incremental_net_ci95_normal": [float(incremental.mean()-1.96*se), float(incremental.mean()+1.96*se)],
            "action_counts": pd.Series(scenario_actions).value_counts().to_dict(),
            "total_action_cost_usd": _exact_action_cost(scenario_actions, scenario_costs),
            "budget_remaining_usd": float(budget-_exact_action_cost(scenario_actions, scenario_costs))})
    budget_sensitivity = []
    for multiplier in (.25, .5, 1., 2.):
        scenario_budget = budget * multiplier
        actions = allocate_constrained(net_values, costs, budget=scenario_budget,
            capacities=budget_caps, eligibility=elig)["assignments"].reindex(index).to_numpy()
        component = _policy_contributions(actions, observed, test.spend.to_numpy(), costs)
        incremental = component["net"] - no_email_contrib
        se = float(incremental.std(ddof=1) / np.sqrt(len(test)))
        budget_sensitivity.append({"budget_usd_maximum": scenario_budget,
            "gross_policy_value_per_customer": float(component["gross"].mean()),
            "net_policy_value_per_customer": float(component["net"].mean()),
            "incremental_net_value_vs_no_email_per_customer": float(incremental.mean()),
            "incremental_net_ci95_normal": [float(incremental.mean()-1.96*se), float(incremental.mean()+1.96*se)],
            "action_counts": pd.Series(actions).value_counts().to_dict(),
            "total_action_cost_usd": _exact_action_cost(actions, costs),
            "budget_remaining_usd": float(scenario_budget-_exact_action_cost(actions, costs))})
    result = {"schema_version": 2, "dataset": "Hillstrom randomized email experiment", "outcome": "post-treatment spend, USD/customer",
        "test_rows": len(test), "test_split_seed": seed, "assignment_probabilities": PROB,
        "split_provenance": {"train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
            "model_selection_split": "validation", "final_test_used_for_model_selection_or_fitting": False,
            "policy_selection": "validation-selected serialized models; benchmark definitions fixed before final test scoring"},
        "value_definitions": {"gross_policy_value": "IPS estimate of expected post-treatment spend under policy",
            "net_policy_value": "gross policy value minus assigned action costs exactly once",
            "incremental_net_value": "net policy value minus the no-email policy net value",
            "ips_reward": "1[action == observed randomized treatment] * spend / propensity; subtract policy-assigned email cost once",
            "propensity": "documented equal randomization probabilities, 1/3 for each of the three arms"},
        "action_costs_assumed_usd_per_send": costs, "budget_usd_maximum_not_required_to_spend": budget,
        "eligible_customers": len(test), "eligible_actions": action_names, "action_capacities": budget_caps,
        "optimizer_provenance": {"constrained_status": constrained_result["status"],
            "constrained_method": constrained_result.get("solver_method", "scipy_highs_milp"),
            "objective": "maximize sum of per-customer incremental net benefit under max spend, one-action, eligibility, and capacity constraints",
            "greedy_method": "best positive net benefit per customer, sorted descending; exact only under special cost/capacity conditions",
            "incremental_net_benefit_method": "best positive net-benefit-to-cost ratio per customer, sorted descending"},
        "ranking_diagnostics": ranking,
        "model_implied_cost_sensitivity_evaluated_by_ips": cost_sensitivity,
        "constrained_policy_budget_sensitivity_evaluated_by_ips": budget_sensitivity,
        "results": records,
        "limitations": ["IPS is unbiased for known randomized propensities when positivity and consistency hold.",
            "Ranking metrics use the experimental spend outcome; weak or noisy Qini is not proof a model is invalid.",
            "Costs are explicit assumptions, not measured source data.",
            "Historical offline policy value is not a live deployment guarantee.",
            "The test outputs from the prior evaluator were already inspected; code changes here correct estimands and budget matching, not model tuning. Treat results as exploratory and use a newly collected holdout for confirmatory claims."]}
    (out_dir / "policy_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in record.items()}
                  for record in records]).to_csv(out_dir / "policy_comparison.csv", index=False)
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/hillstrom.parquet"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("reports/generated"))
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--budget", type=float, default=25.)
    parser.add_argument("--mens-cost", type=float, default=.01)
    parser.add_argument("--womens-cost", type=float, default=.01)
    args = parser.parse_args()
    result = evaluate(args.data, args.artifacts, args.out, seed=args.seed, budget=args.budget,
        costs={"mens_email": args.mens_cost, "womens_email": args.womens_cost})
    print(f"Evaluated {len(result['results'])} locked policies on {result['test_rows']:,} final test customers.")

if __name__ == "__main__": main()
