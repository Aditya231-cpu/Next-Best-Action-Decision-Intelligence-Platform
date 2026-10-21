"""Decision, allocation, evaluation gate, and evidence audit routes."""
from __future__ import annotations
import os
import uuid
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from nba.llm.evidence_schema import ActionEvidence, DecisionEvidence
from nba.llm.explanation import explain
from nba.optimization.constrained_solver import allocate_constrained
from nba.value.economics import net_value_from_spend
from api.schemas import AllocationRequest, EvaluationRequest, RecommendationRequest

router = APIRouter()

@router.get("/health")
def health(): return {"status": "ok", "empirical_benchmark": "pending_real_data"}

@router.get("/models")
def models(): return {"models": [], "status": "no_trained_model_registered", "note": "Load a validated artifact before making model-derived estimates."}

@router.post("/recommend")
def recommend(payload: RecommendationRequest, request: Request):
    decision_id = str(uuid.uuid4())
    effects, eligible = payload.incremental_spend, set(payload.eligible_actions)
    values = {"no_action": 0.0}
    for action, effect in effects.items():
        if action in eligible: values[action] = float(net_value_from_spend(effect, payload.action_costs[action]))
    chosen = max(values, key=lambda action: (values[action], action == "no_action"))
    action_evidence = {"no_action": ActionEvidence(eligible=True, incremental_effect=0.0,
        net_value=0.0, cost=0.0, evidence_type="control_baseline")}
    for action, effect in effects.items():
        if action not in eligible: continue
        action_evidence[action] = ActionEvidence(eligible=True, incremental_effect=effect,
            net_value=values[action], cost=payload.action_costs[action],
            effect_ci95=payload.effect_intervals.get(action),
            evidence_type=payload.evidence_types.get(action, "external_estimate"))
    reason = "no_positive_incremental_net_value" if chosen == "no_action" else "highest_supplied_incremental_net_value"
    limitations = list(payload.limitations)
    limitations.append("Treatment-effect estimates were supplied to the service; this endpoint does not estimate causal effects.")
    if any(action_evidence[a].evidence_type == "simulated" for a in action_evidence):
        limitations.append("At least one action estimate is simulated and is not real-world intervention evidence.")
    evidence = DecisionEvidence(decision_id=decision_id, anonymous_customer_id=payload.customer_ref,
        eligible_actions=list(payload.eligible_actions), actions=action_evidence, chosen_action=chosen,
        reason_codes=[reason], model_version=payload.model_version, policy_version=payload.policy_version,
        limitations=limitations)
    explanation = explain(evidence)
    warnings = ["Effects are externally supplied and require validation against the stated model version."]
    if any(a not in {"mens_email", "womens_email"} for a in effects):
        warnings.append("Non-email actions are outside the Hillstrom randomized evidence; treat results as simulated or externally identified.")
    request.app.state.audit.record(decision_id, {"evidence": evidence.model_dump(),
        "explanation": explanation.model_dump(), "net_values": values})
    return {"decision_id": decision_id, "customer_ref": payload.customer_ref,
        "available_actions": list(values), "estimated_effects": effects,
        "estimated_incremental_values": values, "recommended_action": chosen,
        "reason_codes": [reason], "model_version": payload.model_version,
        "policy_version": payload.policy_version, "warnings": warnings,
        "assumptions": payload.limitations, "explanation": explanation.model_dump()}

@router.post("/allocate")
def allocate(payload: AllocationRequest, request: Request):
    refs = [c.customer_ref for c in payload.customers]
    all_actions = sorted(set(payload.action_costs) | {a for c in payload.customers for a in c.incremental_spend})
    effects, eligibility = pd.DataFrame(index=refs), pd.DataFrame(index=refs)
    for action in all_actions:
        effects[action] = [c.incremental_spend.get(action, 0.0) for c in payload.customers]
        eligibility[action] = [action in c.eligible_actions and action in c.incremental_spend for c in payload.customers]
    values = pd.DataFrame({"no_action": np.zeros(len(refs))}, index=refs)
    for action in all_actions:
        if action not in payload.action_costs: raise HTTPException(422, f"Missing cost for action {action}")
        values[action] = net_value_from_spend(effects[action].to_numpy(), payload.action_costs[action])
    try:
        result = allocate_constrained(values, payload.action_costs, budget=payload.budget,
            capacities=payload.capacities, eligibility=eligibility)
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc
    assignments = result["assignments"].to_dict()
    decision_ids = {}
    binding = []
    if abs(result["total_cost"] - payload.budget) <= 1e-7: binding.append("total_budget")
    for action, cap in payload.capacities.items():
        if cap is not None and result["action_counts"].get(action, 0) == cap: binding.append(f"capacity:{action}")
    for ref, action in assignments.items():
        decision_id = str(uuid.uuid4())
        decision_ids[ref] = decision_id
        request.app.state.audit.record(decision_id, {"customer_ref": ref, "selected_action": action,
            "action_values": values.loc[ref].to_dict(), "binding_constraints": binding,
            "solver_status": result["status"], "model_version": payload.model_version,
            "policy_version": payload.policy_version, "allocation_budget": payload.budget,
            "selected_net_value": float(values.loc[ref, action])})
    return {"assignments": assignments, "decision_ids": decision_ids, "total_cost": result["total_cost"],
        "remaining_budget": result.get("remaining_budget", payload.budget), "action_counts": result["action_counts"],
        "solver_status": result["status"], "constraints_satisfied": result.get("constraints_satisfied", False),
        "total_expected_net_value": result["total_value"], "model_version": payload.model_version,
        "policy_version": payload.policy_version,
        "warnings": ["Action effects are supplied by an upstream estimator; this API does not estimate causal effects.",
            "Actions outside the two randomized email treatments require separate causal evidence or explicit simulation labeling."]}

@router.post("/evaluate")
def evaluate(payload: EvaluationRequest):
    approved = {item.strip() for item in os.getenv("NBA_APPROVED_EVALUATION_CONFIGS", "").split(",") if item.strip()}
    if payload.evaluation_config_id not in approved:
        raise HTTPException(409, "Evaluation configuration is not explicitly approved. No evaluation data was opened or model was trained.")
    raise HTTPException(501, "Approved evaluation configurations are not implemented yet.")

@router.get("/audit/{decision_id}")
def audit(decision_id: str, request: Request):
    result = request.app.state.audit.get(decision_id)
    if result is None: raise HTTPException(404, "Decision ID was not found")
    return result
