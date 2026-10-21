import pytest
from pydantic import ValidationError
from nba.llm.evidence_schema import DecisionEvidence
from nba.llm.explanation import explain
from nba.llm.evaluation import explanation_is_faithful

def evidence():
    return DecisionEvidence.model_validate({"decision_id": "d-1", "anonymous_customer_id": "anon",
        "eligible_actions": ["no_action", "mens_email"], "chosen_action": "mens_email",
        "actions": {"no_action": {"eligible": True, "cost": 0., "net_value": 0.},
            "mens_email": {"eligible": True, "incremental_effect": .75, "net_value": .74, "cost": .01, "effect_ci95": [.4, 1.1]}},
        "reason_codes": ["highest_supplied_incremental_net_value"], "model_version": "m1", "policy_version": "p1"})

def test_fallback_keeps_action_evidence_and_interval():
    ev = evidence(); explanation = explain(ev)
    assert explanation.selected_action == ev.chosen_action
    assert "0.75" in " ".join(explanation.evidence)
    assert explanation_is_faithful(ev, explanation)
    assert explanation.explanation_method == "deterministic_template"

def test_contradictory_action_or_missing_reason_is_rejected():
    data = evidence().model_dump(); data["chosen_action"] = "retention_offer"
    with pytest.raises(ValidationError): DecisionEvidence.model_validate(data)
    data = evidence().model_dump(); data["reason_codes"] = []
    with pytest.raises(ValidationError): DecisionEvidence.model_validate(data)
