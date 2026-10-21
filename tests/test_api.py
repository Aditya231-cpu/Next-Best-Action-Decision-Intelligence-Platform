from fastapi.testclient import TestClient
from api.main import create_app
from nba.services.audit import AuditStore
import pytest

def test_api_recommend_audit_and_allocation(tmp_path):
    client = TestClient(create_app(tmp_path / "audit.sqlite"))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/models").json()["models"] == []
    recommendation = client.post("/recommend", json={"customer_ref": "anon-1",
        "incremental_spend": {"mens_email": 2.0, "womens_email": .5},
        "action_costs": {"mens_email": .1, "womens_email": .1},
        "eligible_actions": ["no_action", "mens_email", "womens_email"],
        "model_version": "test-fixture", "policy_version": "test-policy"})
    assert recommendation.status_code == 200
    payload = recommendation.json()
    assert payload["recommended_action"] == "mens_email"
    audit = client.get(f"/audit/{payload['decision_id']}")
    assert audit.status_code == 200
    assert audit.json()["evidence"]["evidence"]["chosen_action"] == "mens_email"
    allocation = client.post("/allocate", json={"customers": [
        {"customer_ref": "a", "incremental_spend": {"mens_email": 2.0}, "eligible_actions": ["no_action", "mens_email"]},
        {"customer_ref": "b", "incremental_spend": {"mens_email": 1.0}, "eligible_actions": ["no_action", "mens_email"]}],
        "action_costs": {"mens_email": .2}, "budget": .2, "capacities": {"mens_email": 1},
        "model_version": "test-fixture", "policy_version": "test-policy"})
    assert allocation.status_code == 200
    assert allocation.json()["action_counts"]["mens_email"] == 1
    assert allocation.json()["constraints_satisfied"]

def test_api_rejects_missing_estimates_and_unapproved_evaluation(tmp_path, monkeypatch):
    client = TestClient(create_app(tmp_path / "audit.sqlite"))
    response = client.post("/recommend", json={"customer_ref": "anon-1", "incremental_spend": {},
        "action_costs": {}, "eligible_actions": ["no_action"], "model_version": "x", "policy_version": "y"})
    assert response.status_code == 422
    response = client.post("/evaluate", json={"evaluation_config_id": "final_test"})
    assert response.status_code == 409
    assert "No evaluation data was opened" in response.json()["detail"]

def test_api_unknown_audit_404(tmp_path):
    client = TestClient(create_app(tmp_path / "audit.sqlite"))
    assert client.get("/audit/not-a-real-id").status_code == 404

def test_audit_records_are_immutable(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite")
    store.record("immutable", {"value": 1})
    with pytest.raises(ValueError): store.record("immutable", {"value": 2})
    assert store.get("immutable")["evidence"]["value"] == 1
    store.close()
