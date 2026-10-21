"""Strict schemas separating decision evidence from explanatory prose."""
from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field, model_validator

class ActionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    eligible: bool
    incremental_effect: float | None = None
    net_value: float | None = None
    cost: float = Field(ge=0)
    effect_ci95: tuple[float, float] | None = None
    evidence_type: str = "experimental"
    @model_validator(mode="after")
    def interval_is_ordered(self):
        if self.effect_ci95 is not None and self.effect_ci95[0] > self.effect_ci95[1]:
            raise ValueError("Confidence interval lower bound must not exceed upper bound")
        return self

class DecisionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_id: str
    anonymous_customer_id: str
    eligible_actions: list[str]
    actions: dict[str, ActionEvidence]
    chosen_action: str
    reason_codes: list[str]
    binding_constraints: list[str] = Field(default_factory=list)
    model_version: str
    policy_version: str
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def decision_is_supported(self):
        if self.chosen_action not in self.eligible_actions:
            raise ValueError("Chosen action must be listed as eligible")
        if self.chosen_action not in self.actions or not self.actions[self.chosen_action].eligible:
            raise ValueError("Chosen action must have eligible structured evidence")
        if not self.reason_codes: raise ValueError("At least one reason code is required")
        return self

class Explanation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected_action: str
    main_reason: str
    evidence: list[str]
    uncertainty: str
    binding_constraints: list[str]
    limitations: list[str]
    explanation_method: str = "deterministic_template"
