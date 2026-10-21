"""HTTP request/response contracts."""
from __future__ import annotations
import math
from pydantic import BaseModel, ConfigDict, Field, model_validator

class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_ref: str = Field(min_length=1, max_length=120)
    incremental_spend: dict[str, float]
    action_costs: dict[str, float]
    eligible_actions: list[str]
    effect_intervals: dict[str, tuple[float, float]] = Field(default_factory=dict)
    evidence_types: dict[str, str] = Field(default_factory=dict)
    model_version: str
    policy_version: str
    limitations: list[str] = Field(default_factory=list)
    @model_validator(mode="after")
    def has_valid_inputs(self):
        if not self.incremental_spend: raise ValueError("At least one externally estimated treatment effect is required")
        if "no_action" in self.incremental_spend: raise ValueError("no_action is the zero-effect baseline and must not be supplied as an effect")
        if "no_action" not in self.eligible_actions: raise ValueError("no_action must be eligible")
        if set(self.incremental_spend) - set(self.action_costs): raise ValueError("Every effect needs an explicit action cost")
        if set(self.incremental_spend) - set(self.eligible_actions): raise ValueError("Effects may be supplied only for eligible actions")
        if any(not math.isfinite(v) for v in self.incremental_spend.values()): raise ValueError("Effects must be finite numbers")
        if any(not math.isfinite(v) or v < 0 for v in self.action_costs.values()): raise ValueError("Action costs must be finite and nonnegative")
        if any(lo > hi for lo, hi in self.effect_intervals.values()): raise ValueError("Each interval lower bound must be no greater than its upper bound")
        return self

class AllocationCustomer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_ref: str = Field(min_length=1, max_length=120)
    incremental_spend: dict[str, float]
    eligible_actions: list[str]

class AllocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[AllocationCustomer] = Field(min_length=1, max_length=10000)
    action_costs: dict[str, float]
    budget: float = Field(ge=0)
    capacities: dict[str, int | None] = Field(default_factory=dict)
    model_version: str
    policy_version: str
    time_limit_seconds: float = Field(default=30, gt=0, le=300)
    @model_validator(mode="after")
    def unique_refs_and_costs(self):
        refs = [c.customer_ref for c in self.customers]
        if len(refs) != len(set(refs)): raise ValueError("Customer references must be unique")
        for customer in self.customers:
            missing = set(customer.incremental_spend) - set(self.action_costs)
            if missing: raise ValueError(f"Missing action costs for: {sorted(missing)}")
            if any(not math.isfinite(v) for v in customer.incremental_spend.values()): raise ValueError("Effects must be finite numbers")
        if any(not math.isfinite(v) or v < 0 for v in self.action_costs.values()): raise ValueError("Action costs must be finite and nonnegative")
        if any(cap is not None and cap < 0 for cap in self.capacities.values()): raise ValueError("Capacities must be nonnegative")
        if "no_action" in self.action_costs: raise ValueError("no_action is the zero-cost baseline and must not be listed as an intervention")
        return self

class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluation_config_id: str = Field(min_length=1)

