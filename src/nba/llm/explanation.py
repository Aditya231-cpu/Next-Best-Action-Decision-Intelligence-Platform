"""Deterministic evidence-to-text fallback; no claim-generation model is required."""
from __future__ import annotations
from nba.llm.evidence_schema import DecisionEvidence, Explanation

def explain(evidence: DecisionEvidence | dict) -> Explanation:
    """Describe an already-selected action without changing or recomputing it."""
    ev = evidence if isinstance(evidence, DecisionEvidence) else DecisionEvidence.model_validate(evidence)
    chosen = ev.actions[ev.chosen_action]
    evidence_lines = [f"Reason code: {code}." for code in ev.reason_codes]
    if chosen.incremental_effect is not None:
        evidence_lines.append(f"Estimated incremental effect: {chosen.incremental_effect:.6g}.")
    if chosen.net_value is not None:
        evidence_lines.append(f"Estimated net value: {chosen.net_value:.6g}; estimated action cost: {chosen.cost:.6g}.")
    if chosen.effect_ci95 is not None:
        evidence_lines.append(f"Supplied 95% effect interval: [{chosen.effect_ci95[0]:.6g}, {chosen.effect_ci95[1]:.6g}].")
        uncertainty = "The supplied interval describes estimation uncertainty; individual treatment effects are not observed."
    else:
        uncertainty = "No valid uncertainty interval was supplied for this recommendation."
    return Explanation(selected_action=ev.chosen_action,
        main_reason=f"The decision policy selected {ev.chosen_action} from the eligible actions using supplied net-value evidence.",
        evidence=evidence_lines, uncertainty=uncertainty, binding_constraints=ev.binding_constraints,
        limitations=ev.limitations)
