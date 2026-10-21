"""Structural checks for explanation integrity."""
from nba.llm.evidence_schema import DecisionEvidence, Explanation

def explanation_is_faithful(evidence: DecisionEvidence, explanation: Explanation) -> bool:
    """Return whether selected action and supplied numerical evidence were preserved."""
    if explanation.selected_action != evidence.chosen_action: return False
    text = " ".join(explanation.evidence)
    for key, record in evidence.actions.items():
        for number in (record.incremental_effect, record.net_value):
            if number is not None and key == evidence.chosen_action and f"{number:.6g}" not in text: return False
    return True
