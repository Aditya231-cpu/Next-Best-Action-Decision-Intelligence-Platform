import numpy as np
import pandas as pd
from nba.monitoring.drift import assignment_rate_drift, feature_drift_report, observed_outcome_shift

def test_feature_drift_labels_causal_effect_separately():
    reference = pd.DataFrame({"amount": np.arange(100.), "channel": ["web"]*90+["phone"]*10})
    current = pd.DataFrame({"amount": np.arange(100.)+50, "channel": ["web"]*50+["phone"]*50})
    report = feature_drift_report(reference, current)
    assert report["causal_effect_drift_assessed"] is False
    assert report["feature_drift"]["amount"]["score"] > 0
    assert report["feature_drift"]["channel"]["score"] > 0

def test_assignment_and_outcome_changes_do_not_claim_causal_drift():
    assignment = assignment_rate_drift(pd.Series(["A"]*50+["B"]*50), pd.Series(["A"]*70+["B"]*30))
    outcome = observed_outcome_shift(pd.Series([0., 1.]*50), pd.Series([1., 2.]*50))
    assert abs(assignment["absolute_rate_changes"]["A"] - .2) < 1e-12
    assert outcome["causal_effect_drift_assessed"] is False
