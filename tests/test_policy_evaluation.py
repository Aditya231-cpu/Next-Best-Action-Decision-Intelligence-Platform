import numpy as np
from nba.evaluation.off_policy import inverse_propensity_value, paired_bootstrap_difference
from nba.evaluation.uplift_metrics import auuc, qini_curve, qini_coefficient

def test_ips_recovers_uniform_random_policy_value():
    # A policy that always selects action A under 50/50 random assignment has E[Y]=2.
    arm = np.array(["A", "B"] * 5000)
    y = np.where(arm == "A", 2., 0.)
    result = inverse_propensity_value(arm, y, np.repeat("A", len(arm)), {"A": .5, "B": .5})
    assert abs(result["value"] - 2.) < .04
    assert result["effective_sample_size"] > 4500

def test_paired_bootstrap_difference_is_deterministic():
    a = np.arange(100., dtype=float); b = a - 2
    first = paired_bootstrap_difference(a, b, seed=19, draws=200)
    second = paired_bootstrap_difference(a, b, seed=19, draws=200)
    assert first == second
    assert first["difference"] == 2.

def test_uplift_curve_has_finite_values_for_both_randomized_arms():
    treatment = np.array([0, 1] * 100)
    outcome = np.array([0, 1] * 100)
    scores = np.linspace(0, 1, 200)
    curve = qini_curve(scores, treatment, outcome)
    assert len(curve) == 200
    assert np.isfinite(curve.qini).all()
    assert np.isfinite(qini_coefficient(scores, treatment, outcome))
    assert np.isfinite(auuc(scores, treatment, outcome))

def test_constant_score_ties_have_zero_qini_and_do_not_use_row_order():
    treatment = np.array([0, 1] * 50)
    outcome = np.array([0, 1] * 50)
    scores = np.ones(len(treatment))
    curve = qini_curve(scores, treatment, outcome)
    assert len(curve) == 1
    assert qini_coefficient(scores, treatment, outcome) == 0.0

def test_qini_sign_is_positive_for_correct_ranking_and_negative_when_reversed():
    treatment = np.tile([False, True], 100)
    subgroup = np.repeat([0.0, 1.0], 100)
    scores = subgroup.copy()
    outcome = treatment.astype(float) * subgroup
    assert qini_coefficient(scores, treatment, outcome) > 0
    assert qini_coefficient(-scores, treatment, outcome) < 0

def test_ranking_bootstrap_is_reproducible_and_reports_interval():
    from nba.evaluation.uplift_metrics import ranking_bootstrap
    treatment = np.array([0, 1] * 20)
    outcome = np.array([0, 1] * 20, dtype=float)
    scores = np.linspace(0, 1, 40)
    a = ranking_bootstrap(scores, treatment, outcome, draws=20, seed=7)
    b = ranking_bootstrap(scores, treatment, outcome, draws=20, seed=7)
    assert a == b
    assert a["draws"] == 20
    assert a["qini_coefficient"]["low"] <= a["qini_coefficient"]["high"]
