import numpy as np
import pandas as pd
from nba.causal.experiment_analysis import analyze
from nba.causal.diagnostics import check_pre_treatment_features, randomized_experiment_diagnostics
from nba.causal.uncertainty import bootstrap_ate, ranking_stability

def test_randomized_arm_analysis_recovers_known_mean_differences():
    rng = np.random.default_rng(55)
    n = 18000
    treatment = np.repeat(["No E-Mail", "Mens E-Mail", "Womens E-Mail"], n // 3)
    y = rng.normal(2.0, 1.0, n)
    y[treatment == "Mens E-Mail"] += 0.35
    y[treatment == "Womens E-Mail"] += 0.15
    x = rng.normal(size=n)
    frame = pd.DataFrame({"customer_id": [str(i) for i in range(n)], "treatment": treatment,
        "recency": x, "history": x, "mens": np.zeros(n), "womens": np.ones(n), "newbie": np.zeros(n),
        "zip_code": ["Urban"] * n, "channel": ["Web"] * n, "visit": (y > 2).astype(int),
        "conversion": (y > 3).astype(int), "spend": y})
    report = analyze(frame, seed=2, bootstrap_draws=150)
    observed = {(r["treatment"], r["outcome"]): r["effect"] for r in report["effects"]}
    assert abs(observed[("Mens E-Mail", "spend")] - .35) < .06
    assert abs(observed[("Womens E-Mail", "spend")] - .15) < .06
    assert all(r["n_treatment"] == n // 3 and r["n_control"] == n // 3 for r in report["effects"])

def test_diagnostics_detect_leakage_and_measure_randomization():
    frame = pd.DataFrame({"treatment": ["No E-Mail", "Mens E-Mail", "Womens E-Mail"] * 20,
        "recency": np.tile(np.arange(3), 20), "history": np.ones(60), "mens": np.zeros(60),
        "womens": np.ones(60), "newbie": np.zeros(60), "zip_code": ["Urban"] * 60,
        "channel": ["Web"] * 60})
    check_pre_treatment_features(["recency", "history"])
    try:
        check_pre_treatment_features(["visit"])
    except ValueError:
        pass
    else:
        raise AssertionError("Post-treatment feature should be rejected")
    result = randomized_experiment_diagnostics(frame, feature_columns=["recency", "history"])
    assert result["sample_ratio_mismatch_p"] > .99
    assert result["maximum_absolute_smd"] < .2

def test_bootstrap_interval_contains_known_effect_and_ranking_stability_is_bounded():
    rng = np.random.default_rng(90)
    frame = pd.DataFrame({"treatment": ["control"] * 500 + ["treated"] * 500,
        "outcome": np.r_[rng.normal(0, 1, 500), rng.normal(.3, 1, 500)]})
    ci = bootstrap_ate(frame, treatment="treated", control="control", outcome="outcome", draws=300, seed=1)
    assert ci["ci_low"] < ci["estimate"] < ci["ci_high"]
    scores = rng.normal(size=(12, 80, 2))
    stable = ranking_stability(scores, top_fraction=.1)
    assert sum(stable["action_selection_frequency"]) == 1.
    assert 0 <= stable["mean_top_customer_jaccard"] <= 1
