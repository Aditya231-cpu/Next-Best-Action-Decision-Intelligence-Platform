import numpy as np
import pandas as pd
from nba.causal.meta_learners import DoublyRobustLearner, SLearner, TLearner, XLearner
from nba.causal.baselines import ResponsePropensity

def test_binary_treatment_learners_return_finite_customer_effects():
    rng = np.random.default_rng(11)
    n = 800
    x = rng.normal(size=n)
    t = rng.binomial(1, .5, size=n)
    y = 1.5 + .2*x + t*(.4 + .5*(x > 0)) + rng.normal(0, .7, size=n)
    X = pd.DataFrame({"x": x, "segment": np.where(x > 0, "positive", "negative")})
    for estimator in (SLearner(3), TLearner(3), XLearner(3), DoublyRobustLearner(3, folds=2)):
        estimator.fit(X, t, y)
        effect = estimator.predict_effect(X.iloc[:25])
        assert effect.shape == (25,)
        assert np.isfinite(effect).all()

def test_response_targeting_and_uplift_targeting_rank_different_groups():
    rng = np.random.default_rng(41)
    n = 10000
    group = rng.binomial(1, .5, n)  # 1: high baseline response; 0: lower baseline, larger lift
    treatment = rng.binomial(1, .5, n)
    p0 = np.where(group == 1, .75, .05)
    p1 = np.where(group == 1, .75, .45)
    outcome = rng.binomial(1, np.where(treatment == 1, p1, p0))
    X = pd.DataFrame({"group": group})
    response = ResponsePropensity(7).fit(X, treatment, outcome).predict(X)
    uplift = TLearner(7).fit(X, treatment, outcome).predict_effect(X)
    assert response[group == 1].mean() > response[group == 0].mean() + .2
    assert uplift[group == 0].mean() > uplift[group == 1].mean() + .2
