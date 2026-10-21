import pandas as pd
import pytest
from nba.data.preprocess import canonicalize
from nba.data.validate import validate_frame
from nba.data.split import split_customers

def sample_frame(n=90):
    arms = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"] * (n // 3)
    return pd.DataFrame({"customer_id": [f"x{i}" for i in range(n)], "treatment": arms,
        "recency": [1] * n, "history": [100.] * n, "mens": [0] * n, "womens": [1] * n,
        "newbie": [0] * n, "zip_code": ["Urban"] * n, "channel": ["Web"] * n,
        "visit": [0] * n, "conversion": [0] * n, "spend": [0.] * n})

def test_canonicalize_source_column_names_and_arms():
    raw = sample_frame().drop(columns="customer_id").rename(columns={"treatment": "Segment", "zip_code": "Zip_Code"})
    result = canonicalize(raw)
    assert result.customer_id.is_unique
    assert set(result.treatment) == {"No E-Mail", "Mens E-Mail", "Womens E-Mail"}
    assert validate_frame(result).valid

def test_invalid_treatment_and_missing_schema_are_reported():
    frame = sample_frame().drop(columns="visit")
    frame.loc[0, "treatment"] = "Coupon"
    report = validate_frame(frame)
    assert not report.valid
    assert "visit" in report.missing_columns
    assert "Coupon" in report.invalid_treatments

def test_unexpected_category_is_rejected():
    frame = sample_frame()
    frame.loc[0, "zip_code"] = "Offshore"
    report = validate_frame(frame)
    assert not report.valid
    assert report.invalid_category_values["zip_code"] == ["Offshore"]

def test_missing_randomized_arm_is_rejected():
    frame = sample_frame()
    frame = frame[frame.treatment != "Womens E-Mail"]
    report = validate_frame(frame)
    assert not report.valid
    assert report.missing_treatment_arms == ["Womens E-Mail"]

def test_post_treatment_outcome_is_not_a_feature():
    assert {"visit", "conversion", "spend"}.isdisjoint({"recency", "history", "mens", "womens", "newbie", "zip_code", "channel"})

def test_customer_split_disjoint_reproducible_and_stratified():
    frame = sample_frame(300)
    first = split_customers(frame, seed=7)
    second = split_customers(frame, seed=7)
    assert [set(p.customer_id) for p in first] == [set(p.customer_id) for p in second]
    assert not (set(first[0].customer_id) & set(first[1].customer_id))
    assert not (set(first[0].customer_id) & set(first[2].customer_id))
    assert not (set(first[1].customer_id) & set(first[2].customer_id))

def test_duplicate_identifier_rejected_by_split():
    frame = sample_frame()
    frame.loc[1, "customer_id"] = frame.loc[0, "customer_id"]
    with pytest.raises(ValueError): split_customers(frame)
