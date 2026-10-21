"""Randomization, overlap, balance, and leakage diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from nba.causal.experiment_analysis import _balance_table
from nba.data.validate import OUTCOMES, TREATMENTS

def check_pre_treatment_features(feature_columns) -> None:
    leaked = sorted(set(feature_columns) & OUTCOMES)
    if leaked: raise ValueError(f"Post-treatment outcomes cannot be policy features: {leaked}")

def randomized_experiment_diagnostics(frame: pd.DataFrame, *, feature_columns=None) -> dict:
    if feature_columns is not None: check_pre_treatment_features(feature_columns)
    counts = frame.treatment.value_counts().reindex(sorted(TREATMENTS), fill_value=0)
    expected = np.full(len(counts), len(frame) / len(counts))
    srm = chisquare(counts.to_numpy(), f_exp=expected)
    balance = _balance_table(frame)
    assignments = {"observed_propensity": (counts / len(frame)).to_dict(),
        "minimum_arm_count": int(counts.min()), "maximum_arm_count": int(counts.max()),
        "positive_probability_all_arms": bool((counts > 0).all())}
    return {"assignment_counts": counts.astype(int).to_dict(), "assignment_probabilities": assignments,
        "sample_ratio_mismatch_chi2": float(srm.statistic), "sample_ratio_mismatch_p": float(srm.pvalue),
        "maximum_absolute_smd": max((abs(item["smd"]) for item in balance), default=0.0),
        "balance": balance, "interpretation": "Randomization diagnostics can reveal anomalies; passing checks do not prove every assumption."}
