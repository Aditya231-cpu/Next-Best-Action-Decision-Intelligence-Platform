"""Predictive-response and transparent segment baselines."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = list(frame.select_dtypes(include=["number", "bool"]).columns)
    categorical = [c for c in frame.columns if c not in numeric]
    return ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical)])

class ResponsePropensity:
    """Estimate outcome probability in the treated arm; this is not uplift."""
    def __init__(self, seed: int = 0): self.seed = seed
    def fit(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray):
        treated = np.asarray(treatment).astype(bool)
        self.model_ = Pipeline([("features", _preprocessor(X)), ("model", LogisticRegression(max_iter=1000, random_state=self.seed))])
        self.model_.fit(X.loc[treated], np.asarray(outcome)[treated])
        return self
    def predict(self, X: pd.DataFrame) -> np.ndarray: return self.model_.predict_proba(X)[:, 1]

def category_rule(X: pd.DataFrame, action: str) -> np.ndarray:
    """Return the pre-period category rule score for one email action."""
    col = "mens" if action == "Mens E-Mail" else "womens" if action == "Womens E-Mail" else None
    if col is None or col not in X: raise ValueError(f"No supported category feature for action {action!r}")
    return X[col].astype(float).to_numpy()
