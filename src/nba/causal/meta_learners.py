"""S-, T-, X-, and doubly robust learners for a binary treatment contrast."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = list(X.select_dtypes(include=["number", "bool"]).columns)
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical)])

def _regressor(seed: int = 0):
    return RandomForestRegressor(n_estimators=160, min_samples_leaf=25, max_features=1.0, n_jobs=1, random_state=seed)

def _pipeline(X: pd.DataFrame, model):
    return Pipeline([("features", _preprocessor(X)), ("model", model)])

class TLearner:
    def __init__(self, seed: int = 0): self.seed = seed
    def fit(self, X: pd.DataFrame, treatment, y):
        t, y = np.asarray(treatment).astype(bool), np.asarray(y, dtype=float)
        if t.sum() == 0 or (~t).sum() == 0: raise ValueError("Both treatment arms must be represented")
        self.control_ = _pipeline(X, _regressor(self.seed)).fit(X.loc[~t], y[~t])
        self.treated_ = _pipeline(X, _regressor(self.seed + 1)).fit(X.loc[t], y[t])
        return self
    def predict_effect(self, X): return self.treated_.predict(X) - self.control_.predict(X)

class SLearner:
    def __init__(self, seed: int = 0): self.seed = seed
    def fit(self, X: pd.DataFrame, treatment, y):
        self.columns_ = list(X.columns)
        augmented = X.copy(); augmented["__treatment_indicator__"] = np.asarray(treatment, dtype=float)
        self.model_ = _pipeline(augmented, _regressor(self.seed)).fit(augmented, np.asarray(y, dtype=float))
        return self
    def _augmented(self, X, arm):
        result = X.copy(); result["__treatment_indicator__"] = float(arm); return result
    def predict_effect(self, X): return self.model_.predict(self._augmented(X, 1)) - self.model_.predict(self._augmented(X, 0))

class XLearner:
    """X-learner with random-forest outcome and imputed-effect models."""
    def __init__(self, seed: int = 0): self.seed = seed
    def fit(self, X: pd.DataFrame, treatment, y):
        t, y = np.asarray(treatment).astype(bool), np.asarray(y, dtype=float)
        if t.sum() == 0 or (~t).sum() == 0: raise ValueError("Both treatment arms must be represented")
        self.mu0_ = _pipeline(X, _regressor(self.seed)).fit(X.loc[~t], y[~t])
        self.mu1_ = _pipeline(X, _regressor(self.seed + 1)).fit(X.loc[t], y[t])
        d1 = y[t] - self.mu0_.predict(X.loc[t]); d0 = self.mu1_.predict(X.loc[~t]) - y[~t]
        self.tau1_ = _pipeline(X, _regressor(self.seed + 2)).fit(X.loc[t], d1)
        self.tau0_ = _pipeline(X, _regressor(self.seed + 3)).fit(X.loc[~t], d0)
        self.propensity_ = float(t.mean())
        return self
    def predict_effect(self, X):
        return self.propensity_ * self.tau0_.predict(X) + (1 - self.propensity_) * self.tau1_.predict(X)

class DoublyRobustLearner:
    """Cross-fitted AIPW pseudo-outcome regression for a randomized contrast."""
    def __init__(self, seed: int = 0, folds: int = 3): self.seed, self.folds = seed, folds
    def fit(self, X: pd.DataFrame, treatment, y):
        t, y = np.asarray(treatment).astype(bool), np.asarray(y, dtype=float)
        if min(t.sum(), (~t).sum()) < self.folds: raise ValueError("Each arm needs at least one observation per fold")
        m0, m1 = np.zeros(len(y)), np.zeros(len(y))
        splitter = KFold(self.folds, shuffle=True, random_state=self.seed)
        for train, test in splitter.split(X):
            for arm, target in ((False, m0), (True, m1)):
                mask = t[train] == arm
                if not mask.any(): raise ValueError("A training fold is missing a treatment arm")
                model = _pipeline(X.iloc[train], _regressor(self.seed + int(arm)))
                model.fit(X.iloc[train].iloc[mask], y[train][mask])
                target[test] = model.predict(X.iloc[test])
        p = float(t.mean())
        if not 0 < p < 1: raise ValueError("Treatment probability must be strictly between zero and one")
        pseudo = m1 - m0 + t * (y - m1) / p - (~t) * (y - m0) / (1 - p)
        self.effect_ = _pipeline(X, _regressor(self.seed + 99)).fit(X, pseudo)
        return self
    def predict_effect(self, X): return self.effect_.predict(X)
