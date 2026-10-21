"""Reproducible customer-level train/validation/test splits."""
from __future__ import annotations
from sklearn.model_selection import train_test_split
import pandas as pd

def split_customers(frame: pd.DataFrame, *, seed: int = 20261018, test_size: float = .2, validation_size: float = .2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return disjoint train/validation/test frames, stratified by randomized arm."""
    if "customer_id" not in frame or frame["customer_id"].duplicated().any():
        raise ValueError("A unique customer_id is required for customer-level partitioning")
    train_valid, test = train_test_split(frame, test_size=test_size, random_state=seed, stratify=frame["treatment"])
    val_fraction_of_remainder = validation_size / (1 - test_size)
    train, validation = train_test_split(train_valid, test_size=val_fraction_of_remainder,
                                         random_state=seed + 1, stratify=train_valid["treatment"])
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)
