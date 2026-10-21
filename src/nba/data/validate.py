"""Schema and quality checks for the canonical Hillstrom dataset."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import pandas as pd

TREATMENTS = {"No E-Mail", "Mens E-Mail", "Womens E-Mail"}
VALID_CATEGORIES = {"zip_code": {"Rural", "Suburban", "Urban"}, "channel": {"Web", "Phone", "Multichannel"}}
PRE_TREATMENT = {"recency", "history", "mens", "womens", "newbie", "zip_code", "channel"}
OUTCOMES = {"visit", "conversion", "spend"}
REQUIRED = PRE_TREATMENT | OUTCOMES | {"treatment"}

@dataclass
class ValidationReport:
    valid: bool
    rows: int
    columns: list[str]
    missing_columns: list[str]
    treatment_counts: dict[str, int]
    missing_treatment_arms: list[str]
    missing_by_column: dict[str, int]
    duplicate_customer_ids: int | None
    invalid_treatments: list[str]
    invalid_category_values: dict[str, list[str]]
    warnings: list[str]
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def validate_frame(frame: pd.DataFrame, *, expected_rows: int | None = None) -> ValidationReport:
    missing_columns = sorted(REQUIRED - set(frame.columns))
    if "treatment" in frame:
        treatment_counts = frame["treatment"].value_counts(dropna=False).astype(int).to_dict()
        invalid = sorted(str(x) for x in set(frame["treatment"].dropna()) - TREATMENTS)
        missing_arms = sorted(TREATMENTS - set(frame["treatment"].dropna()))
        if frame["treatment"].isna().any(): invalid.append("<missing>")
    else:
        treatment_counts, invalid, missing_arms = {}, ["treatment column missing"], sorted(TREATMENTS)
    invalid_categories = {}
    warnings: list[str] = []
    if missing_arms:
        warnings.append(f"Expected randomized treatment arms are absent: {missing_arms}")
    for column, allowed in VALID_CATEGORIES.items():
        if column in frame:
            unexpected = sorted(str(v) for v in set(frame[column].dropna().astype(str)) - allowed)
            if unexpected:
                invalid_categories[column] = unexpected
                warnings.append(f"Unexpected category values in {column}: {unexpected}")
    if expected_rows is not None and len(frame) != expected_rows:
        warnings.append(f"Expected {expected_rows:,} records but found {len(frame):,}; verify source version.")
    outcome_missing = {c: int(frame[c].isna().sum()) for c in OUTCOMES if c in frame}
    if any(outcome_missing.values()): warnings.append("Post-treatment outcome columns contain missing values.")
    if not (OUTCOMES & set(frame.columns)).isdisjoint(PRE_TREATMENT):
        warnings.append("Potential post-treatment leakage: an outcome appears among pre-treatment features.")
    duplicate_ids = None
    if "customer_id" in frame:
        duplicate_ids = int(frame["customer_id"].duplicated().sum())
        if duplicate_ids: warnings.append(f"Found {duplicate_ids} duplicated customer_id values.")
    valid = not missing_columns and not invalid and not missing_arms and not invalid_categories and not any(outcome_missing.values()) and duplicate_ids in (None, 0)
    return ValidationReport(valid, len(frame), [str(c) for c in frame.columns], missing_columns,
        {str(k): v for k, v in treatment_counts.items()}, missing_arms,
        {str(k): int(v) for k, v in frame.isna().sum().items()},
        duplicate_ids, invalid, invalid_categories, warnings)

def validate_csv(path: Path, *, expected_rows: int | None = 64000) -> ValidationReport:
    if not path.is_file(): raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    if path.stat().st_size < 1024: raise ValueError(f"Dataset is unexpectedly small ({path.stat().st_size} bytes): {path}")
    frame = pd.read_csv(path)
    report = validate_frame(frame, expected_rows=expected_rows)
    if not report.valid:
        raise ValueError(f"Hillstrom validation failed: missing={report.missing_columns}; invalid treatments={report.invalid_treatments}; invalid categories={report.invalid_category_values}; warnings={report.warnings}")
    return report
