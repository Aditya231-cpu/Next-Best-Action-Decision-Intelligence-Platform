"""Write a complete data-quality report without altering source data."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from nba.data.validate import validate_frame
from nba.causal.diagnostics import randomized_experiment_diagnostics

def build_audit(frame: pd.DataFrame) -> dict:
    validation = validate_frame(frame, expected_rows=64000)
    result = {"validation": validation.to_dict(), "numerical_descriptives": {},
        "outcome_distributions_by_treatment": {},
        "randomization_diagnostics": randomized_experiment_diagnostics(frame,
            feature_columns=[c for c in ("recency", "history", "history_segment", "mens", "womens", "newbie", "zip_code", "channel") if c in frame]),
        "leakage_note": "visit, conversion, and spend are post-treatment outcomes and are excluded from policy features.",
        "source_identifier_note": "The published dataset has no customer identifier; canonical customer_id is a generated row key."}
    for col in ("recency", "history", "mens", "womens", "newbie", "visit", "conversion", "spend"):
        if col in frame:
            result["numerical_descriptives"][col] = frame[col].describe().to_dict()
    if {"treatment", "visit", "conversion", "spend"}.issubset(frame.columns):
        grouped = frame.groupby("treatment", observed=True)[["visit", "conversion", "spend"]]
        result["outcome_distributions_by_treatment"] = {
            "means": grouped.mean().to_dict(orient="index"),
            "medians": grouped.median().to_dict(orient="index"),
            "nonzero_spend_rate": (frame.assign(_positive=frame.spend > 0).groupby("treatment", observed=True)["_positive"].mean()).to_dict(),
        }
    return result

def write_audit(data_path: Path, out_dir: Path) -> dict:
    if not data_path.is_file():
        raise FileNotFoundError(f"No processed real dataset at {data_path.resolve()}; run the acquisition and preprocessing steps first.")
    frame = pd.read_parquet(data_path)
    report = build_audit(frame)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_quality.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    validation = report["validation"]
    lines = ["# Data quality audit", "", f"- Rows: {validation['rows']:,}", f"- Columns: {len(validation['columns'])}",
        f"- Validation passed: {validation['valid']}", "", "## Treatment counts", "", "| Treatment | N |", "|---|---:|"]
    lines += [f"| {arm} | {count:,} |" for arm, count in validation["treatment_counts"].items()]
    lines += ["", "## Missingness", "", "| Column | Missing |", "|---|---:|"]
    lines += [f"| {col} | {n:,} |" for col, n in validation["missing_by_column"].items()]
    lines += ["", "## Warnings", ""] + ([f"- {w}" for w in validation["warnings"]] or ["- None."])
    lines += ["", "## Covariate balance", "", f"Maximum absolute SMD: {report['randomization_diagnostics']['maximum_absolute_smd']:.4f}.",
        f"Sample-ratio mismatch p-value: {report['randomization_diagnostics']['sample_ratio_mismatch_p']:.4g}.",
        "Balance checks can detect anomalies but do not prove all causal assumptions."]
    lines += ["", "## Leakage control", "", report["leakage_note"], ""]
    (out_dir / "data_quality.md").write_text("\n".join(lines), encoding="utf-8")
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/hillstrom.parquet"))
    parser.add_argument("--out", type=Path, default=Path("reports/generated"))
    args = parser.parse_args()
    write_audit(args.data, args.out)
    print(f"Wrote quality audit under {args.out.resolve()}")

if __name__ == "__main__": main()
