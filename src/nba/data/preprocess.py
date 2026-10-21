"""Canonicalize the verified source CSV and write a separate Parquet file."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from nba.data.download import DEFAULT_PATH, MANUAL_PAGE, SOURCE_URL, sha256_file
from nba.data.validate import validate_frame

ALIASES = {"segment": "treatment", "history_segment": "history_segment", "zip_code": "zip_code"}
def canonicalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize source headers/categories; outcomes remain outcomes, not features."""
    frame = raw.copy()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    frame = frame.rename(columns=ALIASES)
    for col in ("treatment", "zip_code", "channel", "history_segment"):
        if col in frame:
            frame[col] = frame[col].astype("string").str.strip()
    # The original public Hillstrom CSV labels this category "Surburban".
    # Preserve the downloaded source unchanged and normalize only the canonical copy.
    if "zip_code" in frame:
        frame["zip_code"] = frame["zip_code"].replace({"Surburban": "Suburban"})
    treatment_map = {"no e-mail": "No E-Mail", "mens e-mail": "Mens E-Mail", "womens e-mail": "Womens E-Mail"}
    if "treatment" in frame:
        frame["treatment"] = frame["treatment"].str.lower().map(treatment_map)
    # The public source has no customer identifier. This stable row key is an index,
    # not a claimed source identifier, and must not be joined to external records.
    if "customer_id" not in frame:
        frame.insert(0, "customer_id", [f"hillstrom-{i:06d}" for i in range(len(frame))])
    expected = ["customer_id", "treatment", "recency", "history", "history_segment", "mens", "womens",
                "newbie", "zip_code", "channel", "visit", "conversion", "spend"]
    return frame[[c for c in expected if c in frame] + [c for c in frame if c not in expected]]

def preprocess(raw_path: Path = DEFAULT_PATH, output_path: Path = Path("data/processed/hillstrom.parquet")) -> pd.DataFrame:
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing {raw_path.resolve()}. Download from {MANUAL_PAGE} and save it at this path.")
    raw = pd.read_csv(raw_path)
    canonical = canonicalize(raw)
    report = validate_frame(canonical, expected_rows=64000)
    if not report.valid:
        raise ValueError(f"Validation failed: {report.to_dict()}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(output_path, index=False)
    raw_provenance_path = raw_path.with_suffix(raw_path.suffix + ".provenance.json")
    raw_provenance = json.loads(raw_provenance_path.read_text(encoding="utf-8")) if raw_provenance_path.is_file() else {}
    metadata = {"dataset": "Hillstrom MineThatData Email Analytics Challenge", "source_url": SOURCE_URL,
        "manual_download_page": MANUAL_PAGE, "retrieved_at_utc": raw_provenance.get("retrieved_at_utc"),
        "retrieval_date_unknown_for_manual_download": "retrieved_at_utc" not in raw_provenance,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": sha256_file(raw_path), "raw_path": str(raw_path), "processed_path": str(output_path),
        "rows": len(canonical), "treatment_counts": report.treatment_counts,
        "pre_treatment_features": ["recency", "history", "history_segment", "mens", "womens", "newbie", "zip_code", "channel"],
        "outcomes": ["visit", "conversion", "spend"], "customer_id_note": "Generated row key; source supplies no identifier."}
    output_path.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return canonical

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--output", type=Path, default=Path("data/processed/hillstrom.parquet"))
    args = parser.parse_args()
    frame = preprocess(args.raw, args.output)
    print(f"Wrote {len(frame):,} canonical rows to {args.output}")

if __name__ == "__main__": main()
