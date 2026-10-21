"""Validate manually supplied source CSV and save a report without modifying it."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from nba.data.preprocess import canonicalize
from nba.data.validate import validate_frame

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/hillstrom.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/generated/data_validation.json"))
    args = parser.parse_args()
    if not args.raw.is_file(): raise SystemExit(f"Missing {args.raw.resolve()}. Download the original CSV from the MineThatData experiment page.")
    frame = canonicalize(pd.read_csv(args.raw))
    report = validate_frame(frame, expected_rows=64000)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2))
    if not report.valid: raise SystemExit(1)

if __name__ == "__main__": main()
