"""Run the real-data Stage 1 pipeline. This never fabricates a data substitute."""
from pathlib import Path
from nba.data.download import DEFAULT_PATH, acquire
from nba.data.preprocess import preprocess
from nba.data.audit import write_audit
from nba.causal.experiment_analysis import write_report

def main() -> None:
    raw_path = DEFAULT_PATH
    if not raw_path.exists():
        try:
            acquire(raw_path)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    processed_path = Path("data/processed/hillstrom.parquet")
    frame = preprocess(raw_path, processed_path)
    write_audit(processed_path, Path("reports/generated"))
    write_report(frame, Path("reports/generated"))
    print("Stage 1 real-data pipeline completed. Reports are under reports/generated.")

if __name__ == "__main__": main()
