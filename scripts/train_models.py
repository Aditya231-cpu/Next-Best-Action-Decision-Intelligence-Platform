"""Train and validation-select causal learners; never reads the final test partition."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import joblib
import pandas as pd
from nba.causal.meta_learners import DoublyRobustLearner, SLearner, TLearner, XLearner
from nba.data.split import split_customers
from nba.evaluation.uplift_metrics import qini_coefficient

FEATURES = ["recency", "history", "history_segment", "mens", "womens", "newbie", "zip_code", "channel"]
ARMS = {"mens_email": "Mens E-Mail", "womens_email": "Womens E-Mail"}

def train(data_path: Path, artifact_dir: Path, seed: int = 20261018) -> dict:
    if not data_path.is_file(): raise FileNotFoundError(f"Missing canonical real dataset: {data_path.resolve()}")
    frame = pd.read_parquet(data_path)
    train_df, validation, _untouched_test = split_customers(frame, seed=seed)
    features = [c for c in FEATURES if c in frame.columns]
    if not features: raise ValueError("No supported pre-treatment features found")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report = {"dataset_rows": len(frame), "seed": seed, "features": features, "selection_partition": "validation",
        "final_test_used": False, "outcome": "spend", "models": {}}
    for idx, (key, arm) in enumerate(ARMS.items()):
        arm_train = train_df[train_df.treatment.isin(["No E-Mail", arm])]
        arm_val = validation[validation.treatment.isin(["No E-Mail", arm])]
        X_train = arm_train[features]; t_train = (arm_train.treatment == arm).to_numpy(); y_train = arm_train.spend.to_numpy(float)
        X_val = arm_val[features]; t_val = (arm_val.treatment == arm).to_numpy(); y_val = arm_val.spend.to_numpy(float)
        candidates = [SLearner(seed + idx), TLearner(seed + idx), XLearner(seed + idx), DoublyRobustLearner(seed + idx, folds=3)]
        scores = []
        for candidate in candidates:
            candidate.fit(X_train, t_train, y_train)
            tau = candidate.predict_effect(X_val)
            metric = qini_coefficient(tau, t_val, y_val)
            scores.append({"name": type(candidate).__name__, "model": candidate, "validation_qini": metric})
        selected = max(scores, key=lambda item: item["validation_qini"])
        model_path = artifact_dir / f"{key}_spend.joblib"
        joblib.dump(selected["model"], model_path)
        report["models"][key] = {"selected": selected["name"], "validation_qini": selected["validation_qini"],
            "candidates": [{"name": s["name"], "validation_qini": s["validation_qini"]} for s in scores],
            "artifact": str(model_path)}
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    report["dataset_sha256"] = digest
    report["model_version"] = hashlib.sha256(f"{digest}:{seed}".encode()).hexdigest()[:16]
    (artifact_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/hillstrom.parquet"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    result = train(args.data, args.artifacts, args.seed)
    print(f"Trained models selected on validation data only. Final test used: {result['final_test_used']}")

if __name__ == "__main__": main()
