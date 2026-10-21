"""Randomized Hillstrom arm comparisons with bootstrap spending intervals."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from nba.data.validate import validate_frame

ARMS = ("Mens E-Mail", "Womens E-Mail")
OUTCOMES = ("visit", "conversion", "spend")

def _bootstrap_difference(treated: np.ndarray, control: np.ndarray, *, seed: int, draws: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    t = rng.choice(treated, size=(draws, len(treated)), replace=True).mean(axis=1)
    c = rng.choice(control, size=(draws, len(control)), replace=True).mean(axis=1)
    low, high = np.quantile(t - c, [.025, .975])
    return float(low), float(high)

def analyze(frame: pd.DataFrame, *, seed: int = 20261018, bootstrap_draws: int = 2000) -> dict:
    """Calculate unadjusted intention-to-treat differences and Welch intervals."""
    check = validate_frame(frame)
    if not check.valid: raise ValueError(f"Invalid frame: {check.to_dict()}")
    control = frame.loc[frame.treatment == "No E-Mail"]
    results = []
    raw_p = []
    for arm_i, arm in enumerate(ARMS):
        treated = frame.loc[frame.treatment == arm]
        for outcome_i, outcome in enumerate(OUTCOMES):
            x, y = treated[outcome].astype(float), control[outcome].astype(float)
            effect = float(x.mean() - y.mean())
            se = float(np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)))
            test = stats.ttest_ind(x, y, equal_var=False)
            df_num = (x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)) ** 2
            df_den = (x.var(ddof=1) / len(x)) ** 2 / (len(x) - 1) + (y.var(ddof=1) / len(y)) ** 2 / (len(y) - 1)
            dof = df_num / df_den if df_den else float("inf")
            crit = stats.t.ppf(.975, dof)
            if outcome == "spend":
                ci_low, ci_high = _bootstrap_difference(x.to_numpy(), y.to_numpy(), seed=seed + arm_i * 10 + outcome_i, draws=bootstrap_draws)
                interval_method = f"percentile bootstrap ({bootstrap_draws} resamples, stratified within arms)"
            else:
                ci_low, ci_high = effect - crit * se, effect + crit * se
                interval_method = "Welch unequal-variance t interval"
            p = float(test.pvalue)
            raw_p.append(p)
            results.append({"treatment": arm, "control": "No E-Mail", "outcome": outcome,
                "effect": effect, "standard_error": se, "ci95_low": ci_low, "ci95_high": ci_high,
                "p_value": p, "n_treatment": int(len(x)), "n_control": int(len(y)),
                "mean_treatment": float(x.mean()), "mean_control": float(y.mean()), "interval_method": interval_method,
                "direction": "positive" if effect > 0 else "negative" if effect < 0 else "zero",
                "practical_interpretation": (f"{effect:.6g} currency units incremental spend per randomized customer" if outcome == "spend" else f"{effect*100:.4g} percentage-point difference in {outcome}")})
    # Holm step-down adjustment over the six arm-by-outcome confirmatory comparisons.
    order = np.argsort(raw_p)
    adjusted = np.empty(len(raw_p))
    running = 0.0
    m = len(raw_p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * raw_p[idx]))
        adjusted[idx] = running
    for result, p_adj in zip(results, adjusted): result["p_value_holm"] = float(p_adj)
    balance = _balance_table(frame)
    return {"n": int(len(frame)), "treatment_counts": check.treatment_counts, "balance": balance,
        "effects": results, "limitations": ["Population-average intention-to-treat estimates; individual effects are not observed.",
        "The source is a historical two-week email experiment and does not identify banking-action effects.",
        "Spend is zero-inflated; percentile bootstrap intervals are reported alongside the Welch standard error."]}

def _balance_table(frame: pd.DataFrame) -> list[dict]:
    features = [c for c in ("recency", "history", "mens", "womens", "newbie") if c in frame]
    records = []
    control = frame[frame.treatment == "No E-Mail"]
    for arm in ARMS:
        group = frame[frame.treatment == arm]
        for feature in features:
            a, b = group[feature].astype(float), control[feature].astype(float)
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            smd = float((a.mean() - b.mean()) / pooled) if pooled else 0.0
            records.append({"treatment": arm, "feature": feature, "smd": smd})
        for feature in ("zip_code", "channel", "history_segment"):
            if feature not in frame:
                continue
            levels = sorted(set(frame[feature].dropna().astype(str)))
            for level in levels:
                a = (group[feature].astype("string") == level).astype(float)
                b = (control[feature].astype("string") == level).astype(float)
                pooled = np.sqrt((a.mean() * (1 - a.mean()) + b.mean() * (1 - b.mean())) / 2)
                smd = float((a.mean() - b.mean()) / pooled) if pooled else 0.0
                records.append({"treatment": arm, "feature": f"{feature}={level}", "smd": smd})
    return records

def write_report(frame: pd.DataFrame, output_dir: Path, *, seed: int = 20261018) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = analyze(frame, seed=seed)
    (output_dir / "experiment_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report["effects"]).to_csv(output_dir / "average_treatment_effects.csv", index=False)
    pd.DataFrame(report["balance"]).to_csv(output_dir / "covariate_balance.csv", index=False)
    lines = ["# Hillstrom randomized experiment analysis", "", f"Observations: {report['n']:,}", "",
             "This report contains estimates from the randomized email arms only. Spending is measured in dollars per randomized customer. Binary outcome effects are probability differences (percentage points after multiplying by 100).", "",
             "| Email vs control | Outcome | Effect | 95% interval | Holm p-value |", "|---|---:|---:|---:|---:|"]
    for r in report["effects"]:
        lines.append(f"| {r['treatment']} | {r['outcome']} | {r['effect']:.6g} | [{r['ci95_low']:.6g}, {r['ci95_high']:.6g}] | {r['p_value_holm']:.4g} |")
    lines += ["", "## Limitations", *[f"- {item}" for item in report["limitations"]], "",
              "This analysis is an intention-to-treat arm comparison. It does not reveal any customer's true individual effect."]
    (output_dir / "experiment_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/hillstrom.parquet"))
    parser.add_argument("--out", type=Path, default=Path("reports/generated"))
    args = parser.parse_args()
    if not args.data.is_file(): raise SystemExit(f"Missing processed real data: {args.data.resolve()}. Run preprocessing first; synthetic substitution is not permitted.")
    write_report(pd.read_parquet(args.data), args.out)
    print(f"Wrote real-data experiment report under {args.out.resolve()}")

if __name__ == "__main__": main()
