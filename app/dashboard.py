"""Streamlit decision-intelligence dashboard with real/simulated states separated."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from nba.optimization.constrained_solver import allocate_constrained
from nba.simulation.runner import run_simulation
from nba.services.audit import AuditStore

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "generated"
st.set_page_config(page_title="NBA Decision Intelligence", layout="wide")
st.title("Next-Best-Action Decision Intelligence")
st.caption("Randomized email evidence, simulated scenarios, and policy assumptions are labeled separately.")
page = st.sidebar.selectbox("View", ["Experiment overview", "Causal model comparison", "Policy comparison", "Campaign allocation", "Sequential simulation", "Decision explanation and audit"])

def read_json(name):
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

if page == "Experiment overview":
    quality, analysis = read_json("data_quality.json"), read_json("experiment_analysis.json")
    if quality is None or analysis is None:
        st.warning("Real Hillstrom results are not available yet. Put the original CSV at data/raw/hillstrom.csv, then run python scripts/run_pipeline.py.")
        st.write("No empirical results are displayed until the source CSV passes validation.")
    else:
        st.subheader("Randomized benchmark"); st.write("Source: MineThatData Hillstrom Email Analytics Challenge; two-week outcomes.")
        counts = quality["validation"]["treatment_counts"]
        c1, c2 = st.columns(2); c1.metric("Customers", f"{quality['validation']['rows']:,}"); c2.metric("Arms", len(counts))
        st.dataframe(pd.DataFrame({"treatment": counts.keys(), "customers": counts.values()}), use_container_width=True)
        st.subheader("Average randomized treatment effects")
        st.dataframe(pd.DataFrame(analysis["effects"]), use_container_width=True)
        st.subheader("Covariate balance (standardized mean differences)")
        st.dataframe(pd.DataFrame(analysis["balance"]), use_container_width=True)
        st.caption("Visit, conversion, and spend are outcomes after treatment and are excluded from treatment features.")

elif page == "Causal model comparison":
    report = read_json("training_report.json")
    if report is None:
        st.info("No trained causal model artifact is available. Run the real-data pipeline, then python scripts/train_models.py.")
    else:
        st.write({k: report[k] for k in ("dataset_rows", "seed", "features", "final_test_used", "model_version")})
        for action, model in report["models"].items():
            st.subheader(action); st.write(f"Validation-selected model: **{model['selected']}**")
            st.dataframe(pd.DataFrame(model["candidates"]), use_container_width=True)
            st.caption("Validation Qini is a ranking criterion, not proof of customer-level treatment effects. The final test partition was not used for selection.")

elif page == "Policy comparison":
    report = read_json("policy_evaluation.json")
    if report is None:
        st.info("Policy values need a trained artifact and a held-out real randomized test split. No values are prefilled.")
    else:
        st.warning("Offline estimates on the historical experiment; not a live performance guarantee.")
        st.dataframe(pd.DataFrame(report["results"]), use_container_width=True)
        st.write("Explicit costs:", report["action_costs_assumed_usd"])

elif page == "Campaign allocation":
    st.warning("Scenario inputs below are user assumptions, not estimated treatment effects.")
    n = st.number_input("Cohort size", min_value=1, max_value=10000, value=100)
    budget = st.number_input("Budget (USD)", min_value=0.0, value=1.0, step=.1)
    mens_effect = st.number_input("Assumed incremental spend per mens email (USD/customer)", value=.5, step=.05)
    womens_effect = st.number_input("Assumed incremental spend per womens email (USD/customer)", value=.3, step=.05)
    mens_cost = st.number_input("Assumed mens email cost (USD/customer)", min_value=0.0, value=.01, step=.01)
    womens_cost = st.number_input("Assumed womens email cost (USD/customer)", min_value=0.0, value=.01, step=.01)
    mens_capacity = st.number_input("Mens email capacity", min_value=0, value=int(n), step=1)
    womens_capacity = st.number_input("Womens email capacity", min_value=0, value=int(n), step=1)
    if st.button("Allocate scenario"):
        values = pd.DataFrame({"no_action": [0.] * int(n), "mens_email": [mens_effect-mens_cost] * int(n), "womens_email": [womens_effect-womens_cost] * int(n)})
        result = allocate_constrained(values, {"mens_email": mens_cost, "womens_email": womens_cost}, budget=budget,
            capacities={"mens_email": int(mens_capacity), "womens_email": int(womens_capacity)})
        st.write({"solver": result["status"], "total_cost": result["total_cost"], "remaining_budget": result.get("remaining_budget"),
            "action_counts": result["action_counts"], "expected_net_value_under_inputs": result["total_value"]})

elif page == "Sequential simulation":
    st.warning("SIMULATED ONLY. These assumptions do not estimate real banking or email effects.")
    customers = st.slider("Synthetic customers", 100, 5000, 1000, 100)
    periods = st.slider("Periods", 1, 20, 8)
    fatigue = st.slider("Fatigue strength", 0.0, .8, .12, .02)
    shift = st.slider("Distribution shift", 0.0, .8, .15, .02)
    seed = st.number_input("Seed", min_value=0, value=20261018)
    policy = st.selectbox("Policy", ["random", "response", "uplift", "incremental_net_value", "segment_rule"])
    if st.button("Run synthetic simulation"):
        result = run_simulation(customers=customers, periods=periods, seed=seed, fatigue_strength=fatigue, distribution_shift=shift, policy=policy)
        st.line_chart(result.set_index("period")[["cumulative_net_value", "cumulative_cost", "cumulative_conversions"]])
        st.dataframe(result, use_container_width=True)
        st.caption(result.result_type.iloc[0])

else:
    st.caption("Audit retrieval returns the stored decision evidence; it does not rerun current models.")
    decision_id = st.text_input("Decision ID")
    if st.button("Retrieve audit") and decision_id:
        store = AuditStore(ROOT / "data" / "decisions.sqlite")
        record = store.get(decision_id)
        if record is None: st.error("Decision not found")
        else: st.json(record)
