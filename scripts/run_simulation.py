"""Run a reproducible synthetic sequential scenario and label its outputs."""
import argparse
from pathlib import Path
import pandas as pd
import yaml
from nba.simulation.runner import run_simulation

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/simulation.yaml"))
    parser.add_argument("--policy", default="incremental_net_value")
    parser.add_argument("--out", type=Path, default=Path("reports/generated/simulation.csv"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = run_simulation(customers=config["customers"], periods=config["periods"], seed=config["seed"],
        contact_cap=config["contact_cap"], contact_capacity_per_period=config["contact_capacity_per_period"],
        fatigue_strength=config["fatigue_strength"], distribution_shift=config["distribution_shift"],
        conversion_value=config["conversion_value"], action_costs=config["action_costs"], policy=args.policy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Wrote explicitly simulated results to {args.out.resolve()}")

if __name__ == "__main__": main()
