from pathlib import Path
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"

REFERENCE = DRIFT_DIR / "reference_data.csv"
OUTPUT = DRIFT_DIR / "current_data.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["stable", "drift"], default="stable")
args = parser.parse_args()

rng = np.random.default_rng(42)
reference = pd.read_csv(REFERENCE)

if args.mode == "stable":
    # Use the exact reference data to demonstrate no drift.
    current = reference.copy()
else:
    # Create a sample and deliberately shift selected features.
    current = reference.sample(
        n=min(200, len(reference)),
        replace=True,
        random_state=42
    ).reset_index(drop=True)

    for feature in current.columns:
        current[feature] += rng.normal(0, 0.01, size=len(current))

    current["fixed acidity"] += 2.0
    current["volatile acidity"] += 0.25
    current["citric acid"] += 0.15
    current["alcohol"] -= 1.5

current.to_csv(OUTPUT, index=False)

print("Mode:", args.mode)
print("Rows:", len(current))
print("File:", OUTPUT)
