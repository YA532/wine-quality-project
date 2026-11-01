import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPORT = ROOT / "drift" / "drift_report.json"
METRICS = ROOT / "drift" / "retraining_metrics.json"
CANDIDATE = ROOT / "drift" / "wine_quality_model_candidate.pkl"
APPROVED = ROOT / "wine_quality_model.pkl"

print("Checking drift and candidate model quality...")

with open(REPORT) as file:
    drift = json.load(file)

with open(METRICS) as file:
    metrics = json.load(file)

if not drift["overall_drift"]:
    print("RETRAINING NOT REQUIRED: No data drift detected.")
    sys.exit(0)

approved_accuracy = metrics["approved_accuracy"]
candidate_accuracy = metrics["candidate_accuracy"]
minimum_accuracy = metrics["minimum_accuracy"]

print("Approved accuracy:", round(approved_accuracy, 4))
print("Candidate accuracy:", round(candidate_accuracy, 4))
print("Minimum required:", minimum_accuracy)

if not CANDIDATE.exists() or not APPROVED.exists():
    print("APPROVAL FAILED: Model file missing.")
    sys.exit(1)

if candidate_accuracy < minimum_accuracy:
    print("APPROVAL FAILED: Minimum accuracy not achieved.")
    sys.exit(1)

if candidate_accuracy <= approved_accuracy:
    print("APPROVAL FAILED: Candidate did not improve performance.")
    print("Approved model will be retained.")
    sys.exit(1)

print("APPROVAL PASSED: Candidate is better.")
print("Candidate is eligible for manual promotion.")
print("No automatic replacement performed.")
sys.exit(0)
