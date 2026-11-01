import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "datasets" / "WineQT.csv"
RECENT_DATA = ROOT / "drift" / "current_labeled_data.csv"
APPROVED_MODEL = ROOT / "wine_quality_model.pkl"

CANDIDATE_MODEL = ROOT / "drift" / "wine_quality_model_candidate.pkl"
METRICS_FILE = ROOT / "drift" / "retraining_metrics.json"

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Load original and recent labeled data.
historical = pd.read_csv(DATASET)
recent = pd.read_csv(RECENT_DATA)

required_columns = FEATURES + ["quality"]

if not set(required_columns).issubset(recent.columns):
    raise ValueError("Recent labeled data is missing required columns.")

if not APPROVED_MODEL.exists():
    raise FileNotFoundError("Approved model was not found.")

# Re-create the original fixed train/test split.
train_data, test_data = train_test_split(
    historical,
    test_size=0.20,
    random_state=42,
    stratify=historical["quality"]
)

# Evaluate the approved model on the same test data.
X_test = test_data[FEATURES]
y_test = test_data["quality"]

approved_model = joblib.load(APPROVED_MODEL)
approved_predictions = approved_model.predict(X_test)
approved_accuracy = accuracy_score(y_test, approved_predictions)

# Combine historical training data with recent labeled data.
combined = pd.concat(
    [train_data[required_columns], recent[required_columns]],
    ignore_index=True
)

X_train = combined[FEATURES]
y_train = combined["quality"]

# Train a separate candidate model.
candidate_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ))
])

candidate_model.fit(X_train, y_train)

candidate_predictions = candidate_model.predict(X_test)
candidate_accuracy = accuracy_score(y_test, candidate_predictions)

# Save candidate separately. Do not replace the approved model.
joblib.dump(candidate_model, CANDIDATE_MODEL)

metrics = {
    "approved_accuracy": float(approved_accuracy),
    "candidate_accuracy": float(candidate_accuracy),
    "minimum_accuracy": 0.40,
    "candidate_training_records": len(X_train),
    "testing_records": len(X_test),
    "quality_classes": [
        int(value) for value in candidate_model.classes_
    ],
    "features": FEATURES,
    "candidate_model": str(CANDIDATE_MODEL)
}

with open(METRICS_FILE, "w") as file:
    json.dump(metrics, file, indent=4)

print("\nCandidate retraining completed.")
print("--------------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Approved model accuracy:", round(approved_accuracy, 4))
print("Candidate model accuracy:", round(candidate_accuracy, 4))
print("Minimum required accuracy: 0.4000")
print("\nCandidate saved to:", CANDIDATE_MODEL)
print("Metrics saved to:", METRICS_FILE)
print("\nApproved model remains unchanged.")
