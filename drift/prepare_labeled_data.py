from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE = PROJECT_ROOT / "datasets" / "WineQT.csv"
OUTPUT = PROJECT_ROOT / "drift" / "current_labeled_data.csv"

data = pd.read_csv(SOURCE)

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

# Use only the original training partition.
train_data, _ = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["quality"]
)

# Select 200 real labeled WineQT records.
recent = train_data.sample(
    n=200,
    random_state=123
).copy()

# Simulate changes in selected chemical features.
recent["fixed acidity"] += 2.0
recent["volatile acidity"] += 0.25
recent["citric acid"] += 0.15
recent["alcohol"] -= 1.5

# Save features and their original quality labels.
recent[FEATURES + ["quality"]].to_csv(
    OUTPUT,
    index=False
)

print("Labeled retraining data created.")
print("Rows:", len(recent))
print("Features:", len(FEATURES))
print("Target: quality")
print("Quality distribution:")
print(recent["quality"].value_counts().sort_index())
print("Saved to:", OUTPUT)
