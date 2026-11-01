from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"

SOURCE = PROJECT_ROOT / "datasets" / "WineQT.csv"
OUTPUT = DRIFT_DIR / "reference_data.csv"

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

df = pd.read_csv(SOURCE)

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

reference = df[FEATURES].dropna().copy()
reference.to_csv(OUTPUT, index=False)

print("Reference dataset created.")
print("Rows:", len(reference))
print("Features:", len(FEATURES))
print("File:", OUTPUT)
