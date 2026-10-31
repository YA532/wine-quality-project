import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "datasets" / "WineQT.csv"


def train_model():
    print("Loading WineQT dataset...")

    data = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))
    print("Dataset shape:", data.shape)

    # Use chemical properties as input features.
    # Do not use Id or quality as input features.
    features = data.drop(columns=["quality", "Id"], errors="ignore").columns.tolist()

    X = data[features]
    y = data["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Quality classes:", sorted(y.unique().tolist()))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("Training Wine Quality model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("\nConfusion Matrix:")
    print(matrix)

    model_path = BASE_DIR / "wine_quality_model.pkl"
    metrics_path = BASE_DIR / "metrics.json"

    joblib.dump(model, model_path)
    print("\nModel saved as wine_quality_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "quality_classes": [int(value) for value in model.classes_],
        "features": features
    }

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")

    return accuracy


if __name__ == "__main__":
    train_model()
