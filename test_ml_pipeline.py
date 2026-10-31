import json
import unittest
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "datasets" / "WineQT.csv"
MODEL_PATH = BASE_DIR / "wine_quality_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(DATASET_PATH.exists())

        data = pd.read_csv(DATASET_PATH)
        self.assertEqual(len(data), 1143)
        self.assertIn("quality", data.columns)

    def test_model_created(self):
        self.assertTrue(MODEL_PATH.exists())

    def test_metrics_created(self):
        self.assertTrue(METRICS_PATH.exists())

    def test_accuracy_is_valid(self):
        with open(METRICS_PATH, "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        data = pd.read_csv(DATASET_PATH)
        features = data.drop(
            columns=["quality", "Id"],
            errors="ignore"
        ).columns.tolist()

        model = joblib.load(MODEL_PATH)
        sample = data[features].median().to_frame().T

        prediction = model.predict(sample)[0]
        valid_classes = [int(value) for value in model.classes_]

        self.assertIn(int(prediction), valid_classes)

    def test_high_alcohol_wine_prediction(self):
        data = pd.read_csv(DATASET_PATH)
        features = data.drop(
            columns=["quality", "Id"],
            errors="ignore"
        ).columns.tolist()

        model = joblib.load(MODEL_PATH)
        sample = data[features].median().to_frame().T
        sample["alcohol"] = data["alcohol"].max()

        prediction = model.predict(sample)[0]
        valid_classes = [int(value) for value in model.classes_]

        self.assertIn(int(prediction), valid_classes)

    def test_low_alcohol_wine_prediction(self):
        data = pd.read_csv(DATASET_PATH)
        features = data.drop(
            columns=["quality", "Id"],
            errors="ignore"
        ).columns.tolist()

        model = joblib.load(MODEL_PATH)
        sample = data[features].median().to_frame().T
        sample["alcohol"] = data["alcohol"].min()

        prediction = model.predict(sample)[0]
        valid_classes = [int(value) for value in model.classes_]

        self.assertIn(int(prediction), valid_classes)


if __name__ == "__main__":
    unittest.main()
