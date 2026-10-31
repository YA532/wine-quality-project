
import unittest
from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_valid_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "fixed acidity": 7.4,
                "volatile acidity": 0.70,
                "citric acid": 0.00,
                "residual sugar": 1.9,
                "chlorides": 0.076,
                "free sulfur dioxide": 11,
                "total sulfur dioxide": 34,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], 999)

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={"fixed acidity": 7.4}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())

    def test_empty_request(self):
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
