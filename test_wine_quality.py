import unittest
from wine_quality import load_dataset, predict_quality


class TestWineQuality(unittest.TestCase):

    def test_good_quality(self):
        self.assertEqual(predict_quality(7), "Good")

    def test_average_quality(self):
        self.assertEqual(predict_quality(5), "Average")

    def test_boundary_quality(self):
        self.assertEqual(predict_quality(6), "Good")

    def test_dataset_rows(self):
        df = load_dataset()
        self.assertEqual(len(df), 1143)

    def test_dataset_columns(self):
        df = load_dataset()
        self.assertIn("quality", df.columns)

    def test_missing_values(self):
        df = load_dataset()
        self.assertEqual(df.isnull().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
