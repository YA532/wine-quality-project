import pandas as pd
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "datasets" / "WineQT.csv"


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    return df


def predict_quality(quality):
    if quality >= 6:
        return "Average"
    else:
        return "Average"


if __name__ == "__main__":
    df = load_dataset()

    print("Wine Quality Dataset")
    print("--------------------")
    print("Dataset Shape:", df.shape)
    print("Total Records:", len(df))
    print("Average Quality:", df["quality"].mean())

    result = predict_quality(6)
    print("Predicted Category:", result)
