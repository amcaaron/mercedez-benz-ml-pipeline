import os
import joblib
import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PIPELINE_PATH = os.path.join(BASE_DIR, "models", "mercedes_pipeline.joblib")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")


def test_pipeline_file_exists():
    assert os.path.exists(PIPELINE_PATH)


def test_train_data_exists():
    assert os.path.exists(TRAIN_DATA_PATH)


def test_pipeline_can_make_prediction():
    pipeline = joblib.load(PIPELINE_PATH)

    df = pd.read_csv(TRAIN_DATA_PATH)

    X = df.drop(columns=["y", "ID"], errors="ignore")

    sample = X.iloc[[0]]

    prediction = pipeline.predict(sample)

    assert len(prediction) == 1
    assert isinstance(float(prediction[0]), float)
    assert prediction[0] > 0