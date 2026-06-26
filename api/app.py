import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "mercedes_pipeline.joblib")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")


app = FastAPI(
    title="Mercedes-Benz ML Prediction API",
    description="API for predicting Mercedes-Benz vehicle test bench time using a saved preprocessing + XGBoost pipeline.",
    version="2.0.0"
)


pipeline = joblib.load(PIPELINE_PATH)

train_df = pd.read_csv(TRAIN_DATA_PATH)
raw_feature_columns = train_df.drop(columns=["y", "ID"], errors="ignore").columns.tolist()


class PredictionInput(BaseModel):
    features: dict


@app.get("/")
def home():
    return {
        "message": "Mercedes-Benz ML Prediction API is running.",
        "version": "2.0.0",
        "pipeline_loaded": True
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "pipeline_loaded": True,
        "number_of_raw_features": len(raw_feature_columns)
    }


@app.get("/features")
def get_required_features():
    return {
        "required_raw_features": raw_feature_columns
    }


@app.post("/predict")
def predict(input_data: PredictionInput):
    input_features = input_data.features

    missing_features = [
        col for col in raw_feature_columns
        if col not in input_features
    ]

    extra_features = [
        col for col in input_features
        if col not in raw_feature_columns
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required raw features.",
                "missing_features": missing_features[:20],
                "missing_count": len(missing_features)
            }
        )

    if extra_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unexpected extra raw features found.",
                "extra_features": extra_features[:20],
                "extra_count": len(extra_features)
            }
        )

    try:
        input_df = pd.DataFrame([input_features])
        input_df = input_df[raw_feature_columns]

        prediction = pipeline.predict(input_df)[0]

        return {
            "predicted_test_time": float(prediction)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction failed.",
                "message": str(e)
            }
        )