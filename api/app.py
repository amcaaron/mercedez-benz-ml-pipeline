import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(
    title="Mercedes-Benz ML Prediction API",
    description="API for predicting Mercedes-Benz vehicle test bench time.",
    version="1.0.0"
)


model = joblib.load("models/xgboost_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


class PredictionInput(BaseModel):
    features: dict


@app.get("/")
def home():
    return {
        "message": "Mercedes-Benz ML Prediction API is running."
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "number_of_features": len(feature_columns)
    }


@app.get("/features")
def get_required_features():
    return {
        "required_features": feature_columns
    }


@app.post("/predict")
def predict(input_data: PredictionInput):
    input_features = input_data.features

    missing_features = [
        col for col in feature_columns
        if col not in input_features
    ]

    extra_features = [
        col for col in input_features
        if col not in feature_columns
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required features.",
                "missing_features": missing_features[:20],
                "missing_count": len(missing_features)
            }
        )

    if extra_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unexpected extra features found.",
                "extra_features": extra_features[:20],
                "extra_count": len(extra_features)
            }
        )

    input_df = pd.DataFrame([input_features])
    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]

    return {
        "predicted_test_time": float(prediction)
    }