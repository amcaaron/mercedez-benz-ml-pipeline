import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

from xgboost import XGBRegressor


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "models", "mercedes_pipeline.joblib")


def load_data(path):
    df = pd.read_csv(path)
    return df


def split_features_target(df):
    X = df.drop(columns=["y"])
    y = df["y"]

    if "ID" in X.columns:
        X = X.drop(columns=["ID"])

    return X, y


def build_pipeline(X):
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.7,
        random_state=42,
        objective="reg:squarederror"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("Model Evaluation")
    print("----------------")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")


def main():
    os.makedirs("models", exist_ok=True)

    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline = build_pipeline(X_train)

    pipeline.fit(X_train, y_train)

    evaluate_model(pipeline, X_test, y_test)

    joblib.dump(pipeline, MODEL_OUTPUT_PATH)

    print(f"Saved full preprocessing + model pipeline to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()