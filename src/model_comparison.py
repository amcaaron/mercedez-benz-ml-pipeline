import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


def compare_models(X_train, X_val, y_train, y_val):
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    }

    results = []

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        r2 = r2_score(y_val, predictions)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        mae = mean_absolute_error(y_val, predictions)

        results.append({
            "Model": model_name,
            "R2 Score": r2,
            "RMSE": rmse,
            "MAE": mae
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/model_comparison.csv", index=False)

    return results_df