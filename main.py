import joblib
from sklearn.model_selection import train_test_split

from src.load_data import load_data
from src.preprocessing import prepare_data
from src.train import train_xgboost
from src.evaluate import evaluate_model
from src.predict import create_submission
from src.visualizations import save_feature_importance
from src.shap_analysis import generate_shap_summary

def main():
    print("Loading data...")
    train_df, test_df = load_data()

    print("Preprocessing data...")
    X_train, X_test, y, test_ids = prepare_data(train_df, test_df)

    print("Splitting data...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training final XGBoost model WITHOUT PCA...")
    model = train_xgboost(X_train_final, y_train_final)

    print("Evaluating model...")
    r2, rmse, mae = evaluate_model(model, X_val, y_val)

    print("R2 Score:", r2)
    print("RMSE:", rmse)
    print("MAE:", mae)

    with open("outputs/model_metrics.txt", "w") as file:
        file.write("Final Model: Tuned XGBoost Without PCA\n")
        file.write(f"R2 Score: {r2}\n")
        file.write(f"RMSE: {rmse}\n")
        file.write(f"MAE: {mae}\n")

    print("Saving model and feature columns...")
    joblib.dump(model, "models/xgboost_model.pkl")
    joblib.dump(X_train.columns.tolist(), "models/feature_columns.pkl")

    print("Saving feature importance...")
    save_feature_importance(model)
    print("Generating SHAP explainability chart...")
    generate_shap_summary(model, X_train_final)

    print("Creating predictions...")
    create_submission(model, X_test, test_ids)

    print("Project completed successfully!")


if __name__ == "__main__":
    main()