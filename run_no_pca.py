from sklearn.model_selection import train_test_split

from src.load_data import load_data
from src.preprocessing import prepare_data
from src.tune_model import tune_xgboost
from src.evaluate import evaluate_model


print("Loading data...")

train_df, test_df = load_data()

print("Preparing data...")

X_train, X_test, y, test_ids = prepare_data(
    train_df,
    test_df
)

print("Splitting data...")

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train,
    y,
    test_size=0.2,
    random_state=42
)

print("Training tuned XGBoost WITHOUT PCA...")

best_model, best_params, best_cv_score = tune_xgboost(
    X_train_final,
    y_train_final
)

r2, rmse, mae = evaluate_model(
    best_model,
    X_val,
    y_val
)

print("\n===== RESULTS WITHOUT PCA =====")
print("Best Parameters:")
print(best_params)

print("\nBest CV Score:")
print(best_cv_score)

print("\nValidation Metrics")
print("R2:", r2)
print("RMSE:", rmse)
print("MAE:", mae)