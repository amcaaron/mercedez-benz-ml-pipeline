from sklearn.model_selection import train_test_split

from src.load_data import load_data
from src.preprocessing import prepare_data
from src.dimensionality_reduction import apply_pca
from src.tune_model import tune_xgboost
from src.evaluate import evaluate_model


train_df, test_df = load_data()

X_train, X_test, y, test_ids = prepare_data(train_df, test_df)

X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test, n_components=50)

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_pca,
    y,
    test_size=0.2,
    random_state=42
)

best_model, best_params, best_cv_score = tune_xgboost(
    X_train_final,
    y_train_final
)

r2, rmse, mae = evaluate_model(best_model, X_val, y_val)

print("Best Parameters:")
print(best_params)

print("Best CV Score:")
print(best_cv_score)

print("Validation R2:", r2)
print("Validation RMSE:", rmse)
print("Validation MAE:", mae)

with open("outputs/tuning_results.txt", "w") as file:
    file.write("Best Parameters:\n")
    file.write(str(best_params))
    file.write("\n\nBest CV Score:\n")
    file.write(str(best_cv_score))
    file.write("\n\nValidation Metrics:\n")
    file.write(f"R2: {r2}\n")
    file.write(f"RMSE: {rmse}\n")
    file.write(f"MAE: {mae}\n")