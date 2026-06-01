import pandas as pd
from sklearn.model_selection import train_test_split

from src.load_data import load_data
from src.preprocessing import prepare_data
from src.dimensionality_reduction import apply_pca
from src.train import train_xgboost
from src.evaluate import evaluate_model


def main():
    print("Loading data...")
    train_df, test_df = load_data()

    print("Preprocessing data...")
    X_train, X_test, y, test_ids = prepare_data(train_df, test_df)

    results = []

    # -----------------------------
    # Pipeline 1: XGBoost WITH PCA
    # -----------------------------
    print("Running XGBoost WITH PCA...")

    X_train_pca, X_test_pca, pca = apply_pca(
        X_train,
        X_test,
        n_components=50
    )

    X_train_pca_final, X_val_pca, y_train_pca_final, y_val_pca = train_test_split(
        X_train_pca,
        y,
        test_size=0.2,
        random_state=42
    )

    model_pca = train_xgboost(X_train_pca_final, y_train_pca_final)

    r2_pca, rmse_pca, mae_pca = evaluate_model(
        model_pca,
        X_val_pca,
        y_val_pca
    )

    results.append({
        "Pipeline": "XGBoost + PCA",
        "R2": r2_pca,
        "RMSE": rmse_pca,
        "MAE": mae_pca
    })

    # -----------------------------
    # Pipeline 2: XGBoost WITHOUT PCA
    # -----------------------------
    print("Running XGBoost WITHOUT PCA...")

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train,
        y,
        test_size=0.2,
        random_state=42
    )

    model_no_pca = train_xgboost(X_train_final, y_train_final)

    r2_no_pca, rmse_no_pca, mae_no_pca = evaluate_model(
        model_no_pca,
        X_val,
        y_val
    )

    results.append({
        "Pipeline": "XGBoost Without PCA",
        "R2": r2_no_pca,
        "RMSE": rmse_no_pca,
        "MAE": mae_no_pca
    })

    # Save results
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "outputs/pca_comparison.csv",
        index=False
    )

    with open("outputs/pca_comparison.txt", "w") as file:
        file.write("PCA vs No PCA Comparison\n\n")
        file.write(results_df.to_string(index=False))
        file.write("\n\nConclusion:\n")

        if r2_no_pca > r2_pca:
            file.write(
                "XGBoost performed better without PCA. "
                "This suggests that PCA removed useful feature information, "
                "so the final model should use the original processed features.\n"
            )
        else:
            file.write(
                "XGBoost performed better with PCA. "
                "This suggests that dimensionality reduction helped reduce noise "
                "and improved generalization.\n"
            )

    print("\nComparison complete!")
    print(results_df)


if __name__ == "__main__":
    main()