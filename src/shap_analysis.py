import shap
import matplotlib.pyplot as plt


def generate_shap_summary(model, X_train, output_path="outputs/shap_summary.png"):
    # Use a smaller sample so SHAP runs faster
    if len(X_train) > 100:
        X_sample = X_train.sample(
            n=100,
            random_state=42
        )
    else:
        X_sample = X_train

    # SHAP requires at least 2 * number_of_features + 1 evaluations
    max_evals = 2 * X_sample.shape[1] + 1

    explainer = shap.Explainer(
        model.predict,
        X_sample
    )

    shap_values = explainer(
        X_sample,
        max_evals=max_evals
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=20
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        bbox_inches="tight"
    )
    plt.close()