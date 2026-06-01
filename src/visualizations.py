import matplotlib.pyplot as plt
from xgboost import plot_importance
import numpy as np


def save_feature_importance(model):
    plt.figure(figsize=(10, 8))

    plot_importance(
        model,
        max_num_features=20,
        height=0.5
    )

    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()


def save_pca_variance_plot(pca):
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(10, 6))
    plt.plot(explained_variance)
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/pca_variance.png")
    plt.close()