import streamlit as st
import pandas as pd
from PIL import Image


st.set_page_config(
    page_title="Mercedes-Benz ML Pipeline",
    layout="wide"
)

st.title("Mercedes-Benz Manufacturing Optimization ML Pipeline")

st.write(
    """
    This dashboard presents the results of a machine learning pipeline designed
    to predict Mercedes-Benz vehicle testing time from manufacturing configuration data.
    """
)

st.header("Final Model Metrics")

try:
    with open("outputs/model_metrics.txt", "r") as file:
        metrics = file.read()

    st.text(metrics)

except FileNotFoundError:
    st.warning("Run main.py first to generate model_metrics.txt.")


st.header("Model Comparison")

try:
    comparison_df = pd.read_csv("outputs/model_comparison.csv")
    st.dataframe(comparison_df)

except FileNotFoundError:
    st.warning("Run main.py or model comparison first to generate model_comparison.csv.")


st.header("PCA vs No PCA Experiment")

try:
    pca_df = pd.read_csv("outputs/pca_comparison.csv")
    st.dataframe(pca_df)

    st.write(
        """
        The no-PCA pipeline performed better, showing that XGBoost benefited
        from retaining the original processed feature space.
        """
    )

except FileNotFoundError:
    st.warning("Run run_pca_comparison.py first to generate pca_comparison.csv.")


st.header("Feature Importance")

try:
    feature_img = Image.open("outputs/feature_importance.png")
    st.image(feature_img, caption="Top Feature Importances")

except FileNotFoundError:
    st.warning("Run main.py first to generate feature_importance.png.")


st.header("SHAP Explainability")

try:
    shap_img = Image.open("outputs/shap_summary.png")
    st.image(shap_img, caption="SHAP Summary Plot")

except FileNotFoundError:
    st.warning("Run main.py first to generate shap_summary.png.")


st.header("Prediction Output Preview")

try:
    predictions_df = pd.read_csv("outputs/predictions.csv")
    st.dataframe(predictions_df.head(20))

except FileNotFoundError:
    st.warning("Run main.py first to generate predictions.csv.")