# Mercedes-Benz Manufacturing Optimization ML Pipeline

## Project Overview

This project is an end-to-end machine learning pipeline designed to predict the amount of time a Mercedes-Benz vehicle spends on the test bench during manufacturing. The goal is to use manufacturing configuration data to improve testing efficiency, reduce production bottlenecks, and support data-driven optimization.

The project uses a high-dimensional tabular dataset containing anonymized vehicle configuration features. The final model was trained using XGBoost after comparing multiple machine learning models, performing dimensionality reduction experiments, tuning hyperparameters, and adding explainability through SHAP analysis.

This project also includes a Streamlit dashboard for visualizing results and a FastAPI endpoint for serving real-time predictions.

---

## Goals

The main goals of this project were to:

* Build a complete machine learning pipeline from raw CSV data to final predictions
* Clean and preprocess high-dimensional manufacturing data
* Compare multiple regression models
* Tune model hyperparameters for improved performance
* Evaluate whether PCA dimensionality reduction improves model performance
* Add model explainability using feature importance and SHAP values
* Build a Streamlit dashboard to present results
* Create a FastAPI prediction endpoint to serve the trained model

---

## Tech Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* XGBoost
* SHAP

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Streamlit

### API Development

* FastAPI
* Uvicorn

### Model Persistence

* Joblib

---

## Project Structure

```text
MercedezBenzProject/
│
├── api/
│   └── app.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── raw/
│       ├── train.csv
│       └── test.csv
│
├── models/
│   ├── xgboost_model.pkl
│   └── feature_columns.pkl
│
├── outputs/
│   ├── predictions.csv
│   ├── model_metrics.txt
│   ├── model_comparison.csv
│   ├── pca_comparison.csv
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── tuning_results.txt
│
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── dimensionality_reduction.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── model_comparison.py
│   ├── visualizations.py
│   ├── shap_analysis.py
│   └── tune_model.py
│
├── main.py
├── run_tuning.py
├── run_pca_comparison.py
├── test_api_request.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The project uses two CSV files:

```text
data/raw/train.csv
data/raw/test.csv
```

The training dataset contains the target column:

```text
y
```

The target variable represents the vehicle testing time. The remaining columns represent anonymized Mercedes-Benz manufacturing configuration features.

The test dataset contains the same feature structure but does not include the target column. The trained model is used to generate predictions for this test dataset.

---

## Machine Learning Workflow

The project follows this machine learning workflow:

```text
Load Data
↓
Check Missing Values
↓
Separate Target and Features
↓
Remove ID Column
↓
Encode Categorical Features
↓
Remove Zero-Variance Features
↓
Train/Test Validation Split
↓
Compare Models
↓
Tune XGBoost
↓
Evaluate PCA vs No PCA
↓
Train Final Model
↓
Generate Predictions
↓
Save Model and Outputs
↓
Build Dashboard and API
```

---

## Data Preprocessing

The preprocessing phase includes:

* Loading `train.csv` and `test.csv`
* Separating the target variable `y`
* Dropping the `ID` column from training features
* Saving test IDs for prediction output
* Label encoding categorical variables
* Removing zero-variance columns
* Aligning train and test feature columns

Zero-variance columns were removed because they provide no useful information to the model.

Categorical features were label encoded so that machine learning models could process them numerically.

---

## Models Compared

The project compares the following regression models:

| Model                       | Description                     |
| --------------------------- | ------------------------------- |
| Random Forest Regressor     | Baseline ensemble model         |
| Gradient Boosting Regressor | Boosting-based comparison model |
| XGBoost Regressor           | Final optimized model candidate |

---

## Model Comparison Results

| Model             | R² Score |  RMSE |   MAE |
| ----------------- | -------: | ----: | ----: |
| Random Forest     |    0.460 | 9.171 | 6.340 |
| Gradient Boosting |    0.521 | 8.637 | 5.844 |
| XGBoost Baseline  |    0.485 | 8.954 | 6.058 |
| Tuned XGBoost     |    0.602 | 7.871 | 5.242 |

The tuned XGBoost model achieved the best overall performance and was selected as the final model.

---

## Hyperparameter Tuning

XGBoost was tuned using `RandomizedSearchCV`.

The best parameters found were:

```python
{
    "subsample": 0.9,
    "n_estimators": 300,
    "max_depth": 2,
    "learning_rate": 0.03,
    "colsample_bytree": 0.7
}
```

Final tuned XGBoost performance:

| Metric   | Score |
| -------- | ----: |
| R² Score | 0.602 |
| RMSE     | 7.871 |
| MAE      | 5.242 |

The tuning process improved the model compared to the baseline XGBoost model.

---

## PCA vs No PCA Experiment

A key experiment in this project was testing whether PCA dimensionality reduction improved model performance.

Two pipelines were compared:

```text
Pipeline 1:
Preprocessed Data → PCA → XGBoost

Pipeline 2:
Preprocessed Data → XGBoost
```

### Results

| Pipeline            | R² Score |  RMSE |   MAE |
| ------------------- | -------: | ----: | ----: |
| XGBoost + PCA       |    0.547 | 8.394 | 5.738 |
| XGBoost Without PCA |    0.602 | 7.871 | 5.242 |

The no-PCA pipeline performed better across all major evaluation metrics.

### Conclusion

XGBoost performed better without PCA. This suggests that PCA removed useful information from the original manufacturing configuration features.

Since tree-based models such as XGBoost can naturally handle high-dimensional feature spaces and nonlinear feature interactions, the final model was trained using the original processed features instead of PCA-transformed features.

---

## Model Explainability

This project includes two forms of model explainability:

### Feature Importance

The feature importance chart identifies which features contributed most strongly to the XGBoost model.

Output file:

```text
outputs/feature_importance.png
```

### SHAP Explainability

SHAP was used to explain how individual features influenced the model predictions.

Output file:

```text
outputs/shap_summary.png
```

SHAP adds interpretability by showing both the direction and magnitude of each feature's impact on predictions.

---

## Streamlit Dashboard

The project includes a Streamlit dashboard for presenting the model results visually.

The dashboard displays:

* Final model metrics
* Model comparison results
* PCA vs no-PCA experiment results
* Feature importance chart
* SHAP summary chart
* Prediction output preview

### Run the Dashboard

From the project root folder, run:

```bash
streamlit run dashboard/app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## FastAPI Prediction API

The project includes a FastAPI backend that serves the trained model through a prediction endpoint.

### API Features

| Endpoint        | Description                                |
| --------------- | ------------------------------------------ |
| `GET /`         | Confirms the API is running                |
| `GET /health`   | Checks API health and model loading status |
| `GET /features` | Returns the required feature columns       |
| `POST /predict` | Returns a predicted vehicle test time      |

### Run the API

From the project root folder, run:

```bash
python -m uvicorn api.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The `/docs` page provides an interactive Swagger UI for testing the API.

---

## API Testing

The project includes a test script:

```text
test_api_request.py
```

This script:

* Loads the dataset
* Preprocesses the test data
* Selects one sample row
* Sends it to the FastAPI `/predict` endpoint
* Prints the predicted test time

### Run API Test

First, make sure the API is running:

```bash
python -m uvicorn api.app:app --reload
```

Then open a second terminal and run:

```bash
python test_api_request.py
```

Expected response:

```text
Status Code: 200
Response:
{'predicted_test_time': ...}
```

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd MercedezBenzProject
```

### 2. Create a Virtual Environment

Optional but recommended:

```bash
python -m venv venv
```

Activate it:

#### Mac/Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Dataset Files

Place the dataset files in:

```text
data/raw/train.csv
data/raw/test.csv
```

### 5. Run the Main Pipeline

```bash
python main.py
```

This will:

* Train the final model
* Evaluate model performance
* Save model files
* Generate predictions
* Generate feature importance
* Generate SHAP explainability chart

### 6. Run Hyperparameter Tuning

```bash
python run_tuning.py
```

### 7. Run PCA Comparison

```bash
python run_pca_comparison.py
```

### 8. Run Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

### 9. Run FastAPI

```bash
python -m uvicorn api.app:app --reload
```

---

## Outputs

The project generates the following outputs:

| File                             | Description                                                 |
| -------------------------------- | ----------------------------------------------------------- |
| `outputs/predictions.csv`        | Final predictions on test data                              |
| `outputs/model_metrics.txt`      | Final model evaluation results                              |
| `outputs/model_comparison.csv`   | Comparison of Random Forest, Gradient Boosting, and XGBoost |
| `outputs/pca_comparison.csv`     | PCA vs no-PCA experiment results                            |
| `outputs/feature_importance.png` | XGBoost feature importance chart                            |
| `outputs/shap_summary.png`       | SHAP explainability summary plot                            |
| `outputs/tuning_results.txt`     | Hyperparameter tuning results                               |

---

## Results Summary

The final selected model was a tuned XGBoost regressor trained without PCA.

Final model performance:

| Metric   | Score |
| -------- | ----: |
| R² Score | 0.602 |
| RMSE     | 7.871 |
| MAE      | 5.242 |

The final model outperformed the PCA version and the baseline models.

---

## Key Takeaways

* XGBoost performed best after hyperparameter tuning.
* PCA reduced performance for this dataset.
* Tree-based models benefited from retaining the original processed feature space.
* SHAP and feature importance added interpretability to the final model.
* Streamlit made the results easier to present visually.
* FastAPI allowed the trained model to be served through a prediction endpoint.

---

## Business Value

This project demonstrates how machine learning can support manufacturing optimization by predicting vehicle testing time based on configuration data.

Potential business benefits include:

* Reducing test bench bottlenecks
* Improving production scheduling
* Identifying features that contribute to longer testing times
* Supporting data-driven manufacturing decisions
* Increasing efficiency in the vehicle testing process

---

## Future Improvements

Future improvements could include:

* Saving the full preprocessing pipeline so the API can accept raw input data
* Adding automated unit tests for preprocessing and prediction logic
* Deploying the FastAPI backend to a cloud platform
* Deploying the Streamlit dashboard online
* Adding MLflow for experiment tracking
* Testing additional models such as LightGBM or CatBoost
* Adding Docker support for easier deployment
* Improving API validation for raw manufacturing configuration inputs
* Adding CI/CD with GitHub Actions

---

**Mercedes-Benz Manufacturing Optimization ML Pipeline**

Built an end-to-end machine learning pipeline using Python, XGBoost, SHAP, Streamlit, and FastAPI to predict Mercedes-Benz vehicle test bench time from high-dimensional manufacturing configuration data.


## Author

Aaron Cole
Computer Science Graduate
Interested in Software Engineering, AI Development, Machine Learning, and Quantitative Analysis
