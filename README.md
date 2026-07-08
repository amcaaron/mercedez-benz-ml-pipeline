# Mercedes-Benz Manufacturing Optimization ML Pipeline

## Project Overview

This project is an end-to-end machine learning pipeline designed to predict the amount of time a Mercedes-Benz vehicle spends on the test bench during manufacturing. The goal is to use high-dimensional manufacturing configuration data to improve testing efficiency, reduce production bottlenecks, and support data-driven manufacturing optimization.

The project includes data preprocessing, model comparison, hyperparameter tuning, PCA experimentation, TensorFlow deep learning experimentation, SHAP explainability, a Streamlit dashboard, a FastAPI prediction API, Docker support, automated Pytest testing, and GitHub Actions CI/CD.

---

## Key Features

* End-to-end machine learning pipeline from raw CSV data to predictions
* High-dimensional tabular data preprocessing
* Model comparison across Random Forest, Gradient Boosting, XGBoost, and TensorFlow
* Hyperparameter tuning with `RandomizedSearchCV`
* PCA vs no-PCA performance comparison
* TensorFlow neural network regression experiment
* SHAP explainability and feature importance visualization
* Saved full preprocessing and model pipeline using Joblib
* Saved TensorFlow model using Keras format
* FastAPI prediction API that accepts raw feature input
* Streamlit dashboard for visualizing model results
* Automated unit and API tests with Pytest
* Dockerized API environment
* GitHub Actions CI workflow for automated testing

---

## Screenshots
![Table Comparison](screenshots/summary.png) 
![Docker Run](screenshots/docker_run.png)


## Tech Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* XGBoost
* TensorFlow
* SHAP
* Joblib

### Data Processing

* Pandas
* NumPy

### API Development

* FastAPI
* Uvicorn
* Pydantic

### Testing and DevOps

* Pytest
* Docker
* GitHub Actions

### Visualization

* Matplotlib
* Streamlit

---

## Project Structure

```text
MercedezBenzProject/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── api/
│   ├── __init__.py
│   └── app.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── raw/
│       └── train.csv
│
├── models/
│   ├── mercedes_pipeline.joblib
│   └── tensorflow_model.keras
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
│   ├── train_pipeline.py
│   ├── train_tensorflow.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── model_comparison.py
│   ├── visualizations.py
│   ├── shap_analysis.py
│   └── tune_model.py
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_pipeline.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── main.py
├── run_no_pca.py
├── run_pca_comparison.py
├── run_tuning.py
├── manual_api_request.py
├── manual_pipeline_api.py
├── requirements.txt
├── requirements-api.txt
└── README.md
```

---

## Dataset

The project uses the Mercedes-Benz Greener Manufacturing dataset.

The main training file is stored at:

```text
data/raw/train.csv
```

The target column is:

```text
y
```

The target variable represents the amount of time a vehicle spends on the test bench. The remaining columns represent anonymized vehicle configuration features.

---

## Machine Learning Workflow

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
Compare Traditional ML Models
↓
Tune XGBoost
↓
Evaluate PCA vs No PCA
↓
Train TensorFlow Neural Network
↓
Compare Deep Learning vs Traditional ML
↓
Train Final XGBoost Model
↓
Save Full Preprocessing + Model Pipeline
↓
Generate Predictions
↓
Build Dashboard
↓
Serve Model with FastAPI
↓
Test with Pytest
↓
Validate with GitHub Actions CI
```

---

## Data Preprocessing

The preprocessing phase includes:

* Loading the raw training CSV file
* Separating the target variable `y`
* Dropping the `ID` column
* Handling categorical manufacturing features
* Removing zero-variance columns
* Preparing data for model training
* Saving a complete preprocessing and model pipeline for API inference

The final production pipeline uses Scikit-learn’s `Pipeline` and `ColumnTransformer` so the FastAPI app can accept raw feature input instead of requiring manually preprocessed data.

---

## Models Compared

| Model                       | Description                         |
| --------------------------- | ----------------------------------- |
| Random Forest Regressor     | Baseline ensemble model             |
| Gradient Boosting Regressor | Boosting-based comparison model     |
| XGBoost Regressor           | Final optimized model candidate     |
| TensorFlow Neural Network   | Deep learning regression experiment |

---

## Model Comparison Results

| Model                     | R² Score |  RMSE |   MAE |
| ------------------------- | -------: | ----: | ----: |
| Random Forest             |    0.460 | 9.171 | 6.340 |
| Gradient Boosting         |    0.521 | 8.637 | 5.844 |
| XGBoost Baseline          |    0.485 | 8.954 | 6.058 |
| TensorFlow Neural Network |    0.520 | 8.643 | 5.880 |
| Tuned XGBoost             |    0.602 | 7.871 | 5.242 |

The tuned XGBoost model achieved the best overall performance and was selected as the final production model. The TensorFlow neural network was added as a deep learning experiment to compare neural network performance against traditional machine learning models on high-dimensional manufacturing data.

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

## TensorFlow Deep Learning Experiment

A TensorFlow neural network was added to test whether a deep learning model could improve prediction accuracy on the Mercedes-Benz manufacturing dataset.

The TensorFlow model used:

* Dense neural network layers
* ReLU activation functions
* Batch normalization
* Dropout regularization
* Adam optimizer
* Mean squared error loss
* Early stopping to reduce overfitting

The model was trained as a regression model to predict the vehicle test bench time.

### TensorFlow Results

| Metric   | Score |
| -------- | ----: |
| R² Score | 0.520 |
| RMSE     | 8.643 |
| MAE      | 5.880 |

The TensorFlow model performed competitively with Gradient Boosting but did not outperform the tuned XGBoost model. This result is useful because it shows that multiple modeling approaches were tested and that the final model choice was based on measurable performance rather than assumption.

### Run TensorFlow Training

```bash
python src/train_tensorflow.py
```

### Saved TensorFlow Model

After training, the TensorFlow model is saved to:

```text
models/tensorflow_model.keras
```

This `.keras` file is a saved model file and is not meant to be opened directly in a text editor.

To test that the saved TensorFlow model loads correctly, run:

```bash
python -c "import tensorflow as tf; model=tf.keras.models.load_model('models/tensorflow_model.keras'); print('Model loaded successfully')"
```

Expected output:

```text
Model loaded successfully
```

---

## PCA vs No PCA Experiment

A key experiment in this project was testing whether PCA dimensionality reduction improved performance.

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

### Conclusion

XGBoost performed better without PCA. This suggests that PCA removed useful information from the original manufacturing configuration features.

Since tree-based models such as XGBoost can naturally handle high-dimensional feature spaces and nonlinear feature interactions, the final model was trained using the original processed features instead of PCA-transformed features.

---

## Model Explainability

This project includes two forms of model explainability.

### Feature Importance

The feature importance chart identifies which features contributed most strongly to the XGBoost model.

Output file:

```text
outputs/feature_importance.png
```

### SHAP Explainability

SHAP was used to explain how individual features influenced model predictions.

Output file:

```text
outputs/shap_summary.png
```

SHAP adds interpretability by showing both the direction and magnitude of each feature’s impact on predictions.

---

## Saved Production Pipeline

The final production pipeline is saved as:

```text
models/mercedes_pipeline.joblib
```

This file contains both:

* Preprocessing logic
* Tuned XGBoost model

This allows the FastAPI app to accept raw feature values and run the same preprocessing steps used during training before generating a prediction.

To retrain and save the production pipeline, run:

```bash
python src/train_pipeline.py
```

---

## FastAPI Prediction API

The project includes a FastAPI backend that serves the saved machine learning pipeline.

### API Endpoints

| Endpoint    | Method | Description                                   |
| ----------- | ------ | --------------------------------------------- |
| `/`         | GET    | Confirms the API is running                   |
| `/health`   | GET    | Checks API health and pipeline loading status |
| `/features` | GET    | Returns the required raw feature columns      |
| `/predict`  | POST   | Returns a predicted vehicle test bench time   |

### Run the API Locally

From the project root folder, run:

```bash
python -m uvicorn api.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The `/docs` page provides an interactive Swagger UI for testing the API.

### Example API Request

```json
{
  "features": {
    "X0": "k",
    "X1": "v",
    "X2": "at"
  }
}
```

The full required feature list can be viewed by visiting:

```text
http://127.0.0.1:8000/features
```

---

## Manual API Testing

The project includes manual API scripts for local testing:

```text
manual_api_request.py
manual_pipeline_api.py
```

These scripts are not part of the automated Pytest suite. They are used for manually sending requests to the running FastAPI server.

To run a manual API test:

### 1. Start the API

```bash
python -m uvicorn api.app:app --reload
```

### 2. Open a second terminal and run:

```bash
python manual_pipeline_api.py
```

Expected response:

```text
Status Code: 200
Response:
{"predicted_test_time": ...}
```

---

## Streamlit Dashboard

The project includes a Streamlit dashboard for presenting model results visually.

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

## Automated Testing

The project includes automated tests with Pytest.

Test files are located in:

```text
tests/
```

Current tests validate:

* Saved pipeline file exists
* Training data exists
* Pipeline can generate predictions
* FastAPI root endpoint works
* FastAPI health endpoint works
* FastAPI feature endpoint works
* API handles missing features correctly

### Run Tests Locally

```bash
PYTHONPATH=. python -m pytest
```

Expected result:

```text
7 passed
```

---

## GitHub Actions CI/CD

This project uses GitHub Actions to automatically run tests when code is pushed to the repository.

Workflow file:

```text
.github/workflows/ci.yml
```

The CI workflow:

* Checks out the repository
* Sets up Python
* Installs dependencies
* Verifies required project files
* Runs the Pytest test suite

This helps ensure the machine learning pipeline and API remain stable after future changes.

---

## Docker Support

The FastAPI app can be run inside a Docker container.

### Dockerfile

The project Dockerfile is configured to run the FastAPI app:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build the Docker Image

```bash
docker build -t mercedes-ml-api .
```

### Run the Docker Container

```bash
docker run --rm -p 8000:8000 mercedes-ml-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

### Test TensorFlow Inside Docker

The Dockerfile runs the API by default. To test the TensorFlow training script inside the same Docker image, override the default command:

```bash
docker run --rm mercedes-ml-api python src/train_tensorflow.py
```

This runs the TensorFlow training script inside the container and confirms that the project environment works with TensorFlow installed.

### Important Docker Requirement

Because the project includes a TensorFlow training script, `requirements-api.txt` should include:

```text
tensorflow
```

If TensorFlow is only used locally and not needed inside the API container, it can be kept in `requirements.txt` instead. However, to test TensorFlow with Docker, it must be available in the Docker-installed requirements file.

---

## Requirements

### Main Project Requirements

The main project dependencies are stored in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

### API and Docker Requirements

The API and Docker dependencies are stored in:

```text
requirements-api.txt
```

Install them with:

```bash
pip install -r requirements-api.txt
```

Recommended API/Docker dependencies include:

```text
fastapi
uvicorn
pandas
numpy
scikit-learn
xgboost
joblib
tensorflow
python-multipart
```

If the API or Docker workflow uses SHAP or plots, also include:

```text
shap
matplotlib
```

Seaborn is not required unless your project imports it directly.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/amcaaron/mercedez-benz-ml-pipeline.git
cd mercedez-benz-ml-pipeline
```

### 2. Create a Virtual Environment

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

For the full project:

```bash
pip install -r requirements.txt
```

For the FastAPI/Docker/testing environment:

```bash
pip install -r requirements-api.txt
```

### 4. Run the Main ML Pipeline

```bash
python main.py
```

### 5. Train the Production Pipeline

```bash
python src/train_pipeline.py
```

### 6. Train the TensorFlow Model

```bash
python src/train_tensorflow.py
```

### 7. Run Hyperparameter Tuning

```bash
python run_tuning.py
```

### 8. Run PCA Comparison

```bash
python run_pca_comparison.py
```

### 9. Run the FastAPI App

```bash
python -m uvicorn api.app:app --reload
```

### 10. Run the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

### 11. Run Tests

```bash
PYTHONPATH=. python -m pytest
```

### 12. Build and Run with Docker

```bash
docker build -t mercedes-ml-api .
docker run --rm -p 8000:8000 mercedes-ml-api
```

---

## Outputs

The project generates the following outputs:

| File                              | Description                                                             |
| --------------------------------- | ----------------------------------------------------------------------- |
| `outputs/predictions.csv`         | Final predictions                                                       |
| `outputs/model_metrics.txt`       | Final model evaluation results                                          |
| `outputs/model_comparison.csv`    | Comparison of Random Forest, Gradient Boosting, XGBoost, and TensorFlow |
| `outputs/pca_comparison.csv`      | PCA vs no-PCA experiment results                                        |
| `outputs/feature_importance.png`  | XGBoost feature importance chart                                        |
| `outputs/shap_summary.png`        | SHAP explainability summary plot                                        |
| `outputs/tuning_results.txt`      | Hyperparameter tuning results                                           |
| `models/mercedes_pipeline.joblib` | Saved production preprocessing and XGBoost pipeline                     |
| `models/tensorflow_model.keras`   | Saved TensorFlow neural network model                                   |

---

## Results Summary

The final selected production model was a tuned XGBoost regressor trained without PCA.

Final tuned XGBoost performance:

| Metric   | Score |
| -------- | ----: |
| R² Score | 0.602 |
| RMSE     | 7.871 |
| MAE      | 5.242 |

TensorFlow neural network performance:

| Metric   | Score |
| -------- | ----: |
| R² Score | 0.520 |
| RMSE     | 8.643 |
| MAE      | 5.880 |

The tuned XGBoost model outperformed the PCA version, baseline models, and TensorFlow neural network. The TensorFlow experiment still strengthened the project by demonstrating deep learning experimentation and model benchmarking.

---

## Business Value

This project demonstrates how machine learning can support manufacturing optimization by predicting vehicle testing time based on vehicle configuration data.

Potential business benefits include:

* Reducing test bench bottlenecks
* Improving production scheduling
* Identifying features that contribute to longer testing times
* Supporting data-driven manufacturing decisions
* Increasing efficiency in the vehicle testing process

---

## Key Takeaways

* XGBoost performed best after hyperparameter tuning.
* PCA reduced performance for this dataset.
* Tree-based models benefited from retaining the original processed feature space.
* TensorFlow was added as a deep learning regression experiment.
* TensorFlow performed competitively but did not outperform tuned XGBoost.
* SHAP and feature importance improved interpretability.
* FastAPI made the model available through a prediction API.
* Docker improved environment consistency and deployment readiness.
* Pytest and GitHub Actions added automated validation.
* The project demonstrates both machine learning development and production engineering practices.

---

## Future Improvements

Future improvements could include:

* Deploying the FastAPI backend to a cloud platform
* Deploying the Streamlit dashboard online
* Adding MLflow for experiment tracking
* Testing additional models such as LightGBM or CatBoost
* Improving TensorFlow tuning with architecture search
* Improving API validation with stricter Pydantic schemas
* Adding Docker image builds to GitHub Actions
* Adding monitoring or logging for production API requests
* Adding model versioning for production deployment

---

## Author

**Aaron Cole**
Computer Science Graduate
Interested in Software Engineering, AI Development, Machine Learning, and Quantitative Analysis
