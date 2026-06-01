import joblib
import pandas as pd
import requests

from src.load_data import load_data
from src.preprocessing import prepare_data


train_df, test_df = load_data()
X_train, X_test, y, test_ids = prepare_data(train_df, test_df)

sample_row = X_test.iloc[0].to_dict()

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "features": sample_row
    }
)

print("Status Code:", response.status_code)
print("Response:")
print(response.json())