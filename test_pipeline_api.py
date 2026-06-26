import requests
import pandas as pd

df = pd.read_csv("data/raw/train.csv")

row = df.drop(columns=["y", "ID"], errors="ignore").iloc[0].to_dict()

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": row}
)

print(response.status_code)
print(response.json())