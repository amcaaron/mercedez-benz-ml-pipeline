import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


# 1. Load data
df = pd.read_csv("data/raw/train.csv")

# 2. Separate features and target
X = df.drop(columns=["y"])
y = df["y"]

# 3. Drop ID column if it exists
if "ID" in X.columns:
    X = X.drop(columns=["ID"])

# 4. Encode categorical columns
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# 5. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# 7. Build TensorFlow model
def build_tensorflow_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="relu"),

        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model


model = build_tensorflow_model(X_train.shape[1])

# 8. Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

# 9. Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 10. Evaluate model
predictions = model.predict(X_val).flatten()

r2 = r2_score(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))
mae = mean_absolute_error(y_val, predictions)

print("TensorFlow Neural Network Results")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# 11. Save model
print("\n" + "=" * 50)
print("TensorFlow Neural Network Results")
print("=" * 50)
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print("=" * 50)

model.save("models/tensorflow_model.keras")
print("Model saved to: models/tensorflow_model.keras")