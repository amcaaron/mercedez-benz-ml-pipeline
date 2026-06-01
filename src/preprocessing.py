import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold


def remove_zero_variance(X_train, X_test):
    selector = VarianceThreshold(threshold=0)
    selector.fit(X_train)

    selected_columns = X_train.columns[selector.get_support()]

    X_train = X_train[selected_columns]
    X_test = X_test[selected_columns]

    return X_train, X_test


def encode_categorical_features(X_train, X_test):
    categorical_cols = X_train.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        encoder = LabelEncoder()

        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        encoder.fit(combined)

        X_train[col] = encoder.transform(X_train[col])
        X_test[col] = encoder.transform(X_test[col])

    return X_train, X_test


def prepare_data(train_df, test_df):
    y = train_df["y"]

    X_train = train_df.drop(["y"], axis=1)
    X_test = test_df.copy()

    if "ID" in X_train.columns:
        X_train = X_train.drop(["ID"], axis=1)

    test_ids = X_test["ID"]

    if "ID" in X_test.columns:
        X_test = X_test.drop(["ID"], axis=1)

    X_train, X_test = encode_categorical_features(X_train, X_test)
    X_train, X_test = remove_zero_variance(X_train, X_test)

    return X_train, X_test, y, test_ids