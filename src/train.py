from xgboost import XGBRegressor


def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        subsample=0.9,
        n_estimators=300,
        max_depth=2,
        learning_rate=0.03,
        colsample_bytree=0.7,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model