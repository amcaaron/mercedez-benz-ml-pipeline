import pandas as pd


def create_submission(model, X_test, test_ids, output_path="outputs/predictions.csv"):
    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        "ID": test_ids,
        "y": predictions
    })

    submission.to_csv(output_path, index=False)

    return submission