from load_data import load_data


train_df, test_df = load_data()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain columns:")
print(train_df.columns)

print("\nMissing values in train:")
print(train_df.isnull().sum().sum())

print("\nMissing values in test:")
print(test_df.isnull().sum().sum())

print("\nTarget summary:")
print(train_df["y"].describe())

print("\nCategorical columns:")
print(train_df.select_dtypes(include=["object"]).columns.tolist())