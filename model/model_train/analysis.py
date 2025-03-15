import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Load Cleveland dataset
data_clv = pd.read_csv("../datasets/processed.cleveland.data.csv")

# Print unique values for each column
for col in data_clv.columns:
    print(data_clv[col].unique())

# Drop specific columns
data_clv.drop(['ca','thal'], axis=1, inplace=True)
print(data_clv.head())
print(data_clv.columns)

# Convert target variable to binary
data_clv["num"] = data_clv["num"].apply(lambda x: 1 if x > 1 else 0)
print(data_clv["num"].max())

# Load Hungarian dataset
data_hung = pd.read_csv("../datasets/processed.hungarian.data.csv", delimiter=",")
print(data_hung.columns)

# Drop specific columns
data_hung.drop(['ca','thal'], axis=1, inplace=True)
print(data_hung.head())

# Check number of '?' values in 'slope' column
n = (data_hung["slope"] == "?").sum()
print(n)

# Replace '?' with NaN
data_hung["slope"].replace("?", np.nan, inplace=True)

# Convert to numeric, coercing errors to NaN
data_hung = data_hung.apply(pd.to_numeric, errors='coerce')

# Split data into known and missing slope data
data_known = data_hung[data_hung["slope"].notna()]
data_missing = data_hung[data_hung["slope"].isna()]

# Prepare data for slope prediction
X = data_known.drop(columns=["slope", "num"])
y = data_known["slope"]

# Split data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train HistGradientBoostingClassifier to predict missing slope values
model = HistGradientBoostingClassifier()
model.fit(X_train, y_train)

# Predict missing slope values
data_missing["slope"] = model.predict(data_missing.drop(columns=["slope", "num"]))

# Combine known and predicted data
data_hung = pd.concat([data_known, data_missing])
data_hung = data_hung.sort_index()

# Check null values
print(data_clv.isnull().sum())
print(data_hung.isnull().sum())

# Handle missing values in Hungarian dataset
data_hung["chol"].fillna(data_hung["chol"].median(), inplace=True)
data_hung["fbs"].fillna(data_hung["fbs"].mode()[0], inplace=True)
data_hung.dropna(subset=["trestbps", "restecg", "thalach", "exang"], inplace=True)
data_hung = data_hung.sort_index()

# Verify null values are handled
print(data_hung.isnull().sum())

# Align column names
data_clv.columns = data_hung.columns

# Combine datasets
combined_data = pd.concat([data_clv, data_hung], ignore_index=True)

# Check null values in combined dataset
print(combined_data.isnull().sum())

# Save combined dataset
combined_data.to_csv("../datasets/combined_heart_data.csv", index=False)