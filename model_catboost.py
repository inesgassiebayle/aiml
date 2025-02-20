# -*- coding: utf-8 -*-
from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

# Load Data
print("Loading data sets...")
train = load_dataset("ieuniversity/competition_ai_ml_24_train")
test = load_dataset("ieuniversity/competition_ai_ml_24_test")
df_train = train["train"].to_pandas()

# Clean Data
print("Cleaning data sets...")
df_train.fillna({"CADASTRALQUALITYID": "None"}, inplace=True)
df_train.fillna({"MAXBUILDINGFLOOR": df_train["MAXBUILDINGFLOOR"].mean()}, inplace=True)

# Feature Selection
X_train = df_train.drop(["ID", "CLASE"], axis=1)
X_train = pd.get_dummies(X_train)
y_train = df_train["CLASE"]

X_test = test["train"].to_pandas()
X_test.fillna({"CADASTRALQUALITYID": "None"}, inplace=True)
X_test.fillna({"MAXBUILDINGFLOOR": df_train["MAXBUILDINGFLOOR"].mean()}, inplace=True)
X_test = X_test.drop(["ID"], axis=1)
X_test = pd.get_dummies(X_test)

# Ensure train and test have the same features
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Encode Target Variable
y_encoder = LabelEncoder()
y_train = y_encoder.fit_transform(y_train)

# Split Data
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=100
)

model.fit(X_train_scaled, y_train_final)

# Evaluate on Validation Set
y_val_pred = model.predict(X_val_scaled)
val_f1 = f1_score(y_val, y_val_pred, average="macro")
print(f"Validation F1 Score: {val_f1:.4f}")