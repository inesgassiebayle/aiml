from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
from datasets import load_dataset

# Load Data
print("Loading data sets...")
train = load_dataset("ieuniversity/competition_ai_ml_24_train")
test = load_dataset("ieuniversity/competition_ai_ml_24_test")

df_train = train["train"].to_pandas()
df_test = test["train"].to_pandas()

# Clean Data
print("Cleaning data sets...")
df_train.fillna({"CADASTRALQUALITYID": "None"}, inplace=True)
df_train.fillna({"MAXBUILDINGFLOOR": df_train["MAXBUILDINGFLOOR"].mean()}, inplace=True)

df_test.fillna({"CADASTRALQUALITYID": "None"}, inplace=True)
df_test.fillna({"MAXBUILDINGFLOOR": df_train["MAXBUILDINGFLOOR"].mean()}, inplace=True)

# Feature Selection
X_train = df_train.drop(["ID", "CLASE"], axis=1)
X_train = pd.get_dummies(X_train)
y_train = df_train["CLASE"]

X_test = df_test.drop(["ID"], axis=1)
X_test = pd.get_dummies(X_test)

# Ensure train and test have the same features
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Split Data for Validation
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Function to remove highly correlated features
def remove_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                to_drop.add(corr_matrix.columns[i])

    print(f"Removing {len(to_drop)} highly correlated features.")
    return df.drop(columns=to_drop, errors='ignore'), list(to_drop)

# Remove correlated features from training and apply the same filter to validation and test
X_train_filtered, dropped_features = remove_highly_correlated_features(
    pd.DataFrame(X_train_scaled, columns=X_train_final.columns)
)
X_val_filtered = pd.DataFrame(X_val_scaled, columns=X_train_final.columns)[X_train_filtered.columns]
X_test_filtered = pd.DataFrame(X_test_scaled, columns=X_train_final.columns)[X_train_filtered.columns]

# Train Initial Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_filtered, y_train_final)

# Feature Importance
feature_importances = rf.feature_importances_
feature_names = np.array(X_train_filtered.columns)
sorted_idx = np.argsort(feature_importances)[::-1]
sorted_features = feature_names[sorted_idx]

# Select Top Features
top_n = 50  # Select top 50 most important features
top_features = sorted_features[:top_n]

X_train_final = X_train_filtered[top_features]
X_val_final = X_val_filtered[top_features]
X_test_final = X_test_filtered[top_features]

# Apply RFE (Recursive Feature Elimination)
rfe = RFE(estimator=rf, n_features_to_select=30, step=1)
X_train_rfe = rfe.fit_transform(X_train_final, y_train_final)
X_val_rfe = rfe.transform(X_val_final)
X_test_rfe = rfe.transform(X_test_final)

selected_features = X_train_final.columns[rfe.support_]
print(f"Selected {len(selected_features)} best features.")

# Train Optimized Random Forest
rf_optimized = RandomForestClassifier(
    n_estimators=300, max_depth=20, min_samples_split=5,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf_optimized.fit(X_train_rfe, y_train_final)

# Evaluate Initial Optimized Model
y_val_pred = rf_optimized.predict(X_val_rfe)
val_f1 = f1_score(y_val, y_val_pred, average="macro")
print(f"Optimized Validation F1 Score: {val_f1:.4f}")

# Hyperparameter Tuning with RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    rf_optimized, param_distributions=param_grid,
    scoring="f1_macro", n_iter=10, cv=3,
    verbose=2, random_state=42, n_jobs=-1
)

random_search.fit(X_train_rfe, y_train_final)

# Get Best Model Parameters
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Train Final Model with Best Hyperparameters
final_rf = RandomForestClassifier(**best_params)
final_rf.fit(X_train_rfe, y_train_final)

# Evaluate Final Model
y_val_pred = final_rf.predict(X_val_rfe)
final_f1 = f1_score(y_val, y_val_pred, average="macro")

# Keep Best Model
if final_f1 < val_f1:
    print("Warning: Tuned model performed worse, keeping the original one.")
    final_rf = rf_optimized
    final_f1 = val_f1

print(f"Final Optimized Random Forest F1 Score: {final_f1:.4f}")

# Make Predictions
predictions = final_rf.predict(X_test_rfe)

# Get Test IDs
test_ids = df_test["ID"].values

# Create Submission File
submission_filename = f"submission_random_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

try:
    submission_df = pd.DataFrame({
        "ID": test_ids,
        "CLASE": predictions
    })
    submission_df.to_csv(submission_filename, index=False)
    print(f"Submission file saved as: {submission_filename}")
except Exception as e:
    print(f"Error saving CSV: {e}")
