from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

# Load Data
print("Loading data sets...")
train = load_dataset("ieuniversity/competition_ai_ml_24_train")
test = load_dataset("ieuniversity/competition_ai_ml_24_test")

df_train = train["train"].to_pandas()

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

# Train XGBoost Classifier
print("Training XGBoost model...")
xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(set(y_train)),
    eval_metric='mlogloss',
    use_label_encoder=False,
    learning_rate=0.1,
    n_estimators=200,
    max_depth=6,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train_final)

# Evaluate on Validation Set
y_val_pred = xgb_model.predict(X_val_scaled)
val_f1 = f1_score(y_val, y_val_pred, average="macro")
print(f"Validation F1 Score: {val_f1:.4f}")

# Predict on Test Set
y_test_pred = xgb_model.predict(X_test_scaled)
y_test_pred_labels = y_encoder.inverse_transform(y_test_pred)


