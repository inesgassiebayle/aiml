from sklearn.linear_model import LogisticRegressionCV
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# Check Correlations
correlation_matrix = X_train.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f',
            vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Split Data for Validation
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
print("Training model...")
scorer = make_scorer(f1_score, average="macro")

logreg = LogisticRegressionCV(
    Cs=[0.01, 0.1, 1, 10],
    cv=3,
    class_weight="balanced",
    n_jobs=-1,
    max_iter=10_000,
    scoring=scorer,
    verbose=1
)

logreg.fit(X_train_scaled, y_train_final)

# Evaluate on Validation Set
y_val_pred = logreg.predict(X_val_scaled)
val_f1 = f1_score(y_val, y_val_pred, average="macro")
print(f"Validation F1 Score: {val_f1:.4f}")

# Predict on Test Set
y_test_pred = logreg.predict(X_test_scaled)


