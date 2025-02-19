from sklearn.linear_model import LogisticRegressionCV
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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

# Split data
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define Weighted LDA with AdaBoost
lda = LinearDiscriminantAnalysis()
boosted_lda = AdaBoostClassifier(base_estimator=lda, n_estimators=50, learning_rate=1.0, algorithm="SAMME")

# Train Boosted LDA
boosted_lda.fit(X_train_scaled, y_train_final)

# Evaluate Model
y_val_pred = boosted_lda.predict(X_val_scaled)
val_f1 = f1_score(y_val, y_val_pred, average="macro")

print(f"Boosted LDA Validation F1 Score: {val_f1:.4f}")

# Predict on Test Set
y_test_pred = boosted_lda.predict(X_test_scaled)

