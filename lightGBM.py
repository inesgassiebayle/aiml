from lightgbm import LGBMClassifier
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import f1_score, make_scorer, ConfusionMatrixDisplay, classification_report, confusion_matrix
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

# Entrenar modelo LightGBM
print("Training LightGBM model...")
lgbm = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

lgbm.fit(X_train_scaled, y_train_final)

# Evaluar en el conjunto de validación
y_val_pred_lgbm = lgbm.predict(X_val_scaled)
val_f1_lgbm = f1_score(y_val, y_val_pred_lgbm, average="macro")

print(f"Validation F1 Score (LightGBM): {val_f1_lgbm:.4f}")

# Mostrar reporte de clasificación
print("Classification Report:")
print(classification_report(y_val, y_val_pred_lgbm))

# Matriz de confusión
cm_lgbm = confusion_matrix(y_val, y_val_pred_lgbm)
ConfusionMatrixDisplay(confusion_matrix=cm_lgbm).plot()
plt.show()

# Hacer predicciones en el conjunto de test
y_test_pred_lgbm = lgbm.predict(X_test_scaled)