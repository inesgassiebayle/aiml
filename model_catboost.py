# -*- coding: utf-8 -*-
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from datetime import datetime
from sklearn.utils import compute_class_weight
import numpy as np

# Load Data
print("Loading data sets...")
train = load_dataset("ieuniversity/competition_ai_ml_24_train")
test = load_dataset("ieuniversity/competition_ai_ml_24_test")
df_train = train["train"].to_pandas()

# Feature Selection
X_train = df_train.drop(["ID", "CLASE"], axis=1)
y_train = df_train["CLASE"]
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# conclusion 1: imbalaced class
df_train['CLASE'].value_counts().plot(kind = 'bar', ylabel = 'frequency')
plt.show()

def preprocess_df(in_df):
    df = in_df.copy()
    df['CADASTRALQUALITYID'] = df['CADASTRALQUALITYID'].replace({'A': 10, 'B': 11, 'C': 12})
    df['CADASTRALQUALITYID'] = pd.to_numeric(df['CADASTRALQUALITYID'])

    current_year = datetime.now().year
    df['AGE'] = current_year - df['CONTRUCTIONYEAR']

    df['RT_AREA_FLOORS'] = df['AREA'] / df['MAXBUILDINGFLOOR']
    df['MU_AREA_FLOORS'] = df['AREA'] * df['MAXBUILDINGFLOOR']

    # Explotar las coordenadas
    # distancia avg_clase1
    # distancia avg_clase2
    # distancia avg_clase3
    # distancia avg_clase4
    # distancia avg_clase5
    # distancia avg_clase6

    return df

X_train_preprocessed = preprocess_df(X_train_final)
X_val_preprocessed = preprocess_df(X_val)
X_train_preprocessed.head(3)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=8,
    eval_metric='TotalF1',
    objective = 'MultiClass',
    random_seed=42,
    verbose=100
)
model.fit(X_train_preprocessed, y_train_final)

# Evaluate on Validation Set
y_val_pred = model.predict(X_val_preprocessed)
val_f1 = f1_score(y_val, y_val_pred, average="macro")
print(f"Validation F1 Score: {val_f1:.4f}")


"""### Crear CSV para subir a la web"""

# Preprocesar data de test
df_test = test["train"].to_pandas()
X_test_preprocessed = preprocess_df(df_test)

# Crear predicciones y CSV
predictions = model.predict(X_test_preprocessed.drop(["ID"], axis=1)).ravel()
ids = X_test_preprocessed["ID"].values
submission_df = pd.DataFrame({
    "ID": ids,
    "CLASE": predictions
})

# Save the predictions to a CSV file
submission_df.to_csv("submission_catboost.csv", index=False)

"""### Importancia de variables"""

# Feature importance
# Get feature importance
feature_importance = model.get_feature_importance()
feature_names = X_train_preprocessed.columns  # Assuming X_train_preprocessed is a DataFrame

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Print feature importance
print(importance_df)

# Plotting feature importance
plt.figure(figsize=(10, 10))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()