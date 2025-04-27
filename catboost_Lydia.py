from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -----------------------------
# Preprocessing Functions
# -----------------------------
def plot_kmeans_clusters(df, params, title="KMeans Clustering"):
    plt.figure(figsize=(6, 4))
    plt.scatter(df['X'], df['Y'], c=df['cluster'], cmap='tab20', alpha=0.6)
    plt.colorbar(label="Cluster")

    centroids = params['kmeans'].cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label="Centroids")

    for i, (x, y) in enumerate(centroids):
        plt.text(x, y, str(i), fontsize=12, color='black', ha='center', va='center', fontweight='bold')

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(title)
    plt.legend()
    plt.show()

def preprocess_df(in_df, train=True, params=None):
    df = in_df.copy()
    if train:
        params = {}

    exclude_cols = ['AREA', 'CONTRUCTIONYEAR', 'MAXBUILDINGFLOOR', 'CADASTRALQUALITYID']
    numerical_cols = df.select_dtypes(include=['number']).columns
    numerical_cols = numerical_cols[~numerical_cols.isin(exclude_cols)].tolist()

    if train:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        params['scaler'] = scaler
    else:
        df[numerical_cols] = params['scaler'].transform(df[numerical_cols])

    for color in ['R', 'G', 'B', 'NIR']:
        cols = [col for col in df.columns if f'Q_{color}_' in col]
        df[f'Q_{color}_AVG'] = df[cols].mean(axis=1)
        df[f'Q_{color}_STD'] = df[cols].std(axis=1)
        df[f'Q_{color}_MAX'] = df[cols].max(axis=1)
        df[f'Q_{color}_MIN'] = df[cols].min(axis=1)

    if train:
        n_clusters = 20
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(df[['X', 'Y']])
        params['kmeans'] = kmeans
        params['n_clusters'] = n_clusters
    else:
        df['cluster'] = params['kmeans'].predict(df[['X', 'Y']])

    title = "KMeans Clustering (Training)" if train else "KMeans Clustering (Testing)"
    plot_kmeans_clusters(df, params, title=title)

    distances = cdist(df[['X', 'Y']], params['kmeans'].cluster_centers_, metric='euclidean')
    for i in range(params['n_clusters']):
        df[f'dist_X_Y_centroid_{(i + 1):02d}'] = distances[:, i]

    if train:
        params['MAXBUILDINGFLOOR_mode'] = round(df["MAXBUILDINGFLOOR"].mean())
    df.fillna({"MAXBUILDINGFLOOR": params['MAXBUILDINGFLOOR_mode']}, inplace=True)

    df['CADASTRALQUALITYID'] = df['CADASTRALQUALITYID'].replace({'A': 10, 'B': 11, 'C': 12})
    df['CADASTRALQUALITYID'] = pd.to_numeric(df['CADASTRALQUALITYID'])

    df['BUILDING_AGE'] = datetime.now().year - df['CONTRUCTIONYEAR']
    df['RT_AREA_FLOORS'] = df['AREA'] / df['MAXBUILDINGFLOOR']
    df['GEOM_RT_R1_R2'] = df['GEOM_R1'] / df['GEOM_R2']
    df['GEOM_RT_R1_R3'] = df['GEOM_R1'] / df['GEOM_R3']
    df['GEOM_RT_R1_R4'] = df['GEOM_R1'] / df['GEOM_R4']
    df['GEOM_RT_R2_R3'] = df['GEOM_R2'] / df['GEOM_R3']
    df['GEOM_RT_R2_R4'] = df['GEOM_R2'] / df['GEOM_R4']
    df['GEOM_RT_R3_R4'] = df['GEOM_R3'] / df['GEOM_R4']
    df['X_Y_INTERACT'] = df['X'] * df['Y']

    return (df, params) if train else df

def remove_highly_correlated_features(df, features, threshold=0.90, plot=True):
    corr_matrix = df[features].corr().abs()

    if plot:
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
        plt.title("Correlation Matrix - Original Features")
        plt.show()

    high_corr_pairs = np.where(corr_matrix > threshold)
    high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y])
                       for x, y in zip(*high_corr_pairs) if x != y and x < y]

    corr_groups = {}
    for feat1, feat2 in high_corr_pairs:
        if feat1 not in corr_groups and feat2 not in corr_groups:
            corr_groups[feat1] = {feat1, feat2}
        else:
            for key in list(corr_groups.keys()):
                if feat1 in corr_groups[key] or feat2 in corr_groups[key]:
                    corr_groups[key].update({feat1, feat2})

    selected_features = set(features)
    dropped_features = set()

    for group in corr_groups.values():
        group = list(group)
        variances = df[group].var()
        best_feature = variances.idxmax()
        group.remove(best_feature)
        selected_features -= set(group)
        dropped_features.update(group)

    if plot:
        plt.figure(figsize=(6, 5))
        sns.heatmap(df[list(selected_features)].corr(), cmap="coolwarm", annot=False, fmt=".2f")
        plt.title("Correlation Matrix - After Removing Redundant Features")
        plt.show()

    print(f'Original features: {len(features)}')
    print(f'After running correlation: {len(selected_features)}')
    print(f'Features removed: {len(dropped_features)}')

    return list(selected_features), list(dropped_features)

def check_and_fix_inf_values(df):
    print("🔍 Checking for NaN, inf, or very large values...")
    df = df.apply(pd.to_numeric, errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.columns:
        if df[col].isna().sum() > 0:
            print(f"⚠️ Fixing {df[col].isna().sum()} NaN values in {col}")
            df[col] = df[col].fillna(df[col].median())
    df = df.clip(-1e6, 1e6)
    return df

# -----------------------------
# Main Pipeline
# -----------------------------

print("Loading data sets...")
train = load_dataset("ieuniversity/competition_ai_ml_24_train")
test = load_dataset("ieuniversity/competition_ai_ml_24_test")
df_train = train["train"].to_pandas()

X_train = df_train.drop(['ID', 'CLASE'], axis=1)
y_train = df_train['CLASE']
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# Preprocess
X_train_preprocessed, train_params = preprocess_df(X_train_final, train=True)
X_val_preprocessed = preprocess_df(X_val, train=False, params=train_params)

# Remove correlated features
original_features = [feature for feature in X_train_preprocessed.columns if feature.startswith(("Q_", "GEOM_"))]
reduced_features, dropped_features = remove_highly_correlated_features(X_train_preprocessed, original_features, threshold=0.95)
selected_features = set(X_train_preprocessed.columns) - set(dropped_features)
selected_features = list(selected_features)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_final)
y_val_encoded = label_encoder.transform(y_val)

# Fix inf/nan
X_train_preprocessed = check_and_fix_inf_values(X_train_preprocessed)
X_val_preprocessed = check_and_fix_inf_values(X_val_preprocessed)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_preprocessed[selected_features])
X_val_scaled = scaler.transform(X_val_preprocessed[selected_features])

# Train CatBoost
print("Training CatBoost model...")
cat_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    eval_metric='TotalF1:average=Macro',
    random_seed=42,
    verbose=False
)

train_pool = Pool(X_train_scaled, label=y_train_encoded)
val_pool = Pool(X_val_scaled, label=y_val_encoded)
cat_model.fit(train_pool)

# Evaluate
y_val_pred = cat_model.predict(val_pool).ravel()
val_f1 = f1_score(y_val_encoded, y_val_pred, average="macro")
print(f"Validation F1 Score: {val_f1:.4f}")

# Test preprocessing
df_test = test["train"].to_pandas()
ids = df_test["ID"].astype(str).values
X_test_preprocessed = preprocess_df(df_test, train=False, params=train_params)
X_test_preprocessed = check_and_fix_inf_values(X_test_preprocessed)
X_test_scaled = scaler.transform(X_test_preprocessed[selected_features].drop(columns=["ID"], errors="ignore"))
test_pool = Pool(X_test_scaled)

# Predict and export
predictions = cat_model.predict(test_pool).ravel()
decoded_predictions = label_encoder.inverse_transform(predictions.astype(int))

submission_df = pd.DataFrame({
    "ID": ids,
    "CLASE": decoded_predictions
})
submission_filename = f"submission_catboost_{datetime.now():%Y%m%d_%H%M%S}.csv"
submission_df.to_csv(submission_filename, index=False)
print(f"✅ Submission file saved as {submission_filename}")

