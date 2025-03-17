from xgboost import XGBClassifier
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

# Load Data
print("Loading data sets...")
train = load_dataset("ieuniversity/competition_ai_ml_24_train")
test = load_dataset("ieuniversity/competition_ai_ml_24_test")
df_train = train["train"].to_pandas()

# Feature Selection
X_train = df_train.drop(['ID', 'CLASE'], axis=1)
y_train = df_train['CLASE']
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

def plot_kmeans_clusters(df, params, title="KMeans Clustering"):
    plt.figure(figsize=(6, 4))
    plt.scatter(df['X'], df['Y'], c=df['cluster'], cmap='tab20', alpha=0.6)
    plt.colorbar(label="Cluster")

    # Get and plot centroids
    centroids = params['kmeans'].cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label="Centroids")

    # Add cluster number to the plot
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

    # Feature Scaling
    exclude_cols = ['AREA', 'CONTRUCTIONYEAR', 'MAXBUILDINGFLOOR', 'CADASTRALQUALITYID']
    numerical_cols = df.select_dtypes(include=['number']).columns
    numerical_cols = numerical_cols[~numerical_cols.isin(exclude_cols)].tolist()

    if train:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        params['scaler'] = scaler
    else:
        df[numerical_cols] = params['scaler'].transform(df[numerical_cols])

    # Feature engineering for colors
    for color in ['R', 'G', 'B', 'NIR']:
        cols = [col for col in df.columns if f'Q_{color}_' in col]
        df[f'Q_{color}_AVG'] = df[cols].mean(axis=1)
        df[f'Q_{color}_STD'] = df[cols].std(axis=1)
        df[f'Q_{color}_MAX'] = df[cols].max(axis=1)
        df[f'Q_{color}_MIN'] = df[cols].min(axis=1)

    # KMeans Clustering
    if train:
        n_clusters = 20
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(df[['X', 'Y']])
        params['kmeans'] = kmeans
        params['n_clusters'] = n_clusters
    else:
        df['cluster'] = params['kmeans'].predict(df[['X', 'Y']])

    # Plot and label the clusters
    title = "KMeans Clustering (Training)" if train else "KMeans Clustering (Testing)"
    plot_kmeans_clusters(df, params, title=title)

    # Create distance variables (Centroides)
    distances = cdist(df[['X', 'Y']], params['kmeans'].cluster_centers_, metric='euclidean')
    for i in range(params['n_clusters']):
        df[f'dist_X_Y_centroid_{(i + 1):02d}'] = distances[:, i]

    # Missing imputation
    if train:
        params['MAXBUILDINGFLOOR_mode'] = round(df["MAXBUILDINGFLOOR"].mean())
    df.fillna({"MAXBUILDINGFLOOR": params['MAXBUILDINGFLOOR_mode']}, inplace=True)

    # Replace letters to numerical codes
    df['CADASTRALQUALITYID'] = df['CADASTRALQUALITYID'].replace({'A': 10, 'B': 11, 'C': 12})
    df['CADASTRALQUALITYID'] = pd.to_numeric(df['CADASTRALQUALITYID'])

    # Aditional features
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
    # Compute correlation matrix
    corr_matrix = df[features].corr().abs()

    if plot:
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
        plt.title("Correlation Matrix - Original Features")
        plt.show()

    # Identify highly correlated feature pairs
    high_corr_pairs = np.where(corr_matrix > threshold)
    high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y])
                       for x, y in zip(*high_corr_pairs) if x != y and x < y]

    # Group features based on correlation
    corr_groups = {}
    for feat1, feat2 in high_corr_pairs:
        if feat1 not in corr_groups and feat2 not in corr_groups:
            corr_groups[feat1] = {feat1, feat2}
        else:
            for key in list(corr_groups.keys()):
                if feat1 in corr_groups[key] or feat2 in corr_groups[key]:
                    corr_groups[key].update({feat1, feat2})

    # Select the best feature from each correlated group (highest variance)
    selected_features = set(features)
    dropped_features = set()

    for group in corr_groups.values():
        group = list(group)
        variances = df[group].var()
        best_feature = variances.idxmax()  # Keep the feature with the highest variance
        group.remove(best_feature)
        selected_features -= set(group)  # Remove redundant features
        dropped_features.update(group)  # Store removed features

    # Display updated correlation matrix if requested
    if plot:
        plt.figure(figsize=(6, 5))
        sns.heatmap(df[list(selected_features)].corr(), cmap="coolwarm", annot=False, fmt=".2f")
        plt.title("Correlation Matrix - After Removing Redundant Features")
        plt.show()

    print(f'Original features: {len(features)}')
    print(features)
    print(f'After running correlation: {len(selected_features)}')
    print(selected_features)
    print(f'Features removed: {len(dropped_features)}')
    print(dropped_features)

    return list(selected_features), list(dropped_features)


def plot_feature_importance(model, feature_names, prefix=None, figsize=(15, 10)):
    # Get feature importance
    feature_importances = model.get_feature_importance()

    # Convert to NumPy arrays
    feature_names = np.array(feature_names)

    # Filter by prefix if provided
    if prefix:
        mask = np.char.startswith(feature_names, prefix)
        filtered_features = feature_names[mask]
        filtered_importances = feature_importances[mask]
    else:
        filtered_features = feature_names
        filtered_importances = feature_importances

    # Sort by importance (descending)
    sorted_idx = np.argsort(filtered_importances)[::-1]
    sorted_features = filtered_features[sorted_idx]
    sorted_importances = filtered_importances[sorted_idx]

    # Print features as a list (copy-paste friendly)
    if len(sorted_features) > 0:
        feature_list_str = ", ".join(f"'{feature}'" for feature in sorted_features)
        print(f"Features with prefix '{prefix}':")
        print(f"[{feature_list_str}]")  # Prints as a valid Python list

        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(x=sorted_importances, y=sorted_features, hue=sorted_features, palette="viridis", legend=False)
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title(f"Feature Importance {f'for Features Starting with {prefix}' if prefix else ''}")
        plt.show()
    else:
        print(f"No features found with prefix '{prefix}'.")


# Usage
X_train_preprocessed, train_params = preprocess_df(X_train_final, train=True)
X_val_preprocessed = preprocess_df(X_val, train=False, params=train_params)

original_features = [feature for feature in X_train_preprocessed.columns if feature.startswith(("Q_", "GEOM_"))]
print(original_features)
reduced_features, dropped_features = remove_highly_correlated_features(X_train_preprocessed, original_features,
                                                                       threshold=0.95)

selected_features = set(X_train_preprocessed.columns)
print(f'Antes de reducir: {len(selected_features)}')
selected_features -= set(dropped_features)
selected_features = list(selected_features)
print(f'Despues de reducir: {len(selected_features)}')


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_final)
y_val_encoded = label_encoder.transform(y_val)

def check_and_fix_inf_values(df):
    """ Identifies and replaces inf, NaN, and very large values in the dataset. """
    print("🔍 Checking for NaN, inf, or very large values...")

    # Convert all columns to numeric if possible (forcing errors to NaN)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Replace inf/-inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaNs with column median
    for col in df.columns:
        if df[col].isna().sum() > 0:
            print(f"⚠️ Fixing {df[col].isna().sum()} NaN values in {col}")
            df[col] = df[col].fillna(df[col].median())

    # Clip extreme values
    df = df.clip(-1e6, 1e6)

    return df  # ✅ Return cleaned DataFrame


# Apply the function
X_train_preprocessed = check_and_fix_inf_values(X_train_preprocessed)
X_val_preprocessed = check_and_fix_inf_values(X_val_preprocessed)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_preprocessed[selected_features])
X_val_scaled = scaler.transform(X_val_preprocessed[selected_features])

# Train XGBoost Classifier
print("Training XGBoost model...")
xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(set(y_train)),
    eval_metric='mlogloss',
    learning_rate=0.1,
    n_estimators=200,
    max_depth=6,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train_encoded)

# Evaluate on Validation Set
y_val_pred = xgb_model.predict(X_val_scaled)
val_f1 = f1_score(y_val_pred, y_val_encoded, average="macro")
print(f"Validation F1 Score: {val_f1:.4f}")
df_test = test["train"].to_pandas()

# ✅ Extract "ID" before any modifications
ids = df_test["ID"].astype(str).values  # Ensure IDs remain as strings (avoids float conversion)

# Preprocess test data
X_test_preprocessed = preprocess_df(df_test, train=False, params=train_params)
X_test_preprocessed = check_and_fix_inf_values(X_test_preprocessed)

# Ensure test data has the same features as training data
X_test_scaled = scaler.transform(X_test_preprocessed[selected_features].drop(columns=["ID"], errors="ignore"))

# Make predictions
predictions = xgb_model.predict(X_test_scaled).ravel()
decoded_predictions = label_encoder.inverse_transform(predictions)

# ✅ Use the original extracted IDs
submission_df = pd.DataFrame({
    "ID": ids,  # ✅ IDs remain unchanged
    "CLASE": decoded_predictions
})

# Save submission file
submission_filename = f"submission_xgboost_{datetime.now()}.csv"
submission_df.to_csv(submission_filename, index=False)

print(f"✅ Submission file saved as {submission_filename}")
