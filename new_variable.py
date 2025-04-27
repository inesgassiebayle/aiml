# === Imports ===
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cdist
from xgboost import XGBClassifier
from datasets import load_dataset
from scipy.interpolate import interp1d
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from sklearn.model_selection import StratifiedKFold, cross_val_score

import geopandas as gpd
import osmnx as ox

# === Step 1: Load Dataset ===
print("Loading datasets...")
train = load_dataset("ieuniversity/competition_ai_ml_24_train")
test = load_dataset("ieuniversity/competition_ai_ml_24_test")
df_train = train["train"].to_pandas()
df_test = test["train"].to_pandas()

# Feature Selection
X_train = df_train.drop(['ID', 'CLASE'], axis=1)
y_train = df_train['CLASE']

# Encode categorical target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

# Split in train-val
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# === Step 2: Coordinate Correction (Train & Test) ===
wrong_coords_raw = [
    ("2.137485B", "165.8023M"),
    ("2.209819B", "165.9628M"),
    ("2.27992B",  "165.8857M"),
    ("2.211186B", "165.4501M")
]
real_coords = np.array([
    [-3.837412, 40.47574],
    [-3.682961, 40.51706],
    [-3.54207,  40.48164],
    [-3.67857,  40.37793]
])
def convert_wrong_coords(coords_raw):
    return np.array([
        (float(x[:-1]) * (1e9 if x.endswith("B") else 1e6),
         float(y[:-1]) * (1e9 if y.endswith("B") else 1e6))
        for x, y in coords_raw
    ])
wrong_coords = convert_wrong_coords(wrong_coords_raw)
interp_long = interp1d(wrong_coords[:, 0], real_coords[:, 0], fill_value="extrapolate")
interp_lat = interp1d(wrong_coords[:, 1], real_coords[:, 1], fill_value="extrapolate")
def transform_column_interp(value, interp_fn):
    if isinstance(value, str):
        value = float(value.strip()[:-1]) * (1e9 if value.endswith("B") else 1e6)
    return interp_fn(value)
for df in [df_train, df_test]:
    df["X"] = df["X"].apply(lambda x: transform_column_interp(x, interp_long))
    df["Y"] = df["Y"].apply(lambda y: transform_column_interp(y, interp_lat))

# === Step 3: Enrich with Street Features from OSM ===
def enrich_with_street_features(df, roads_proj):
    df["geometry"] = df.apply(lambda row: Point(row["X"], row["Y"]), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf_proj = gdf.to_crs(roads_proj.crs)

    nearest_idx = gdf_proj.geometry.apply(lambda geom: roads_proj.distance(geom).idxmin())
    nearest_roads = roads_proj.loc[nearest_idx].reset_index(drop=True)

    gdf["street_name"] = nearest_roads["name"].fillna("unknown")
    gdf["highway_type"] = nearest_roads["highway"]
    gdf["dist_to_street"] = gdf_proj.geometry.reset_index(drop=True).distance(nearest_roads.geometry)
    gdf = pd.get_dummies(gdf, columns=["highway_type"], prefix="roadtype")

    return gdf.drop(columns=["geometry"])

# Download OSM data
center_coords = (df_train["Y"].mean(), df_train["X"].mean())
road_graph = ox.graph_from_point(center_coords, dist=3000, network_type='drive')
roads = ox.graph_to_gdfs(road_graph, nodes=False)
roads_proj = roads.to_crs(epsg=3857)

# Enrich both datasets
df_train = enrich_with_street_features(df_train, roads_proj)
df_test = enrich_with_street_features(df_test, roads_proj)

# === Step 4: Label Encoding and Splitting ===
y_train = LabelEncoder().fit_transform(df_train["CLASE"])
X_train = df_train.drop(columns=["ID", "CLASE"])
X_test = df_test.drop(columns=["ID"])

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# === Step 5: Preprocessing ===
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
        kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(df[['X', 'Y']])
        params['kmeans'] = kmeans
    else:
        df['cluster'] = params['kmeans'].predict(df[['X', 'Y']])

    distances = cdist(df[['X', 'Y']], params['kmeans'].cluster_centers_, metric='euclidean')
    for i in range(20):
        df[f'dist_X_Y_centroid_{i+1:02d}'] = distances[:, i]

    df['CADASTRALQUALITYID'] = df['CADASTRALQUALITYID'].replace({'A': 10, 'B': 11, 'C': 12})
    df['CADASTRALQUALITYID'] = pd.to_numeric(df['CADASTRALQUALITYID'])
    if train:
        params['MAXBUILDINGFLOOR_mean'] = round(df["MAXBUILDINGFLOOR"].mean())
        params['CADASTRALQUALITYID_mode'] = df["CADASTRALQUALITYID"].mode()[0]

    df.fillna({
        "MAXBUILDINGFLOOR": params['MAXBUILDINGFLOOR_mean'],
        "CADASTRALQUALITYID": params['CADASTRALQUALITYID_mode']
    }, inplace=True)

    df['MAXBUILDINGFLOOR'] = (df['MAXBUILDINGFLOOR'] == 0).astype(int)
    df['CADASTRALQUALITYID'] = df['CADASTRALQUALITYID'].astype('category')
    df = pd.get_dummies(df, columns=['CADASTRALQUALITYID'])

    df['BUILDING_AGE'] = datetime.now().year - df['CONTRUCTIONYEAR']
    df['RT_AREA_FLOORS'] = df['AREA'] / df['MAXBUILDINGFLOOR']
    df['GEOM_RT_R1_R2'] = df['GEOM_R1'] / df['GEOM_R2']
    df['GEOM_RT_R1_R3'] = df['GEOM_R1'] / df['GEOM_R3']
    df['GEOM_RT_R1_R4'] = df['GEOM_R1'] / df['GEOM_R4']
    df['GEOM_RT_R2_R3'] = df['GEOM_R2'] / df['GEOM_R3']
    df['GEOM_RT_R2_R4'] = df['GEOM_R2'] / df['GEOM_R4']
    df['GEOM_RT_R3_R4'] = df['GEOM_R3'] / df['GEOM_R4']

    return (df, params) if train else df

X_train_preprocessed, train_params = preprocess_df(X_train_final, train=True)
X_val_preprocessed = preprocess_df(X_val, train=False, params=train_params)
X_test_preprocessed = preprocess_df(X_test, train=False, params=train_params)

def remove_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features to drop
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    df = df.drop(columns=to_drop)
    print(f"After correlation filtering: {df.shape[1]} features")
    print(", ".join(df.columns)+"\n")
    return list(df.columns)

def select_important_features(df, y, keep=0.8, n_top=None):
    model = XGBClassifier(n_estimators=500, objective="multi:softmax", eval_metric='mlogloss', num_class=len(label_encoder.classes_), random_state=42)
    model.fit(df, y)

    feature_importances = pd.Series(model.feature_importances_, index=df.columns)
    sorted_features = feature_importances.sort_values(ascending=False)

    if n_top is None:
        n_top = int(len(sorted_features) * keep)

    selected_features = sorted_features.iloc[:n_top].index
    df = df[selected_features]
    print(f"After XGBoost selection: {df.shape[1]} features")
    print(", ".join(df.columns)+"\n")
    return list(df.columns)

X_train_preprocessed, train_params = preprocess_df(X_train_final, train=True)
X_val_preprocessed = preprocess_df(X_val, train=False, params=train_params)

print(X_train_preprocessed.shape)

# Feature Selection
print(f"Original features: {len(X_train_preprocessed.columns)}")
print(", ".join(list(X_train_preprocessed.columns))+"\n")

# Step 1: Remove correlated features
selected_features = remove_highly_correlated_features(X_train_preprocessed)

# Step 2: Feature selection with XGBoost
selected_features = select_important_features(X_train_preprocessed[selected_features], y_train_final, keep=0.80)

# Step 3: Inclusion business logic
selected_features += ['GEOM_R4','CADASTRALQUALITYID_1.0','CADASTRALQUALITYID_5.0','CADASTRALQUALITYID_10.0','CADASTRALQUALITYID_11.0']
print(", ".join(selected_features)+"\n")

X_test_preprocessed = preprocess_df(X_test, train=False, params=train_params)
X_test_final = X_test_preprocessed[selected_features]

rf_optimized = RandomForestClassifier(
    n_estimators=700, max_depth=20, min_samples_split=5,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf_optimized.fit(X_train_preprocessed[selected_features], y_train_final)

# Predict on Training Set
y_train_pred = rf_optimized.predict(X_train_preprocessed[selected_features])
train_f1 = f1_score(y_train_final, y_train_pred, average="macro")
print(f"Training F1 Score: {train_f1:.4f}")

# Predict on Validation Set
y_val_pred = rf_optimized.predict(X_val_preprocessed[selected_features])
val_f1 = f1_score(y_val, y_val_pred, average="macro")
print(f"Validation F1 Score: {val_f1:.4f}")
