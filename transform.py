import pandas as pd
import geopandas as gpd

# === Step 1: Descargar la geometría de Madrid directamente ===
import osmnx.features as oxf

print("Descargando geometría de Madrid...")
gdf_distritos = oxf.features_from_place(
    "Madrid, Spain",
    tags={"admin_level": "9"}  # Distritos
)
print(gdf_distritos.head())


print(f"Descargados {len(gdf_distritos)} polígonos.")

# === Step 2: Tu dataset ===
from datasets import load_dataset

train = load_dataset("ieuniversity/competition_ai_ml_24_train")
df_train = train["train"].to_pandas()

# Conversión de X y Y
def convert_coord_string(val):
    if isinstance(val, str):
        val = val.strip()
        if val.endswith("B"):
            return float(val[:-1]) * 1e9
        elif val.endswith("M"):
            return float(val[:-1]) * 1e6
        return float(val)
    return float(val)

df_train["X"] = df_train["X"].apply(convert_coord_string)
df_train["Y"] = df_train["Y"].apply(convert_coord_string)

# Modelos de regresión para corregir lat/lon
from sklearn.linear_model import LinearRegression

puntos_clave = pd.DataFrame({
    'X': [2137485000, 2209819000, 2279920000, 2211186000],
    'Y': [165802300, 165962800, 165885700, 165450100],
    'lat_real': [40.47574, 40.51706, 40.48164, 40.37793],
    'lon_real': [-3.837412, -3.682961, -3.54207, -3.67857]
})

X = puntos_clave[['X', 'Y']]
modelo_lat = LinearRegression().fit(X, puntos_clave['lat_real'])
modelo_lon = LinearRegression().fit(X, puntos_clave['lon_real'])

df_train["lat"] = modelo_lat.predict(df_train[["X", "Y"]])
df_train["lon"] = modelo_lon.predict(df_train[["X", "Y"]])

# === Step 3: Crear GeoDataFrame de tus puntos ===
gdf_points = gpd.GeoDataFrame(
    df_train,
    geometry=gpd.points_from_xy(df_train.lon, df_train.lat),
    crs="EPSG:4326"
)

# === Step 4: Spatial join ===
print("Realizando unión espacial...")
gdf_points_with_districts = gpd.sjoin(gdf_points, gdf_distritos, how="left", predicate="within")

# === Step 5: Asignar nombre del barrio o distrito ===
# En OpenStreetMap puede estar en el campo "name"
df_train["distrito"] = gdf_points_with_districts["name"]

print("¡Distritos asignados!")

# (Opcional) Mapa
import plotly.express as px

df_plot = df_train.dropna(subset=["lat", "lon"])

fig = px.scatter_mapbox(
    df_plot,
    lat="lat",
    lon="lon",
    hover_name="ID",
    hover_data=["distrito"],
    zoom=10,
    center={"lat": 40.4168, "lon": -3.7038},
    mapbox_style="carto-positron",
    height=600,
    title="Puntos geográficos estimados sobre Madrid con distritos (OSM)"
)

fig.show()

# Mostrar ID, latitud, longitud y distrito de los primeros 5 datos
print(df_train[["ID", "lat", "lon", "distrito"]].head())
