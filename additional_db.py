import pandas as pd
import geopandas as gpd
import zipfile
import os
from unidecode import unidecode
from datasets import load_dataset
from sklearn.linear_model import LinearRegression
import osmnx.features as oxf


# ========== 1. Funciones auxiliares ==========

def convert_coord_string(val):
    if isinstance(val, str):
        val = val.strip()
        if val.endswith("B"):
            return float(val[:-1]) * 1e9
        elif val.endswith("M"):
            return float(val[:-1]) * 1e6
        return float(val)
    return float(val)


def enrich_dataset(df):
    # Convert X and Y
    df["X"] = df["X"].apply(convert_coord_string)
    df["Y"] = df["Y"].apply(convert_coord_string)

    # Predict lat and lon
    df["lat"] = modelo_lat.predict(df[["X", "Y"]])
    df["lon"] = modelo_lon.predict(df[["X", "Y"]])

    # Create GeoDataFrame
    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326"
    )

    # Spatial join with districts
    gdf_points_with_districts = gpd.sjoin(gdf_points, gdf_distritos, how="left", predicate="within")
    df["distrito"] = gdf_points_with_districts["name"]
    df["distrito_clean"] = df["distrito"].apply(lambda x: unidecode(str(x)).upper().strip())

    # Merge with all additional datasets
    for dataset in datasets_to_merge:
        df = df.merge(dataset, how="left", on="distrito_clean")

    # Fill NaN with 0 in added features
    for col in columns_to_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# ========== 2. Preparar referencias para lat/lon ==========

puntos_clave = pd.DataFrame({
    'X': [2137485000, 2209819000, 2279920000, 2211186000],
    'Y': [165802300, 165962800, 165885700, 165450100],
    'lat_real': [40.47574, 40.51706, 40.48164, 40.37793],
    'lon_real': [-3.837412, -3.682961, -3.54207, -3.67857]
})

modelo_lat = LinearRegression().fit(puntos_clave[['X', 'Y']], puntos_clave['lat_real'])
modelo_lon = LinearRegression().fit(puntos_clave[['X', 'Y']], puntos_clave['lon_real'])

# ========== 3. Descargar distritos ==========

print("Descargando distritos de Madrid...")
gdf_distritos = oxf.features_from_place(
    "Madrid, Spain",
    tags={"admin_level": "9"}
)

# ========== 4. Cargar datasets adicionales ==========

datasets_to_merge = []
columns_to_fill = []

# -- Aceras
if not os.path.exists("aceras_2024"):
    with zipfile.ZipFile("ACCESIBILIDAD_ACERAS_2024.zip", 'r') as zip_ref:
        zip_ref.extractall("aceras_2024")
gdf_aceras = gpd.read_file([os.path.join("aceras_2024", f) for f in os.listdir("aceras_2024") if f.endswith('.shp')][0])
gdf_aceras["distrito_clean"] = gdf_aceras["distritotx"].apply(lambda x: x.split(" ", 1)[1] if " " in x else x)
columna_supacera = [col for col in gdf_aceras.columns if 'supacera' in col.lower()][0]
aceras_por_distrito = gdf_aceras.groupby("distrito_clean").agg(
    num_aceras=("geometry", "count"),
    superficie_total_acera=(columna_supacera, "sum")
).reset_index()
datasets_to_merge.append(aceras_por_distrito)
columns_to_fill.extend(["num_aceras", "superficie_total_acera"])

# -- Áreas Infantiles
df_areas = pd.read_csv("areas_infantiles202504.csv", sep=";")
df_areas["distrito_clean"] = df_areas["DISTRITO"].str.upper().str.strip()
areas_por_distrito = df_areas.groupby("distrito_clean").agg(
    total_areas_infantiles=("ID", "count")
).reset_index()
datasets_to_merge.append(areas_por_distrito)
columns_to_fill.append("total_areas_infantiles")

# -- Aseos Públicos
df_aseos = pd.read_csv("Aseos Publicos Operativos.csv", sep=";")
df_aseos["distrito_clean"] = df_aseos["DISTRITO"].str.upper().str.strip()
aseos_por_distrito = df_aseos.groupby("distrito_clean").agg(
    total_aseos_publicos=("DISTRITO", "count")
).reset_index()
datasets_to_merge.append(aseos_por_distrito)
columns_to_fill.append("total_aseos_publicos")

# -- Bibliotecas
df_bibliotecas = pd.read_csv("201747-0-bibliobuses-bibliotecas.csv", sep=";", encoding="latin1")
df_bibliotecas["distrito_clean"] = df_bibliotecas["DISTRITO"].str.upper().str.strip()
bibliotecas_por_distrito = df_bibliotecas.groupby("distrito_clean").agg(
    total_bibliotecas=("DISTRITO", "count")
).reset_index()
datasets_to_merge.append(bibliotecas_por_distrito)
columns_to_fill.append("total_bibliotecas")

# -- Centros Educativos
df_centros = pd.read_csv("300614-0-centros-educativos.csv", sep=";", encoding="latin1")
df_centros["distrito_clean"] = df_centros["DISTRITO"].apply(lambda x: unidecode(str(x)).upper().strip())
centros_por_distrito = df_centros.groupby("distrito_clean").agg(
    total_centros_educativos=("DISTRITO", "count")
).reset_index()
datasets_to_merge.append(centros_por_distrito)
columns_to_fill.append("total_centros_educativos")

# -- Centros Mayores
df_mayores = pd.read_csv("200337-0-centros-mayores.csv", sep=";", encoding="latin1")
df_mayores["distrito_clean"] = df_mayores["DISTRITO"].apply(lambda x: unidecode(str(x)).upper().strip())
mayores_por_distrito = df_mayores.groupby("distrito_clean").agg(
    total_centros_mayores=("DISTRITO", "count")
).reset_index()
datasets_to_merge.append(mayores_por_distrito)
columns_to_fill.append("total_centros_mayores")

# -- Colegios Públicos
df_colegios = pd.read_csv("202311-0-colegios-publicos.csv", sep=";", encoding="latin1")
df_colegios["distrito_clean"] = df_colegios["DISTRITO"].apply(lambda x: unidecode(str(x)).upper().strip())
colegios_por_distrito = df_colegios.groupby("distrito_clean").agg(
    total_colegios_publicos=("DISTRITO", "count")
).reset_index()
datasets_to_merge.append(colegios_por_distrito)
columns_to_fill.append("total_colegios_publicos")

# ========== 5. Cargar y enriquecer train y test ==========

print("Cargando train y test...")
train = load_dataset("ieuniversity/competition_ai_ml_24_train")
test = load_dataset("ieuniversity/competition_ai_ml_24_test")

df_train = train["train"].to_pandas()
df_test = test["train"].to_pandas()

print("Enriqueciendo train...")
df_train = enrich_dataset(df_train)
print("Enriqueciendo test...")
df_test = enrich_dataset(df_test)

# ========== 6. Guardar datasets enriquecidos ==========

print("Guardando CSVs...")

df_train.to_csv("train_enriched.csv", index=False)
df_test.to_csv("test_enriched.csv", index=False)

print("¡CSVs guardados como 'train_enriched.csv' y 'test_enriched.csv'!")
