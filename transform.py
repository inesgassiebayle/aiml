import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.interpolate import interp1d
import plotly.express as px

# Paso 1: Coordenadas incorrectas → reales (usa strings con B y M)
wrong_coords_raw = [
    ("2.137485B", "165.8023M"),
    ("2.209819B", "165.9628M"),
    ("2.27992B",  "165.8857M"),
    ("2.211186B", "165.4501M")
]

# Coordenadas reales correspondientes (longitud, latitud)
real_coords = np.array([
    [-3.837412, 40.47574],
    [-3.682961, 40.51706],
    [-3.54207,  40.48164],
    [-3.67857,  40.37793]
])

# Convertir coordenadas erróneas a float
def convert_wrong_coords(wrong_coords_raw):
    converted = []
    for x, y in wrong_coords_raw:
        x_val = float(x[:-1]) * (1e9 if x.endswith("B") else 1e6)
        y_val = float(y[:-1]) * (1e9 if y.endswith("B") else 1e6)
        converted.append((x_val, y_val))
    return np.array(converted)

wrong_coords = convert_wrong_coords(wrong_coords_raw)

# Crear interpolaciones lineales
interp_long = interp1d(wrong_coords[:, 0], real_coords[:, 0], fill_value="extrapolate")
interp_lat = interp1d(wrong_coords[:, 1], real_coords[:, 1], fill_value="extrapolate")

# Cargar dataset
train = load_dataset("ieuniversity/competition_ai_ml_24_train")
df_train = train["train"].to_pandas()

# Función para transformar cada coordenada usando interpolación
def transform_column_interp(value, interp_fn):
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("B"):
            value = float(value[:-1]) * 1e9
        elif value.endswith("M"):
            value = float(value[:-1]) * 1e6
    else:
        value = float(value)
    return interp_fn(value)

# Aplicar transformación
df_train["X"] = df_train["X"].apply(lambda x: transform_column_interp(x, interp_long))
df_train["Y"] = df_train["Y"].apply(lambda y: transform_column_interp(y, interp_lat))

# Graficar los puntos corregidos
fig = px.scatter(
    df_train,
    x="X",
    y="Y",
    title="Coordenadas Corregidas con Interpolación Lineal",
    labels={"X": "Longitud", "Y": "Latitud"},
    width=800,
    height=600
)

fig.update_traces(
    hovertemplate="Longitud=%{x:.6f}<br>Latitud=%{y:.6f}<extra></extra>"
)

fig.show()
